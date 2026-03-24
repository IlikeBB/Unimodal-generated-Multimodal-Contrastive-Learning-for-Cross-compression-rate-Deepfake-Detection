from __future__ import print_function, division

import gc
import os
import pickle
import re
import sys
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append("model")
sys.path.append("module")
sys.path.append("util")

import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

from util.config import get_args, get_name
from util.dataloader_FF_landmark import get_loader
from util.logger import get_logger

torch.multiprocessing.set_sharing_strategy("file_system")
from module.landmark_model import DualLRNet
from module.rPPG_model import ViT_ST_ST_Compact3_TDC_gra_sharp
from module.base_model import PAD_Classifier
from module.clip import clip  # type: ignore


def load_net(device, rPPG_path=None, LRNet_comp=None, model_text=None, args_set=None):
    if LRNet_comp is None or LRNet_comp == "":
        LRNet_load = False
        LRNet_comp = ""
    else:
        LRNet_load = True

    net_pad = DualLRNet(
        load_pretrained=LRNet_load,
        pretrained_comp=LRNet_comp,
        device=device,
    )

    def load_model(model, path):
        pretrained_state = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_state.items() if k in model.state_dict()}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    net_downstream = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(300, 128, 128),  # type: ignore
        patches=(4, 4, 4),  # type: ignore
        dim=96,
        ff_dim=144,
        num_heads=4,
        num_layers=12,
        dropout_rate=0.1,
        theta=0.7,
    )

    if rPPG_path is not None:
        net_downstream = load_model(net_downstream, rPPG_path)
        print("pretrained rPPG model is loaded")
    else:
        print("No pretrained rPPG model is loaded")

    net = PAD_Classifier(net_pad, net_downstream, model_text, args_set=args_set, training=False)
    net.to(device)
    return net


def build_experiment_name(args) -> str:
    exp_name = f"[{args.train_rate}]{args.exper_name}"
    exp_name += f"[Fix-Prompt_w={args.text_prompt_weight}]"
    exp_name += f"[AffL-a_{args.aff_alpha}_b{args.aff_beta}]"

    if args.warmup:
        exp_name += "[WarmUp]"

    exp_name += f"[phy_w{args.phy_loss_w}-CCTL_w{args.CCTL_w}-AFF_w{args.aff_loss_w}]"
    exp_name += f"[Mask_config-{args.Mask_prob}_{args.Mask_percentage}_{args.text_prompt_filter}_{args.max_mask_count}]"

    if args.classifier_module == "mlp":
        exp_name += f"[Classifier-{args.classifier_module}]"
    return exp_name


def resolve_experiment_name_and_dir(args):
    built_name = build_experiment_name(args)
    built_dir = os.path.abspath(f"./model_results/{built_name}/{args.finetune_dataset}/all/{args.subset}/")
    if os.path.isdir(built_dir):
        return built_name, built_dir

    literal_dir = os.path.abspath(f"./model_results/{args.exper_name}/{args.finetune_dataset}/all/{args.subset}/")
    if os.path.isdir(literal_dir):
        return args.exper_name, literal_dir

    return built_name, built_dir


def find_best_checkpoint(weight_dir, subset):
    best_auc_path = os.path.join(weight_dir, "best_auc.pt")
    if os.path.isfile(best_auc_path):
        return best_auc_path, "best_auc"

    best_model_path = os.path.join(weight_dir, f"best_model_{subset}.pt")
    if os.path.isfile(best_model_path):
        return best_model_path, "best_model"

    raise FileNotFoundError(
        f"No best checkpoint found in {weight_dir}. "
        f"Expected one of: best_auc.pt, best_model_{subset}.pt"
    )


def evaluate_checkpoint(model, test_loader, device, batch_size):
    similar_label = torch.tensor([1], dtype=torch.float).to(device)
    dissimilar_label = torch.tensor([-1], dtype=torch.float).to(device)

    true_labels = []
    predicted_scores = []
    subjects = []

    true_positive = 1e-7
    true_negative = 1e-7
    false_positive = 1e-7
    false_negative = 1e-7

    with torch.no_grad():
        for face_images, facial_landmarks, landmark_differences, ground_truth_labels, sample_subjects in test_loader:
            if face_images.shape[0] != batch_size:
                print(f"Batch size mismatch: {face_images.shape[0]=}, skipping this batch")
                continue

            face_images = face_images.to(device)
            facial_landmarks = facial_landmarks.to(device)
            landmark_differences = landmark_differences.to(device)
            ground_truth_labels = ground_truth_labels.to(device)

            for idx in range(face_images.size(0)):
                current_face = face_images[idx].unsqueeze(0)
                current_landmarks = facial_landmarks[idx].unsqueeze(0)
                current_landmark_diff = landmark_differences[idx].unsqueeze(0)
                current_label = ground_truth_labels[idx].item()

                if current_label == 0:
                    outputs, _, _, _, _, _, _, _ = model(
                        current_face,
                        current_landmarks,
                        current_landmark_diff,
                        dissimilar_label,
                        size=128,
                        eval_step=True,
                    )
                elif current_label == 1:
                    outputs, _, _, _, _, _, _, _ = model(
                        current_face,
                        current_landmarks,
                        current_landmark_diff,
                        similar_label,
                        size=128,
                        eval_step=True,
                    )
                else:
                    print("Warning: Invalid label encountered")
                    continue

                score = F.softmax(outputs, dim=1).cpu().numpy()[:, 1][0]
                predicted_scores.append(score)
                true_labels.append(current_label)
                subjects.append(sample_subjects[idx])

    optimal_threshold = 0.5
    for i, score in enumerate(predicted_scores):
        if score >= optimal_threshold and true_labels[i] == 1:
            true_positive += 1
        elif score < optimal_threshold and true_labels[i] == 0:
            true_negative += 1
        elif score >= optimal_threshold and true_labels[i] == 0:
            false_positive += 1
        elif score < optimal_threshold and true_labels[i] == 1:
            false_negative += 1

    apcer = false_positive / (true_negative + false_positive)
    npcer = false_negative / (false_negative + true_positive)
    acer = (apcer + npcer) / 2
    auc = roc_auc_score(true_labels, predicted_scores)
    acc = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    return {
        "labels": true_labels,
        "scores": predicted_scores,
        "subjects": subjects,
        "threshold": optimal_threshold,
        "apcer": apcer,
        "npcer": npcer,
        "acer": acer,
        "auc": auc,
        "acc": acc,
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


torch.autograd.set_detect_anomaly(True)  # type: ignore

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(device))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("use cpu")

gc.collect()
torch.cuda.empty_cache()

args = get_args()
args.continue_model = False
args.train_dataset = "VIPL"

if args.finetune_dataset != "FF++":
    raise ValueError(f"Unsupported finetune_dataset: {args.finetune_dataset}. This project is FF++ only.")

train_name, _, _ = get_name(args, finetune=True, model_name="classifier", pretrain_model_name="Physformer")  # type: ignore
exp_name, result_dir = resolve_experiment_name_and_dir(args)

print(f"==========={args.subset} to {args.cross_subset}===========")

log_path = f"2025_logger_beta03/test/{exp_name}/{args.finetune_dataset}/all/{args.subset}/"
log_name = f"all_cos_fix_{args.subset}_2_{args.cross_subset}_{args.test_comp}"
log = get_logger(log_path, log_name)

weight_dir = os.path.join(result_dir, "weight")

if not os.path.isdir(weight_dir):
    raise FileNotFoundError(f"Weight directory not found: {weight_dir}")

seq_len = args.seq_len
fake_subset = args.cross_subset if args.cross_subset else args.subset

if args.test_comp == "c23":
    test_comp = args.comp_1_loader
elif args.test_comp == "c40":
    test_comp = args.comp_2_loader
elif args.test_comp == "raw":
    test_comp = args.comp_3_loader
else:
    raise ValueError(f"Unsupported test_comp: {args.test_comp}")

test_loader = get_loader(
    train=False,
    seq_length=seq_len,
    batch_size=args.bs,
    if_fg=True,
    shuffle=False,
    real_or_fake="both",
    real="youtube",
    fake=fake_subset,
    comp=test_comp,
)

print("len iter(test_loader) = ", len(iter(test_loader)))

model_text, _ = clip.load("ViT-B/16", device.type)
model = load_net(
    device,
    rPPG_path="./module/rppg_weight/Physformer_VIPL_fold1.pkl",
    LRNet_comp=args.comp_1_loader,
    model_text=model_text,
    args_set=args,
)

eval_result_save = os.path.join(log_path, "prob_save")
os.makedirs(eval_result_save, exist_ok=True)
ckpt_path, ckpt_name = find_best_checkpoint(weight_dir, args.subset)
print(f"Evaluating checkpoint: {ckpt_name} ({ckpt_path})")
model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
metrics_dict = evaluate_checkpoint(model, test_loader, device, args.bs)

output_name = (
    f"[{args.subset}_2_{fake_subset}]"
    f"{ckpt_name}_labels_probs_[Mask_config-{args.Mask_prob}_{args.Mask_percentage}_{args.text_prompt_filter}_{args.max_mask_count}].pkl"
)
with open(os.path.join(eval_result_save, output_name), "wb") as handle:
    pickle.dump({"labels": metrics_dict["labels"], "scores": metrics_dict["scores"]}, handle)

log.info(
    f"{ckpt_name} | [{args.subset}_2_{fake_subset}] | "
    f"ACER {metrics_dict['acer']:.5f} | APCER {metrics_dict['apcer']:.5f} | NPCER {metrics_dict['npcer']:.5f} | "
    f"AUC {metrics_dict['auc']:.5f} | ACC {metrics_dict['acc']:.5f} | "
    f"TP {metrics_dict['tp']:.0f} | FN {metrics_dict['fn']:.0f} | "
    f"TN {metrics_dict['tn']:.0f} | FP {metrics_dict['fp']:.0f} | "
    f"Threshold: {metrics_dict['threshold']}"
)
