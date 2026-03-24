from __future__ import print_function, division
import os, gc
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.append('model')
sys.path.append('module')
sys.path.append('util')
import torch
import numpy as np
import random
import math
import torch.nn.functional as F
import torch.nn as nn
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

from util.dataloader_FF_landmark import get_loader
from util.optim_tools import optim_tools_opt
from util.loss_function import *
from util.logger import *
from util.config import *
from util.phy_calc import *

torch.multiprocessing.set_sharing_strategy('file_system')
from module.landmark_model import DualLRNet
from module.rPPG_model import ViT_ST_ST_Compact3_TDC_gra_sharp
from module.base_model import PAD_Classifier
from module.clip import clip # type: ignore


def normalize(x):
    return (x-x.mean())/x.std()

def load_net(device, rPPG_path=None, LRNet_comp=None, model_text=None, args_set = None,
             Mask_prob = 0.0, Mask_percentage = 0.0,
             text_prompt_filter = False, max_mask_count = 0):
    
    if LRNet_comp is None or LRNet_comp == "":
        LRNet_load = False
        LRNet_comp = ""
    else:
        LRNet_load = True

    net_pad = DualLRNet(load_pretrained=LRNet_load,
                        pretrained_comp=LRNet_comp,
                        device=device)
    
    def load_model(model,path):
        pretrained_state = torch.load(path, map_location=device)
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_state.items() if k in model.state_dict()}
        model_dict = model.state_dict()

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict) # model_dict or pretrained_dict
        
        
        return model
    net_downstream = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(300,128,128), # type: ignore
                                                      patches=(4,4,4), # type: ignore
                                                        dim=96,
                                                        ff_dim=144,
                                                        num_heads=4,
                                                        num_layers=12,
                                                        dropout_rate=0.1,
                                                        theta=0.7)
    if rPPG_path is not None:
        net_downstream = load_model(net_downstream, rPPG_path)
        print("pretrained rPPG model is loaded")

    else:
        print("No pretrained rPPG model is loaded")

    net = PAD_Classifier(net_pad,net_downstream, model_text, args_set = args_set)
    net.to(device)
    # print(NNNNN)
    return net

class Pearson(nn.Module):    
    def __init__(self):
        super(Pearson,self).__init__()
        return
    def forward(self, preds, labels):
        N = preds.shape[1]
        sum_x = torch.sum(preds, dim=1)
        sum_y = torch.sum(labels, dim=1)
        sum_xy = torch.sum(preds * labels, dim=1)
        sum_x2 = torch.sum(preds ** 2, dim=1)
        sum_y2 = torch.sum(labels ** 2, dim=1)
        
        numerator = N * sum_xy - sum_x * sum_y
        denominator = torch.sqrt((N * sum_x2 - sum_x ** 2) * (N * sum_y2 - sum_y ** 2) + 1e-8)
        pearson = numerator / denominator
        
        loss = torch.mean(pearson)
        return loss


def build_experiment_name(args) -> str:
    """
    根據輸入參數組合實驗名稱，集中管理字串格式，讓主流程更乾淨。
    """

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


def setup_logging_and_dirs(args, exp_name, train_name):
    """
    建立 logger 與輸出資料夾；若偵測到既有結果則直接終止避免覆寫。
    """

    log = get_logger(f"2025_logger_beta03/train/{exp_name}/{args.finetune_dataset}/all/{args.subset}/", train_name)
    result_dir = os.path.abspath(f"./model_results/{exp_name}/{args.finetune_dataset}/all/{args.subset}/")
    weight_dir = os.path.join(result_dir, "weight")

    if os.path.isdir(weight_dir):
        skip_msg = f"Detected existing model results under {weight_dir}. Skip training for subset {args.subset}."
        print(skip_msg)
        log.info(skip_msg)
        sys.exit(0)

    os.makedirs(weight_dir, exist_ok=True)
    return log, result_dir, weight_dir


torch.autograd.set_detect_anomaly(True) # type: ignore

if torch.cuda.is_available():
    device = torch.device("cuda") 
    print(torch.cuda.get_device_name(device))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("use cpu")

# del model
gc.collect()
torch.cuda.empty_cache()

# get parameter
args = get_args()

if args.finetune_dataset != "FF++":
    raise ValueError(f"Unsupported finetune_dataset: {args.finetune_dataset}. This project is FF++ only.")
trainName, _, finetuneName = get_name(args, finetune=True, model_name="classifier", pretrain_model_name="Physformer") # type: ignore

exp_name = build_experiment_name(args)
log, result_dir, weight_dir = setup_logging_and_dirs(args, exp_name, trainName)
# loader
seq_len = args.seq_len

# comp_1_loader = c23, comp_2_loader = c40, comp_3_loader = raw
args.train_shuffle = True #False

args.eval_bs = args.bs//2
train_loader_raw_real = get_loader(
    train=True,
    seq_length=seq_len,
    batch_size=args.bs,
    if_fg=True,
    shuffle=args.train_shuffle,
    real_or_fake="real",
    real="youtube",
    fake=args.subset,
    comp=args.train_rate,
)
train_loader_raw_fake = get_loader(
    train=True,
    seq_length=seq_len,
    batch_size=args.bs,
    if_fg=True,
    shuffle=args.train_shuffle,
    real_or_fake="fake",
    real="youtube",
    fake=args.subset,
    comp=args.train_rate,
)
valid_loader_raw = get_loader(
    train=True,
    seq_length=seq_len,
    batch_size=args.eval_bs,
    if_fg=True,
    shuffle=False,
    real_or_fake="both",
    real="youtube",
    fake=args.subset,
    comp=args.train_rate,
    validate=True,
)
print("Real Length:", len(train_loader_raw_real))
print("Fake Length:", len(train_loader_raw_fake))
sample_length_num = min(len(train_loader_raw_real), len(train_loader_raw_fake))
# Load finetune model
_model = "None/*.pt"
print(f"_model = {_model}")


# text model
model_text, _ = clip.load("ViT-B/16", 'cuda')
# model build
model = load_net(device, rPPG_path="./module/rppg_weight/Physformer_VIPL_fold1.pkl", LRNet_comp=args.comp_1_loader,  model_text=model_text, args_set = args)
# print("Using Physformer_VIPL_fold1.pkl")
epoch_number = 0

dataset_num = len(iter(train_loader_raw_real))

args.load_pretrain = False
args.lr = args.lr
args.weight_decay = 5e-4
args.step_size = 3

# args.warmup = True
if args.warmup:
    # args.lr = 1e-4  # 初始學習率
    args.lr = args.lr
    args.min_lr = 1e-6  # 最小學習率
    # args.total_steps = (args.epoch * 711)//args.bs  # 總步數 = epoch * 每個 epoch 的步數
    args.total_steps = args.epoch * sample_length_num
    args.warmup_steps = int(0.1 * args.total_steps)  # 前 10% 步數作為 Warmup
    def warmup_cosine_lr_lambda(current_step):
        if current_step < args.warmup_steps:
            # Warmup 階段：學習率線性增加
            return current_step / args.warmup_steps
        else:
            # Cosine Annealing 階段
            progress = (current_step - args.warmup_steps) / (args.total_steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    opt_fg = optim_tools_opt(model = model, args = args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_fg, lr_lambda=warmup_cosine_lr_lambda)
else:
    # opt_fg = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.lr_decay_gamma = 0.5
    args.lr_decay_step = args.step_size
    opt_fg = optim_tools_opt(model = model, args = args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt_fg, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
BCE_loss = nn.CrossEntropyLoss()
# track loss status
# loss 
similar = torch.tensor([1], dtype=torch.float).to(device) #unused
dissimilar = torch.tensor([-1], dtype=torch.float).to(device) #unused

similar_label = torch.tensor([1], dtype=torch.float).to(device)
dissimilar_label = torch.tensor([-1], dtype=torch.float).to(device)

log.info(f"Training Parameter....[{args}]")
log.info(f"Saving Path....[{result_dir}]")
best_auc = 0.0
for epoch in range(epoch_number, args.epoch):
    print(f"Start Training Epoch: {epoch}/{args.epoch}")
    current_lr = opt_fg.param_groups[0]['lr']
    for step, (data_raw_real, data_raw_fake) in enumerate(zip(train_loader_raw_real, train_loader_raw_fake)):
        # Multi-modal Prompt-guided Learning
        face_frames_real, landmarks_real, landmarks_diff_real, label_real, subjects_real = data_raw_real
        face_frames_fake, landmarks_fake, landmarks_diff_fake, label_fake, subjects_fake = data_raw_fake
        landmarks_real = random_mask_modality(landmarks_real, prob=0.2)
        landmarks_fake = random_mask_modality(landmarks_fake, prob=0.2)
        # print(face_frames_real.shape, landmarks_real.shape, landmarks_diff_real.shape, label_real.shape)
        # torch.Size([bs, 3, 160, 128, 128]) torch.Size([bs, 160, 136]) torch.Size([bs, 159, 136]) torch.Size([bs, 1])
        # subjects_real:  ['youtube,001,+1'] ||subjects_fake:  ['Deepfakes,001_870,-1']
        # check batch size
        if face_frames_real.shape[0] != args.bs or face_frames_fake.shape[0] != args.bs:
            print(f"{face_frames_real.shape[0]=}, continue")
            print(f"{face_frames_fake.shape[0]=}, continue")
            continue
        try:
            # Move data to device
            face_frames_real = face_frames_real.to(device)
            face_frames_fake = face_frames_fake.to(device)
            landmarks_real = landmarks_real.to(device)
            landmarks_fake = landmarks_fake.to(device)
            landmarks_diff_real = landmarks_diff_real.to(device)
            landmarks_diff_fake = landmarks_diff_fake.to(device)
            label_real = label_real.to(device)
            label_fake = label_fake.to(device)
            # print(f"label_real: {label_real} label_fake: {label_fake}")
            # label_real: tensor([[1]], device='cuda:0') label_fake: tensor([[0]], device='cuda:0')
            # Forward pass for raw data
            out_real_raw, _, \
            sim_tr_real_raw, sim_tl_real_raw, sim_rl_real_raw, \
            _,_,_ \
                = model(face_frames_real, landmarks_real, landmarks_diff_real, similar, size=128)

            out_fake_raw, _, \
            sim_tr_fake_raw, sim_tl_fake_raw, sim_rl_fake_raw, \
            _,_,_ \
                = model(face_frames_fake, landmarks_fake, landmarks_diff_fake, dissimilar, size=128)
            # bce_alpha = 0.7 ; bce_beta = 0.3 # 主輸出權重 / # 反向輸出權重
            bce_alpha = args.bce_alpha ; bce_beta = args.bce_beta # 主輸出權重 / # 反向輸出權重
            # # BCE Loss
            loss_bce_real = BCE_loss(out_real_raw, label_real[:, 0].long())
            loss_bce_fake = BCE_loss(out_fake_raw, label_fake[:, 0].long())
            
            
            loss_bce = loss_bce_real + loss_bce_fake
            log.info(f"[Subset: {args.subset}] Epoch {epoch}, Step {step}: BCE Loss - Real={loss_bce_real.mean().item():.4f}, Fake={loss_bce_fake.mean().item():.4f}, Total={loss_bce.mean().item():.4f}")
            # print(out_real_raw.grad_fn)
            # print(out_fake_raw.grad_fn)


            # print(loss_bce)
            loss_bce.backward()
            
            torch.cuda.empty_cache()
            
            # Multi-modal Prompt-guided Contrastive Learning
            out_real_s64, rppg_real_s64, \
            sim_tr_real_s64, sim_tl_real_s64, sim_rl_real_s64, \
            text_emb_real_s64, rppg_emb_real_s64, lmk_emb_real_s64, \
                = model(face_frames_real, landmarks_real, landmarks_diff_real, similar, size=64)
            
            out_fake_s64, rppg_fake_s64, \
            sim_tr_fake_s64, sim_tl_fake_s64, sim_rl_fake_s64, \
            text_emb_fake_s64, rppg_emb_fake_s64, lmk_emb_fake_s64, \
                = model(face_frames_fake, landmarks_fake, landmarks_diff_fake, dissimilar, size=64)            

            out_real_s32, rppg_real_s32, \
            sim_tr_real_s32, sim_tl_real_s32, sim_rl_real_s32, \
            text_emb_real_s32, rppg_emb_real_s32, lmk_emb_real_s32, \
                = model(face_frames_real, landmarks_real, landmarks_diff_real, similar, size=32)
            
            out_fake_s32, rppg_fake_s32, \
            sim_tr_fake_s32, sim_tl_fake_s32, sim_rl_fake_s32, \
            text_emb_fake_s32, rppg_emb_fake_s32, lmk_emb_fake_s32, \
                = model(face_frames_fake, landmarks_fake, landmarks_diff_fake, dissimilar, size=32)  
            # print(text_emb_real_s64.shape, rppg_emb_real_s64.shape, lmk_emb_real_s64.shape)
            # torch.Size([bs, 320]) torch.Size([bs, 320]) torch.Size([bs, 320])

            pearson_loss_fn = Pearson()
            loss_rppg_between_s64rf = abs(pearson_loss_fn(rppg_real_s64, rppg_fake_s64)) #range 0~1
            loss_rppg_between_s32rf = abs(pearson_loss_fn(rppg_real_s32, rppg_fake_s32)) #range 0~1
            
            loss_rppg_between_s64r_s32f = abs(pearson_loss_fn(rppg_real_s64, rppg_fake_s32)) #range 0~1
            loss_rppg_between_s32r_s64f = abs(pearson_loss_fn(rppg_real_s32, rppg_fake_s64)) #range 0~1
            
            
            loss_rppg_rf = loss_rppg_between_s64rf + loss_rppg_between_s32rf + \
                           loss_rppg_between_s64r_s32f + loss_rppg_between_s32r_s64f #range 0~4
                           
            loss_rppg_between_s64r_s32r = 1 - pearson_loss_fn(rppg_real_s64, rppg_real_s32) #range 0~2
            loss_rppg_between_s64f_s32f = 1 - pearson_loss_fn(rppg_fake_s64, rppg_fake_s32) #range 0~2
            
            loss_rppg_rr_ff = loss_rppg_between_s64r_s32r + loss_rppg_between_s64f_s32f #range 0~4
            
            log.info(f"[Subset: {args.subset}] Epoch {epoch}, Step {step}: CQSL Loss - s64_R-F={loss_rppg_between_s64rf.mean().item():.4f}, s32_R-F={loss_rppg_between_s32rf.mean().item():.4f}, s64-32_R-F={loss_rppg_between_s64r_s32f.mean().item():.4f}, s32-64_R-F={loss_rppg_between_s32r_s64f.mean().item():.4f}, s64-32_R-R={loss_rppg_between_s64r_s32r.mean().item():.4f}, s64-32_F-F={loss_rppg_between_s64f_s32f.mean().item():.4f}")
            if args.phy_loss_w!=0:
                physiological_loss = loss_rppg_rf + loss_rppg_rr_ff
            else:
                physiological_loss = (torch.tensor([0.0]).to('cuda'))
            # Bandpass filter and heart rate calculation
            rppg_real_s64_filtered = butter_bandpass(rppg_real_s64.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
            rppg_fake_s64_filtered = butter_bandpass(rppg_fake_s64.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
            rppg_real_s32_filtered = butter_bandpass(rppg_real_s32.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
            rppg_fake_s32_filtered = butter_bandpass(rppg_fake_s32.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)

            hr_real_s64, _, _ = hr_fft(rppg_real_s64_filtered, fs=30)
            hr_fake_s64, _, _ = hr_fft(rppg_fake_s64_filtered, fs=30)
            hr_real_s32, _, _ = hr_fft(rppg_real_s32_filtered, fs=30)
            hr_fake_s32, _, _ = hr_fft(rppg_fake_s32_filtered, fs=30)

            hr_real_s64 = HR_60_to_120(hr_real_s64)
            hr_fake_s64 = HR_60_to_120(hr_fake_s64)
            hr_real_s32 = HR_60_to_120(hr_real_s32)
            hr_fake_s32 = HR_60_to_120(hr_fake_s32)

            loss_hr = abs(hr_real_s64 - hr_real_s32) / 60 + abs(hr_fake_s64 - hr_fake_s32) / 60
            physiological_loss += loss_hr
            log.info(f"[Subset: {args.subset}] Epoch {epoch}, Step {step}: Physiological Loss - Pull={loss_rppg_rr_ff.mean().item():.4f}, Push={loss_rppg_rf.mean().item():.4f}, HR Loss={loss_hr:.4f}, Total={physiological_loss.mean().item():.4f}")
            #####################################################################################
            # only_test5
            # 需要補上affinity loss的地方！！！！！
            text_prompt_weight = args.text_prompt_weight #default = 0.3
            # main matrix 64
            aff_mat_real_s64_main = build_affinity_matrix(text_emb_real_s64*text_prompt_weight, rppg_emb_real_s64, lmk_emb_real_s64)
            aff_mat_fake_s64_main = build_affinity_matrix(text_emb_fake_s64*text_prompt_weight, rppg_emb_fake_s64, lmk_emb_fake_s64)
            loss_aff_real_s64_main = explicit_affinity_loss(aff_mat_real_s64_main, alpha=args.aff_alpha, beta=args.aff_beta)
            loss_aff_fake_s64_main = explicit_affinity_loss(aff_mat_fake_s64_main, alpha=args.aff_alpha, beta=args.aff_beta)
            total_affinity_loss_s64 = (
                loss_aff_real_s64_main + loss_aff_fake_s64_main
            ) / 2 
            # main matrix 32
            aff_mat_real_s32_main = build_affinity_matrix(text_emb_real_s32*text_prompt_weight, rppg_emb_real_s32, lmk_emb_real_s32)
            aff_mat_fake_s32_main = build_affinity_matrix(text_emb_fake_s32*text_prompt_weight, rppg_emb_fake_s32, lmk_emb_fake_s32)
            loss_aff_real_s32_main = explicit_affinity_loss(aff_mat_real_s32_main, alpha=args.aff_alpha, beta=args.aff_beta)
            loss_aff_fake_s32_main = explicit_affinity_loss(aff_mat_fake_s32_main, alpha=args.aff_alpha, beta=args.aff_beta)
            
            total_affinity_loss_s32 = (
                loss_aff_real_s32_main + loss_aff_fake_s32_main
            ) / 2
            total_affinity_loss = (total_affinity_loss_s64 + total_affinity_loss_s32) / 2
            
            log.info(f"[Subset: {args.subset}] Epoch {epoch}, Step {step}: Aff S64 Real Main={loss_aff_real_s64_main.mean().item():.4f}, Aff S64 Fake Main={loss_aff_fake_s64_main.mean().item():.4f}, Aff S32 Real Main={loss_aff_real_s32_main.mean().item():.4f}, Aff S32 Fake Main={loss_aff_fake_s32_main.mean().item():.4f}, Total Affinity Loss={total_affinity_loss.mean().item():.4f}")
            
            #####################################################################################
            # only_test4
            label_real = torch.tensor([1.0]).to(device)  # Real 標籤
            label_fake = torch.tensor([0.0]).to(device)  # Fake 標籤
            cross_comp_total_loss = torch.tensor([0.0], device=device)
            loss_contrastive = torch.tensor([0.0], device=device)
            loss_compression = torch.tensor([0.0], device=device)

            # Total Loss
            # total_loss = 0.2 * physiological_loss + 0.25 * cross_comp_total_loss + 0.3 * total_affinity_loss
            total_loss = args.phy_loss_w * physiological_loss + args.CCTL_w * cross_comp_total_loss + args.aff_loss_w * total_affinity_loss #default: 0.2-0.25-0.3
            # Backward and optimize
            total_loss.backward()
            opt_fg.step()
            if args.warmup:
                scheduler.step()            
            opt_fg.zero_grad()
            current_step = epoch * len(train_loader_raw_real) + step
            current_lr = opt_fg.param_groups[0]['lr']
            torch.cuda.empty_cache()
            log.info(f"[Subset: {args.subset}] Epoch {epoch}, Step {step}: Total Loss={total_loss.mean().item():.4f}, Learning Rate: {current_lr:.6f}")
            log.info(f"[Subset: {args.subset}] Epoch {epoch}, Step {step}: [cos sim main] s64r: tr({sim_tr_real_s64.mean().item():.4f}), tl({sim_tl_real_s64.mean().item():.4f}), rl({sim_rl_real_s64.mean().item():.4f})")
            log.info(f"[Subset: {args.subset}] Epoch {epoch}, Step {step}: [cos sim main] s32r: tr({sim_tr_real_s32.mean().item():.4f}), tl({sim_tl_real_s32.mean().item():.4f}), rl({sim_rl_real_s32.mean().item():.4f})")
            log.info(f"[Subset: {args.subset}] Epoch {epoch}, Step {step}: [cos sim main] s64f: tr({sim_tr_fake_s64.mean().item():.4f}), tl({sim_tl_fake_s64.mean().item():.4f}), rl({sim_rl_fake_s64.mean().item():.4f})")
            log.info(f"[Subset: {args.subset}] Epoch {epoch}, Step {step}: [cos sim main] s32f: tr({sim_tr_fake_s32.mean().item():.4f}), tl({sim_tl_fake_s32.mean().item():.4f}), rl({sim_rl_fake_s32.mean().item():.4f})")

            

        except torch.cuda.OutOfMemoryError as e:
            log.error(f"Step {step} failed due to CUDA OOM: {e}")
            torch.cuda.empty_cache()
            continue

        finally:
            # Clear memory after each step
            del face_frames_real, face_frames_fake, landmarks_real, landmarks_fake
            del landmarks_diff_real, landmarks_diff_fake, label_real, label_fake
            gc.collect()
            torch.cuda.empty_cache()
        print("-"*100)
    # Save model after each epoch
    if (epoch + 1)%2==0:
        torch.save(model.state_dict(), os.path.join(weight_dir, f'epoch_{epoch + 1}_{args.subset}.pt'))
    if args.warmup!=True:
        scheduler.step()
    # Validation process 
    true_labels = []
    predicted_scores = []
    subjects = []
    
    # 初始化計數器
    true_positive = 0.0000001
    true_negative = 0.0000001
    false_positive = 0.0000001
    false_negative = 0.0000001
    incorrect_predictions = 0
    
    with torch.no_grad():
        for step, test_data in enumerate(valid_loader_raw):
            # 解構數據
            face_images, facial_landmarks, landmark_differences, ground_truth_labels, sample_subjects = test_data
            # print(face_images.shape, facial_landmarks.shape, landmark_differences.shape)
            # 驗證批次大小
            if face_images.shape[0] != args.eval_bs:
                print(f"Batch size mismatch: {face_images.shape[0]=}, skipping this batch")
                continue

            # 移動數據到設備
            face_images = face_images.to(device)
            facial_landmarks = facial_landmarks.to(device)
            landmark_differences = landmark_differences.to(device)
            ground_truth_labels = ground_truth_labels.to(device)

            # 模型推理 (針對 real 和 fake)
            for idx in range(face_images.size(0)):# idx = 0; batch size = 1
                current_face = face_images[idx].unsqueeze(0)
                current_landmarks = facial_landmarks[idx].unsqueeze(0)
                current_landmark_diff = landmark_differences[idx].unsqueeze(0)
                current_label = ground_truth_labels[idx].item()
                # print(current_face.shape, current_landmarks.shape, current_landmark_diff.shape)
                if current_label == 0:  # 假樣本 (Fake)
                    outputs, _, \
                    _, _, _, \
                    _, _, _ \
                        = model(current_face, current_landmarks, current_landmark_diff, dissimilar_label, size=128, eval_step=True)
                elif current_label == 1:  # 真樣本 (Real)
                    outputs, _, \
                    _, _, _, \
                    _, _, _ \
                        = model(current_face, current_landmarks, current_landmark_diff, similar_label, size=128, eval_step=True)
                else:
                    print("Warning: Invalid label encountered")
                    continue

                # 計算 softmax 預測分數
                
                softmax_scores = F.softmax(outputs, dim=1).cpu().data.numpy()[:, 1]
                # print(f"Outputs, soft_Output, Label: ", outputs, softmax_scores, current_label)
                # 記錄結果
                # print(softmax_scores)
                # print(NNNNN)
                predicted_scores.append(softmax_scores[0])
                true_labels.append(current_label)
                subjects.append(sample_subjects[idx])

        # 正規化預測分數
        # predicted_scores = NormalizeData(predicted_scores)
        # print(predicted_scores)
        # 計算 ROC 和評估指標
        
        optimal_threshold = 0.5
        error_entries = []
        for i, score in enumerate(predicted_scores):
            if score >= optimal_threshold and true_labels[i] == 1:  # 真正例
                true_positive += 1
            elif score < optimal_threshold and true_labels[i] == 0:  # 真負例
                true_negative += 1
            elif score >= optimal_threshold and true_labels[i] == 0:  # 假正例
                false_positive += 1
                error_entries.append((subjects[i], true_labels[i]))
            elif score < optimal_threshold and true_labels[i] == 1:  # 假負例
                false_negative += 1
                error_entries.append((subjects[i], true_labels[i]))

        # 計算指標
        apcer = false_positive / (true_negative + false_positive)
        npcer = false_negative / (false_negative + true_positive)
        acer = (apcer + npcer) / 2
        auc = roc_auc_score(true_labels, predicted_scores)
        acc = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        if auc>best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(weight_dir, f'best_model_{args.subset}.pt'))
        # 更新日誌輸出
        log.info(
            f"[Subset: {args.subset}] | Epoch {epoch} | ACER {acer:.5f} | APCER {apcer:.5f} | NPCER {npcer:.5f} | AUC {auc:.5f} | ACC {acc:.5f} | "
            f"TP {true_positive:.0f} | FN {false_negative:.0f} | TN {true_negative:.0f} | FP {false_positive:.0f} | Best AUC: {best_auc:.5f}"
        )
        print("-"*100)
    
    gc.collect()
    torch.cuda.empty_cache()
    # writer.flush()
