from __future__ import print_function, division

import os
import sys
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

sys.path.append("model")
sys.path.append("module")
sys.path.append("util")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from util.config import get_args
from module.landmark_model import DualLRNet
from module.rPPG_model import ViT_ST_ST_Compact3_TDC_gra_sharp
from module.base_model import PAD_Classifier
from module.clip import clip  # type: ignore
def get_runtime_args():
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = get_args()
    finally:
        sys.argv = original_argv
    return args


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


def load_frames(image_dir, seq_len):
    image_paths = []
    for name in sorted(os.listdir(image_dir)):
        lower = name.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_paths.append(os.path.join(image_dir, name))

    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    transform_face = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    selected_paths = image_paths[:seq_len]
    face_frames = []
    for path in selected_paths:
        frame = Image.open(path).convert("RGB")
        face_frames.append(transform_face(frame))

    face_tensor = torch.stack(face_frames).transpose(0, 1)
    if face_tensor.shape[1] < seq_len:
        face_tensor = F.pad(face_tensor, (0, 0, 0, 0, 0, seq_len - face_tensor.shape[1]))

    return face_tensor.unsqueeze(0), image_paths


def load_landmarks(landmark_path, seq_len):
    vectors = np.loadtxt(landmark_path, dtype=np.float32)
    if vectors.shape[0] < seq_len:
        vectors = np.pad(vectors, ((0, seq_len - vectors.shape[0]), (0, 0)), "edge")

    vec = vectors[:seq_len, :]
    vec_next = vectors[1:seq_len, :]
    vec_next = np.pad(vec_next, ((0, 1), (0, 0)), "constant", constant_values=(0, 0))
    vec_diff = (vec_next - vec)[: seq_len - 1, :]

    landmark_tensor = torch.from_numpy(vec).unsqueeze(0)
    landmark_diff_tensor = torch.from_numpy(vec_diff).unsqueeze(0)
    return landmark_tensor, landmark_diff_tensor


def run_inference(cli_args):
    args = get_runtime_args()
    args.text_prompt_filter = True
    args.max_mask_count = 0
    args.Mask_prob = 0.0
    args.Mask_percentage = 0.0

    if cli_args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(cli_args.device if cli_args.device == "cpu" else "cuda")
    model_text, _ = clip.load("ViT-B/16", device.type)
    model = load_net(
        device,
        rPPG_path=cli_args.rppg_path,
        LRNet_comp=cli_args.lrnet_comp,
        model_text=model_text,
        args_set=args,
    )

    checkpoint = torch.load(cli_args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    face_tensor, image_paths = load_frames(cli_args.image_dir, cli_args.seq_len)
    landmark_tensor, landmark_diff_tensor = load_landmarks(cli_args.landmark_path, cli_args.seq_len)

    face_tensor = face_tensor.to(device)
    landmark_tensor = landmark_tensor.to(device)
    landmark_diff_tensor = landmark_diff_tensor.to(device)

    # With eval_step=True, both prompt branches collapse to the same fixed text.
    similar_label = torch.tensor([1], dtype=torch.float).to(device)

    with torch.no_grad():
        outputs, _, sim_tr, sim_tl, sim_rl, _, _, _ = model(
            face_tensor,
            landmark_tensor,
            landmark_diff_tensor,
            similar_label,
            size=128,
            eval_step=True,
        )

    probs = F.softmax(outputs, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = "real" if pred_idx == 1 else "fake"

    print(f"checkpoint: {cli_args.checkpoint}")
    print(f"frames_used: {min(len(image_paths), cli_args.seq_len)} / total_frames: {len(image_paths)}")
    print(f"landmark_path: {cli_args.landmark_path}")
    print(f"prediction: {pred_label}")
    print(f"prob_fake: {probs[0]:.6f}")
    print(f"prob_real: {probs[1]:.6f}")
    print(f"sim_tr: {sim_tr.mean().item():.6f}")
    print(f"sim_tl: {sim_tl.mean().item():.6f}")
    print(f"sim_rl: {sim_rl.mean().item():.6f}")
if __name__ == "__main__":
    class Args:
        image_dir = "/ssd7/chihyi/Dataset/DeepFake_Dataset/FF++_landmark_c23/manipulated_sequences/Deepfakes/c23/crop_MTCNN/000_003/face0"
        landmark_path = "/ssd7/chihyi/Dataset/DeepFake_Dataset/FF++_landmark_c23/manipulated_sequences/Deepfakes/c23/video_landmark_new/000_003.txt"
        checkpoint = "./model_results/[c23]Demo_exper_init_1[Fix-Prompt_w=0.3][AffL-a_0.5_b0.3][WarmUp][phy_w0.1-CCTL_w0.0-AFF_w0.5][Mask_config-0.0_0.5_False_2][Classifier-mlp]/FF++/all/DF/weight/best_model_DF.pt"
        rppg_path = "./module/rppg_weight/Physformer_VIPL_fold1.pkl"
        lrnet_comp = "c23"
        seq_len = 160
        device = "cuda"

    run_inference(Args())
