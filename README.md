# Unimodal-generated-Multimodal-Contrastive-Learning-for-Cross-compression-rate-Deepfake-Detection (UMCL)
UMCL is a multimodal deepfake detection project built for `FaceForensics++ (FF++)`. The model combines three signal sources:

- facial image sequences
- facial landmarks and landmark differences
- rPPG features and text prompts through a CLIP text encoder

The current codebase supports:

- training with `train_main.py`
- evaluation with `test_alignment_all.py`
- single-sample inference with `inference.py`

This README reflects the project as it exists in this folder, including its current hardcoded paths, data assumptions, and output locations.

## 1. Overview

- `FF++` is the only supported finetuning and evaluation dataset.
- The visual branch uses `ViT_ST_ST_Compact3_TDC_gra_sharp`.
- The landmark branch uses `DualLRNet`.
- The text branch uses the bundled implementation under `module/clip`.
- The project expects these pretrained weights:
  - `module/rppg_weight/Physformer_VIPL_fold1.pkl`
  - `module/LRNet_org_weights/g1.pth`
  - `module/LRNet_org_weights/g2.pth`
- Training logs and model outputs are written to:
  - `model_results/`
  - `2025_logger_beta03/`

## 2. Project Structure

```text
UMCL/
|-- train_main.py                  # main training script
|-- test_alignment_all.py          # main evaluation script
|-- inference.py                   # single-sample inference example
|-- train.sh                       # training shell script example
|-- test.sh                        # evaluation shell script example
|-- config_val_160/                # train / val / test split files
|-- module/
|   |-- base_model.py              # multimodal classifier
|   |-- landmark_model.py          # landmark encoder
|   |-- rPPG_model.py              # rPPG / video backbone
|   |-- prompt_rppg.py             # text prompt templates
|   |-- clip/                      # bundled CLIP implementation
|   |-- LRNet_org_weights/         # pretrained LRNet weights
|   `-- rppg_weight/               # pretrained rPPG weights
|-- util/
|   |-- dataloader_FF_landmark.py  # FF++ dataloader
|   |-- config.py                  # argument definitions
|   |-- logger.py                  # logger utility
|   |-- optim_tools.py             # optimizer utility
|   `-- phy_calc.py                # physiological signal utilities
`-- model_results/                 # saved model checkpoints
```

## 3. Environment

Recommended environment:

- Python 3.9 to 3.12
- CUDA-capable GPU
- Linux or WSL

Main Python dependencies:

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `scikit-learn`
- `scipy`
- `pytz`

There is currently no `requirements.txt` in this project. A minimal manual installation looks like this:

```bash
pip install torch torchvision numpy pillow scikit-learn scipy pytz
```

If your PyTorch installation depends on a specific CUDA version, install the matching PyTorch build first, then install the remaining packages.

## 4. Dataset Preparation

### 4.1 Hardcoded Dataset Paths

`util/dataloader_FF_landmark.py` currently hardcodes the FF++ root directories as:

```python
self.real_root_dir = "/ssd7/chihyi/Dataset/DeepFake_Dataset/FF++_landmark_c23/original_sequences"
self.fake_root_dir = "/ssd7/chihyi/Dataset/DeepFake_Dataset/FF++_landmark_c23/manipulated_sequences"
```

If your dataset is stored elsewhere, you must edit these lines before training or evaluation.

### 4.2 Expected Directory Layout

The dataloader expects a structure similar to:

```text
FF++_landmark_c23/
|-- original_sequences/
|   `-- youtube/
|       `-- c23/
|           |-- crop_MTCNN/
|           |   `-- 000/
|           |       `-- face0/
|           |           |-- 0001.jpg
|           |           |-- 0002.jpg
|           |           `-- ...
|           `-- video_landmark_new/
|               `-- 000.txt
`-- manipulated_sequences/
    |-- Deepfakes/
    |-- Face2Face/
    |-- FaceSwap/
    `-- NeuralTextures/
        `-- c23/
            |-- crop_MTCNN/
            |   `-- 000_003/
            |       `-- face0/
            |           |-- 0001.jpg
            |           `-- ...
            `-- video_landmark_new/
                `-- 000_003.txt
```

### 4.3 Split Files

Dataset splits are controlled by the text files under `config_val_160/`:

- `FF++_train.txt`
- `FF++_train2.txt`
- `FF++_val.txt`
- `FF++_test.txt`
- `FF++_test2.txt`

Each line uses this format:

```text
subset,subject,label
```

Example:

```text
youtube,000,+1
Deepfakes,000_003,-1
```

Where:

- `+1` means real
- `-1` means fake

### 4.4 Landmark Format

Landmark files are loaded with `numpy.loadtxt()`, so each file is expected to be a plain text numeric matrix where each row corresponds to one frame.

The current model expects:

- `136` values per frame, corresponding to `68` landmark points in `(x, y)` format

### 4.5 Minimum Frame Requirement

The default sequence length is `160`. If a sample has fewer than `seq_len - 30` usable frames, the dataloader skips it.

## 5. Pretrained and Output Weights

The project currently depends on these pretrained files:

- `module/rppg_weight/Physformer_VIPL_fold1.pkl`
- `module/LRNet_org_weights/g1.pth`
- `module/LRNet_org_weights/g2.pth`

Trained classifier checkpoints are saved under:

```text
model_results/<experiment_name>/FF++/all/<subset>/weight/
```

Typical filenames:

- `best_model_DF.pt`
- `best_model_F2F.pt`
- `best_model_FS.pt`
- `best_model_NT.pt`
- `best_auc.pt`

## 6. Training

### 6.1 Run Training Directly

You can start from the settings in `train.sh`. Example:

```bash
python train_main.py \
  --exper_name Demo_exper_init_1 \
  --train_dataset VIPL \
  --finetune_dataset FF++ \
  --subset DF \
  --comp_1_loader c23 \
  --comp_2_loader c40 \
  --comp_3_loader raw \
  --model_S 3 \
  --epoch 3 \
  --bs 4 \
  --lr 0.001 \
  --seq_len 160 \
  --train_rate c23 \
  --warmup true \
  --Mask_prob 0.0 \
  --Mask_percentage 0.5 \
  --text_prompt_filter false \
  --max_mask_count 2 \
  --text_prompt_weight 0.3 \
  --aff_alpha 0.5 \
  --aff_beta 0.3 \
  --phy_loss_w 0.1 \
  --CCTL_w 0.0 \
  --aff_loss_w 0.5 \
  --classifier_module mlp
```

### 6.2 Train All Four Manipulation Types

`train.sh` loops over:

- `DF`
- `NT`
- `FS`
- `F2F`

Run it with:

```bash
bash train.sh
```

### 6.3 Training Outputs

Training logs are written to:

```text
2025_logger_beta03/train/<experiment_name>/FF++/all/<subset>/
```

Model weights are written to:

```text
model_results/<experiment_name>/FF++/all/<subset>/weight/
```

If `weight/` already exists, `train_main.py` currently treats that subset as already processed and exits.

## 7. Evaluation

### 7.1 Run Evaluation Directly

Example:

```bash
python test_alignment_all.py \
  --exper_name "[c23]Demo_exper_init_1[Fix-Prompt_w=0.3][AffL-a_0.5_b0.3][WarmUp][phy_w0.1-CCTL_w0.0-AFF_w0.5][Mask_config-0.0_0.5_False_2][Classifier-mlp]" \
  --train_dataset VIPL \
  --finetune_dataset FF++ \
  --test_dataset FF++ \
  --subset DF \
  --cross_subset DF \
  --test_comp c23 \
  --comp_1_loader c23 \
  --comp_2_loader c40 \
  --comp_3_loader raw \
  --bs 1 \
  --seq_len 160 \
  --Mask_prob 0.0 \
  --Mask_percentage 0.0 \
  --text_prompt_filter true \
  --max_mask_count 0 \
  --text_prompt_weight 0.5 \
  --CCAL_alpha 0.5 \
  --CCAL_beta 0.5 \
  --aff_alpha 0.5 \
  --aff_beta 0.5 \
  --phy_loss_w 0.1 \
  --CCTL_w 0.5 \
  --aff_loss_w 0.5 \
  --classifier_module mlp
```

### 7.2 Use the Evaluation Script

```bash
bash test.sh
```

### 7.3 Metrics

`test_alignment_all.py` reports:

- `AUC`
- `ACC`
- `APCER`
- `NPCER`
- `ACER`
- `TP / TN / FP / FN`

It also saves labels and prediction scores as `.pkl` files under:

```text
2025_logger_beta03/test/<experiment_name>/FF++/all/<subset>/prob_save/
```

## 8. Single-Sample Inference

`inference.py` is not yet implemented as a standard command-line interface. It currently uses a hardcoded `Args` class at the bottom of the file. There are two practical ways to use it.

### 8.1 Option 1: Edit the Paths in `inference.py`

Update the values in:

```python
class Args:
    image_dir = "path/to/your/frames"
    landmark_path = "path/to/your/landmark.txt"
    checkpoint = "path/to/your/checkpoint.pt"
    rppg_path = "./module/rppg_weight/Physformer_VIPL_fold1.pkl"
    lrnet_comp = "c23"
    seq_len = 160
    device = "cuda"
```

Then run:

```bash
python inference.py
```

### 8.2 Option 2: Call It from Python

```python
from inference import run_inference

class Args:
    image_dir = "/path/to/face0"
    landmark_path = "/path/to/sample.txt"
    checkpoint = "./model_results/.../best_model_DF.pt"
    rppg_path = "./module/rppg_weight/Physformer_VIPL_fold1.pkl"
    lrnet_comp = "c23"
    seq_len = 160
    device = "cuda"

run_inference(Args())
```

### 8.3 Inference Output

Inference prints:

- checkpoint path
- number of frames used
- landmark file path
- predicted label: `real` or `fake`
- `prob_fake`
- `prob_real`
- cross-modal similarity values

## 9. Main Arguments

Common arguments are defined in `util/config.py`.

### 9.1 Dataset and Experiment Settings

- `--exper_name`: experiment name
- `--finetune_dataset`: currently only `FF++`
- `--subset`: training fake subset, typically `DF`, `F2F`, `FS`, or `NT`
- `--cross_subset`: fake subset used for evaluation
- `--train_rate`: training compression level, one of `raw`, `c23`, `c40`
- `--test_comp`: evaluation compression level
- `--seq_len`: sequence length, default `160`
- `--bs`: batch size

### 9.2 Prompt and Mask Settings

- `--text_prompt_weight`: text prompt loss weight
- `--Mask_prob`: probability of applying random modality masking
- `--Mask_percentage`: masking ratio
- `--max_mask_count`: maximum number of masked modalities
- `--text_prompt_filter`: whether evaluation uses fixed prompt text

### 9.3 Loss Weights

- `--phy_loss_w`: physiological loss weight
- `--CCTL_w`: alignment-related loss weight
- `--aff_loss_w`: affinity loss weight
- `--aff_alpha`
- `--aff_beta`
- `--CCAL_alpha`
- `--CCAL_beta`

### 9.4 Classifier Settings

- `--classifier_module mlp`
- `--classifier_module trans_mlp`

## 10. Output Summary

### 10.1 Training

```text
2025_logger_beta03/train/<experiment_name>/FF++/all/<subset>/
model_results/<experiment_name>/FF++/all/<subset>/weight/
```

### 10.2 Evaluation

```text
2025_logger_beta03/test/<experiment_name>/FF++/all/<subset>/
2025_logger_beta03/test/<experiment_name>/FF++/all/<subset>/prob_save/
```

## 11. Known Limitations

- Only `FF++` is supported for finetuning and evaluation.
- Dataset root paths are hardcoded in the dataloader.
- `train_main.py` hardcodes `CUDA_VISIBLE_DEVICES`.
- `test_alignment_all.py` also hardcodes `CUDA_VISIBLE_DEVICES`.
- `inference.py` is not yet a proper CLI tool.
- The project does not include `requirements.txt`, `environment.yml`, or a complete preprocessing pipeline.
- The folder still contains `__pycache__`, historical logs, and saved model outputs. If this repository will be shared publicly, cleanup is recommended.

## 12. Troubleshooting

### 12.1 Dataset Not Found

Check the following:

- the dataset root paths in `util/dataloader_FF_landmark.py`
- whether subject names in `config_val_160/*.txt` match the actual directories
- whether `crop_MTCNN/<subject>/face0/*.jpg` exists
- whether `video_landmark_new/<subject>.txt` exists

### 12.2 Wrong GPU Device

The scripts currently hardcode:

- `train_main.py`: `os.environ["CUDA_VISIBLE_DEVICES"] = "4"`
- `test_alignment_all.py`: `os.environ["CUDA_VISIBLE_DEVICES"] = "2"`

Update them for your machine before running.

### 12.3 Incomplete Final Batch Gets Skipped

`test_alignment_all.py` skips a batch if `face_images.shape[0] != batch_size`. In practice, this usually happens on the last batch when the sample count is not divisible by the batch size.

## 13. Recommended Cleanup

If you want to make this project easier for other people to use, the highest-value next steps are:

1. Replace hardcoded dataset paths with arguments or environment variables.
2. Add a `requirements.txt` or `environment.yml`.
3. Convert `inference.py` into a standard argparse-based CLI.
4. Remove hardcoded `CUDA_VISIBLE_DEVICES`.
5. Clean up old logs, saved outputs, and `__pycache__`.

## 14. Citation

If this code corresponds to a paper, thesis, or internal research project, this section should be updated with:

- paper title
- authors
- publication year
- paper link
- BibTeX

No clear citation metadata was found in this folder, so this README does not invent one.
