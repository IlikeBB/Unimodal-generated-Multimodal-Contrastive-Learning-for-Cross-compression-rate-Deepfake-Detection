#!/bin/bash
train_dataset="VIPL"
finetune_dataset="FF++"

# EPOCH=20
epoch_val=0
model_S=3
lr=0.001

# TODO
delta_T=10 #時間間隔 或 時序相關特徵
numSample=4

comp_c40="c40"
comp_c23="c23"
comp_raw="raw"
train_rate="c23" #raw, c23, c40
bs=4
seq_len=160
MAX_ROUND=1

# FF++
exper_name="Demo_exper_init_1"
all=("DF" "NT" "FS" "F2F")

loss=("NegPerson") 
warmup=true
# ==========twin_dissim_sim_prompt_set==========
twin_dissim_sim_prompt_set=false  #是否啟用雙向Text Prompt進行混合訓練
bce_alpha=0.7            # sim 
bce_beta=0.3             # dissim
# ==========prompt weight==========
text_prompt_weight=0.3 #0.3
# ==========affinity loss config==========
aff_alpha=0.5 #0.5            # [Diagonal Loss] Affinity Loss Alpha Weight Rate 
aff_beta=0.3 #0.5             # [Cross-Modal Loss] Affinity Loss Beta Weight Rate 
# ==========Final loss config==========
phy_loss_w=0.1           # Physiological Loss Weight Rate
CCTL_w=0.0 #0.5               # Contrastive Compression Alignment Loss Weight Rate
aff_loss_w=0.5 #0.5           # Affinity Loss Weight Rate
# ==========Mask Learning config==========
Mask_prob=0.0            # rPPG video, landmark遮罩機率
Mask_percentage=0.5      # text, rPPG video, landmark遮罩比例
text_prompt_filter=false # [Train] true = 使用BMVC Eval Prompt; false = 使用Text Prompt Template
max_mask_count=2         # 最多遮幾個Input Data: 0 = 都不遮、1=遮rPPG或Land、2=遮rPPG和Land

test_all=("all")
EPOCH=3
classifier_module='mlp' #'mlp','trans_mlp'
# # script 
train_rates=("c23")
for round in $(seq 1 $MAX_ROUND)
do
    echo Round \#"$round"
    for loss in "${loss[@]}"; # MSE SmoothL1 L1 Cosine
    do  
        for train_rate in "${train_rates[@]}";
        do
            for subset in "${all[@]}";
            do  
                python3 train_main.py --exper_name=$exper_name --train_dataset=$train_dataset --finetune_dataset=$finetune_dataset --subset=$subset --comp_1_loader=$comp_c23 --comp_2_loader=$comp_c40 \
                    --comp_3_loader=$comp_raw --model_S=$model_S --epoch $EPOCH --bs $bs --lr $lr --seq_len $seq_len --loss $loss --continue_model \
                    --train_rate=$train_rate --warmup=$warmup\
                    --Mask_prob=$Mask_prob --Mask_percentage=$Mask_percentage --text_prompt_filter=$text_prompt_filter --max_mask_count=$max_mask_count \
                    --text_prompt_weight=$text_prompt_weight \
                    --aff_alpha=$aff_alpha --aff_beta=$aff_beta\
                    --phy_loss_w=$phy_loss_w --CCTL_w=$CCTL_w --aff_loss_w=$aff_loss_w \
                    --classifier_module=$classifier_module
            done
        done
    done
done
