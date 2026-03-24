/#!/bin/bash
train_dataset="VIPL"
EPOCH=1 # 20
epoch_val=0
model_S=3
lr=0.00001

# TODO
delta_T=10
numSample=4

comp_c40="c40"
comp_c23="c23"
comp_raw="raw"

bs=1
seq_len=160
MAX_ROUND=1
# FF++
finetune_dataset="FF++"
# exper_name="[c23][FF++]alignment_beta05[epoch30][rm_CCAL][lr1e-3][Fix-Prompt_w=0.3][AffL-a_0.5_b0.1][WarmUp][phy_w0.1-CCTL_w0.0-AFF_w0.5][Mask_config-0.0_0.5_False_0]"
all_test=("DF") #"DF" "F2F" "FS" "NT"
all_cross=("DF") #"DF" "F2F" "FS" "NT"
# comps=("raw" "c23" "c40") #  "c40" "raw"
comps=("c23") #  "c40" "raw"
# 消融實驗

exper_name="[c23]Demo_exper_init_1[Fix-Prompt_w=0.3][AffL-a_0.5_b0.3][WarmUp][phy_w0.1-CCTL_w0.0-AFF_w0.0][Mask_config-0.0_0.5_False_0][Classifier-mlp]"
# all_test=("all")
# all_cross=("all")

loss=("NegPerson") 
warmup=true
# ==========twin_dissim_sim_prompt_set==========
twin_dissim_sim_prompt_set=false
bce_alpha=0.7 #sim
bce_beta=0.3 #dissim
# ==========prompt weight==========
text_prompt_weight=0.5 #0.3
# ==========CCAL loss config==========
CCAL_alpha=0.5
CCAL_beta=0.5
# ==========affinity loss config==========
aff_alpha=0.5
aff_beta=0.5
# ==========Final loss config==========
phy_loss_w=0.1 #0.2
CCTL_w=0.5
aff_loss_w=0.5 #[Final Loss]
# ==========Mask Learning config==========
Mask_prob=0.0 #rPPG video, landmark遮罩機率
Mask_percentage=0.0 #text, rPPG video, landmark遮罩比例
text_prompt_filter=true #true = 完全使用無效Prompt
max_mask_count=0
# EPOCH=25
EPOCH=15 #Test total epoch + 2, 最後一個2是best epoch
classifier_module='mlp' #'mlp','trans_mlp'

for round in $(seq 1 $MAX_ROUND)
do
    echo Round \#"$round"
    for loss in "${loss[@]}"; # MSE SmoothL1 L1 Cosine
    do  
        epoch=$EPOCH

        for test_all in "${all_test[@]}";
        do
            for comp in "${comps[@]}";
            do  
                for cross_subset in "${all_cross[@]}";
                do
                    # test all
                    # python3 test_FLIP_crossattention_all_git.py --exper_name=$exper_name --train_dataset=$train_dataset --finetune_dataset=$finetune_dataset --test_dataset=$finetune_dataset --subset=$test_all --comp_1_loader=$comp_c23 \
                    python3 test_alignment_all.py --exper_name=$exper_name --train_dataset=$train_dataset --finetune_dataset=$finetune_dataset --test_dataset=$finetune_dataset --subset=$test_all --comp_1_loader=$comp_c23 \
                        --comp_3_loader=$comp_raw --comp_2_loader=$comp_c40 --model_S=$model_S --epoch $epoch --bs $bs --seq_len $seq_len --loss $loss --test_comp=$comp --cross_subset=$cross_subset \
                        --Mask_prob=$Mask_prob --Mask_percentage=$Mask_percentage --text_prompt_filter=$text_prompt_filter --max_mask_count=$max_mask_count \
                        --text_prompt_weight=$text_prompt_weight \
                        --CCAL_alpha=$CCAL_alpha --CCAL_beta=$CCAL_beta --aff_alpha=$aff_alpha --aff_beta=$aff_beta \
                        --phy_loss_w=$phy_loss_w --CCTL_w=$CCTL_w --aff_loss_w=$aff_loss_w \
                        --classifier_module=$classifier_module
                done
            done
        done
    done
done
