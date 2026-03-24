import torch
import math
# =======================check bias=======================S
# bn_params = []
# bias_params = []
# non_bn_non_bias_params = []

# for name, param in model.named_parameters():
#     if 'bn' in name.lower() or 'batchnorm' in name.lower():
#         bn_params.append(param)
#     elif 'bias' in name.lower():
#         bias_params.append(param)
#     else:
#         non_bn_non_bias_params.append(param)
# =======================check bias=======================E
def optim_tools_opt(model = None, args = None):
    # 分離需要正則化和不需要正則化的參數
    decay_params = []  # 需要 weight decay 的參數
    no_decay_params = []  # 不需要 weight decay 的參數

    for name, param in model.named_parameters():
        if param.requires_grad:
            # 偏置項和 BatchNorm 不做 weight decay
            if 'bias' in name or 'bn' in name or 'batchnorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # 優化器設置
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},  # 權重正則化
        {'params': no_decay_params, 'weight_decay': 0.0}  # 不正則化
    ], lr=args.lr)
    return optimizer
