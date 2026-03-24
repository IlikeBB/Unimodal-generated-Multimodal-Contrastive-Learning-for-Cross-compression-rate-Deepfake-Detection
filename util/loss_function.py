import torch
from torch.nn import functional as F
from torch import nn

def build_affinity_matrix(feat_text, feat_rppg, feat_landmark):
    """
    構建親和矩陣，形狀為 [B, 3, 3]
    feat_text, feat_rppg, feat_landmark: shape = [B, D]
    """
    # 堆疊多模態特徵: [B, 3, D]
    modal_list = torch.stack([feat_text, feat_rppg, feat_landmark], dim=1)
    # print("build modal_list shape: ", modal_list.shape)
    # L2 正規化
    normed = F.normalize(modal_list, p=2, dim=-1)  # [B, 3, D]

    # 計算親和矩陣: [B, 3, 3]
    affinity = torch.bmm(normed, normed.transpose(1, 2))
    return affinity

def explicit_affinity_loss(affinity_mat, alpha=0.5, beta=0.5):
    """
    基於親和矩陣計算顯式損失
    affinity_mat: shape = [B, 3, 3]
    alpha, beta: 控制對角線與模態間損失的權重
    """
    # 1. 對角線損失 (Diagonal Loss)
    diag_loss = torch.mean((1 - torch.diagonal(affinity_mat, dim1=1, dim2=2))**2)

    # 2. 模態間損失 (Cross-Modal Loss)
    cross_loss = torch.mean((1 - affinity_mat[:, 0, 1])**2) + \
                 torch.mean((1 - affinity_mat[:, 1, 2])**2) + \
                 torch.mean((1 - affinity_mat[:, 0, 2])**2)

    # 3. 總損失
    total_loss = alpha * diag_loss + beta * cross_loss
    return total_loss



def random_mask_modality(features, prob=0.2):
    """
    隨機遮蔽模態特徵，模擬弱模態失效。
    features: shape = [B, D]
    prob: 遮蔽概率
    """
    if prob <= 0.0:
        return features
    
    if len(features.shape)!=3:
        B, C, T, H, W = features.shape
        # 生成遮蔽掩碼，形狀為 [B, T]
        mask = (torch.rand(B, T, device=features.device) < prob).float()  # 每個 frame 有 prob 的概率被遮蔽

        # 擴展掩碼到 [B, 1, T, 1, 1]，與 features 對齊
        mask = mask.view(B, 1, T, 1, 1)

        # 對應位置的 frame 被設為 0
        return features * (1.0 - mask)
    else:
        mask = (torch.rand(features.shape, device=features.device) < prob).float()
        return features * (1.0 - mask)

# ------------------------Affinity　Matrix------------------------------

def build_affinity_matrix(feat_text, feat_rppg, feat_landmark):
    """
    構建親和矩陣，形狀為 [B, 3, 3]
    feat_text, feat_rppg, feat_landmark: shape = [B, D]
    """
    # 堆疊多模態特徵: [B, 3, D]
    modal_list = torch.stack([feat_text, feat_rppg, feat_landmark], dim=1)
    # print("build modal_list shape: ", modal_list.shape)
    # L2 正規化
    normed = F.normalize(modal_list, p=2, dim=-1)  # [B, 3, D]

    # 計算親和矩陣: [B, 3, 3]
    affinity = torch.bmm(normed, normed.transpose(1, 2))
    return affinity

def explicit_affinity_loss(affinity_mat, alpha=0.5, beta=0.5):
    """
    基於親和矩陣計算顯式損失
    affinity_mat: shape = [B, 3, 3]
    alpha, beta: 控制對角線與模態間損失的權重
    """
    # 1. 對角線損失 (Diagonal Loss)
    diag_loss = torch.mean((1 - torch.diagonal(affinity_mat, dim1=1, dim2=2))**2)

    # 2. 模態間損失 (Cross-Modal Loss)
    cross_loss = torch.mean((1 - affinity_mat[:, 0, 1])**2) + \
                 torch.mean((1 - affinity_mat[:, 1, 2])**2) + \
                 torch.mean((1 - affinity_mat[:, 0, 2])**2)

    # 3. 總損失
    total_loss = alpha * diag_loss + beta * cross_loss
    return total_loss