from prompt_rppg import spoof_templates, real_templates
import random, torch, sys
from torch import nn
from torch.nn import functional as F
import clip #type: ignore

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


class PAD_Classifier(nn.Module):
    def __init__(self, PAE_net, downstream_net, model_text, args_set = None,
                 dim_text = 512, text_embed_dim = 512, embed_dim = 320,
                 training = True):
        super(PAD_Classifier, self).__init__()
        self.args_set = args_set
        print("self.args = ", self.args_set)
        self.training = training
        # backbone module LR_NET, rPPG_NET, CLIP_TEXT_NET
        self.Extractor = PAE_net
        self.downstream = downstream_net
        self.text_encode = model_text
        
        # other module
        if self.args_set.classifier_module=='mlp': #type: ignore
            self.final_classifier = nn.Linear(embed_dim*3, 2)
        
        elif self.args_set.classifier_module=='trans_mlp': #type: ignore
            num_heads = self.args_set.classifier_module_heads #type: ignore
            num_layers = self.args_set.classifier_module_layers #type: ignore
            
            self.transformer_classifier = nn.Transformer(
                d_model=embed_dim,
                nhead=num_heads,
                num_encoder_layers=num_layers,
                num_decoder_layers=0,  
                dim_feedforward=embed_dim * 4,
                activation="relu",
            )
            self.final_classifier = nn.Linear(embed_dim, 2)
        
        
        # rPPG embedding dimension
        # self.classifier_rPPG_attention = nn.Linear(80, 2)
        # size 128
        self.rPPG_F_downsample_Linear_128 = nn.Linear((640*96*2), embed_dim//2)
        # self.classifier_rPPG_attention_stack = nn.Linear(122880, 160)
        # size 64
        self.rPPG_F_downsample_Linear_64 = nn.Linear((160*96*2), embed_dim//2)
        # self.classifier_rPPG_attention_stack_64 = nn.Linear(30720, 160)
        # size 32
        self.rPPG_F_downsample_Linear_32 = nn.Linear((40*96*2), embed_dim//2)
        # self.classifier_rPPG_attention_stack_32 = nn.Linear(7680, 160)
        self.rPPG_x_upsample_Linear = nn.Linear(160*2, embed_dim)
        
        # LR embedding dimension
        self.LR_upsample_Linear = nn.Linear(64*2, embed_dim)
        
        # CLIP embedding dimension
        self.text_downsample_linear = nn.Linear(512, embed_dim)
        
        # init feature alingment
        self.custome_feature_alignment = FeatureAlignmentModule(training=self.training)
        
        self.mask_prob = self.args_set.Mask_prob #type: ignore
        self.mask_percentage = self.args_set.Mask_percentage #type: ignore
        self.max_mask_count = self.args_set.max_mask_count #type: ignore
        self.text_prompt_filter = self.args_set.text_prompt_filter #type: ignore
        
        
        
    def forward(self, video, landmark, landmark_diff, sim_or_dis, size, eval_step=False):
        # print("Video shape: ",video.shape)                                  #torch.Size([1, 3, 160, 128, 128])
        # print("landmark norm shape: ", landmark.shape)                      #torch.Size([1, 160, 136])
        # print("landmark diff shape: ", landmark_diff.shape)                 #torch.Size([1, 159, 136])
        def extract_and_displace_all_points(tensor, shift_x, shift_y):
            noise = torch.empty_like(tensor).uniform_(shift_x, shift_y)
            return tensor + noise

        def apply_random_noise(landmark, start, end, shift_x, shift_y):
            noise = torch.empty_like(landmark[:, :, start:end]).uniform_(shift_x, shift_y)
            landmark[:, :, start:end] += noise
            return landmark

        
        # landmark point displace
        if size == 32:
            landmark = extract_and_displace_all_points(landmark, -0.01, 0.01)
            landmark = apply_random_noise(landmark, 98, 136, -0.006, +0.006)  # lip
            landmark = apply_random_noise(landmark, 74, 96, -0.005, +0.005)  # eye
            landmark = apply_random_noise(landmark, 56, 72, -0.001, +0.001)  # nose
        elif size == 64:
            landmark = extract_and_displace_all_points(landmark, -0.0005, 0.0005)
            landmark = apply_random_noise(landmark, 98, 136, -0.0006, +0.0006)  # lip
            landmark = apply_random_noise(landmark, 74, 96, -0.0005, +0.0005)  # eye
            landmark = apply_random_noise(landmark, 56, 72, -0.0001, +0.0001)  # nose

        # random mask module
        random_count = 0
        if self.max_mask_count!=0:
            if self.training:
                prob_mask_set = random.uniform(self.mask_percentage, self.mask_percentage*2)
            else:
                prob_mask_set = self.mask_percentage
            if torch.rand(1).item() < self.mask_prob  and random_count < self.max_mask_count:
                # video shape: torch.Size([1, 3, 160, 128, 128])
                video = random_mask_modality(video, prob= prob_mask_set)
                random_count+=1
                
            if torch.rand(1).item() < self.mask_prob and random_count < self.max_mask_count:
                # landmark shape: torch.Size([1, 160, 136])
                landmark = random_mask_modality(landmark, prob=prob_mask_set)
                landmark_diff = random_mask_modality(landmark_diff, prob=prob_mask_set)
                random_count+=1
        # LR Net
        PAE_feature, PAE_feature_diff = self.Extractor(landmark, landmark_diff)
        
        # print("LR orgi shape: ", PAE_feature.shape)                      #torch.Size([1, 159, 136])
        # print("LR diff shape: ", PAE_feature_diff.shape)                 #torch.Size([1, 159, 136])
        # rppg
        gra_sharp = 2.0
        
        # rPPG Net
        rppg_x ,score1_x,score2_x,score3_x,feature_1,feature_2 = self.downstream(video,gra_sharp,size) #　rppg_x ,score1_x,score2_x,score3_x
        # print("rppg output shape: ", rppg_x.shape)                         #torch.Size([1, 159, 136])
        # print("feature_1 output shape: ", feature_1.shape)                 #torch.Size([1, 159, 136])
        # print("feature_2 output shape: ", feature_2.shape)                 #torch.Size([1, 159, 136])
        
        
        # text tokenize 
        # print(f"self.training = {self.training} self.text_prompt_filter = {self.text_prompt_filter}")
        if video.shape[0]!=1:
            real_text = []
            fake_text = []
            for idx_bs in range(video.shape[0]):
                if self.training==False and self.text_prompt_filter==False:
                    combined_templates = real_templates + spoof_templates
                    real_text.append(random.choice(combined_templates))
                    fake_text.append(random.choice(combined_templates))
                elif self.text_prompt_filter==False and eval_step==False:
                    if random.random() < self.mask_percentage:  # 30% 機率選擇無效 prompt
                        real_text.append(random.choice(real_templates[6:]))
                        fake_text.append(random.choice(spoof_templates[6:]))
                    else:  # 70% 機率選擇有效 prompt
                        real_text.append(random.choice(real_templates[:6]))
                        fake_text.append(random.choice(spoof_templates[:6]))
                    
                elif self.text_prompt_filter or eval_step==True:  # 測試階段
                    # 測試階段始終使用固定的無效 prompt，避免 prompt 對測試結果的干擾
                    real_text.append("Is real or fake?")
                    fake_text.append("Is real or fake?")
                    # real_text = "None"
                    # fake_text = "None"
                else:
                    pass
        else:
            if self.training==False and self.text_prompt_filter==False:
                combined_templates = real_templates + spoof_templates
                real_text = random.choice(combined_templates)
                fake_text = random.choice(combined_templates)        
            elif self.text_prompt_filter==False:
                if random.random() < self.mask_percentage:  # 30% 機率選擇無效 prompt
                    real_text = random.choice(real_templates[6:])
                    fake_text = random.choice(spoof_templates[6:])
                else:  # 70% 機率選擇有效 prompt
                    real_text = random.choice(real_templates[:6])
                    fake_text = random.choice(spoof_templates[:6])
                
            elif self.text_prompt_filter:  # 測試階段
                # 測試階段始終使用固定的無效 prompt，避免 prompt 對測試結果的干擾
                real_text = "Is real or fake?"
                fake_text = "Is real or fake?"
                # real_text = "None"
                # fake_text = "None"
            else:
                pass
        
        if sim_or_dis == 1:
            texts = clip.tokenize(real_text).cuda(non_blocking=True) # tokenize
            other_texts = clip.tokenize(fake_text).cuda(non_blocking=True)
        elif sim_or_dis == -1:
            texts = clip.tokenize(fake_text).cuda(non_blocking=True) #tokenize
            other_texts = clip.tokenize(real_text).cuda(non_blocking=True)
        else:
            print("Waring:self.sim_or_dis")

        # print("prompt text shape: ", texts.shape, other_texts.shape)  
        # #prompt text shape:  torch.Size([2, 77]) torch.Size([2, 77])
        # embed with text encoder
        class_embeddings = self.text_encode.encode_text(texts) 
        # class_embeddings = class_embeddings.mean(dim=0) 
        class_embeddings_other = self.text_encode.encode_text(other_texts) 
        # class_embeddings_other = class_embeddings_other.mean(dim=0) 

        # normalized features
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings_other = class_embeddings_other / class_embeddings_other.norm(dim=-1, keepdim=True)
        
        
        # stack rppg feature
        downstream_features_detach = torch.cat([feature_1, feature_2],dim=1)
        B_, _, _ = downstream_features_detach.shape
        downstream_features_detach = downstream_features_detach.view(B_,-1)
        # print(f"rppg_x: {rppg_x.shape} || rppg_f1: {feature_1.shape} || rppg_f2: {feature_2.shape}")
        # rppg_x: torch.Size([1, 160]) || rppg_f1: torch.Size([1, 640, 96]) || rppg_f2: torch.Size([1, 640, 96])
        # rppg concate => 160
        # print(f"LR_1: {PAE_feature.shape} || LR_2: {PAE_feature_diff.shape}")
        # LR_1: torch.Size([1, 64]) || LR_2: torch.Size([1, 64])
        # print(f"CLIP_text_curr: {class_embeddings.shape} || CLIP_text_other: {class_embeddings_other.shape} \n\n")
        # CLIP_text_curr: torch.Size([1, 512]) || CLIP_text_other: torch.Size([1, 512])
        
        #####################################################################################
        # feature embedding(upsample or downsample)
        # LR
        feature_landmark = torch.cat([PAE_feature, PAE_feature_diff],dim=1)
        feature_landmark = self.LR_upsample_Linear(feature_landmark)
        # print("Embedding landmakr shape: ",feature_landmark.shape)
        # Embedding landmakr shape:  torch.Size([1, 320])
        # rPPG
        if size==128:
            feature_rPPG_F = self.rPPG_F_downsample_Linear_128(downstream_features_detach)
        elif size==64:
            feature_rPPG_F = self.rPPG_F_downsample_Linear_64(downstream_features_detach)
        elif size==32:
            feature_rPPG_F = self.rPPG_F_downsample_Linear_32(downstream_features_detach)
        else:
            print("Waring: fail rPPG size input")
        feature_rPPG_xF = torch.cat([rppg_x, feature_rPPG_F],dim=1)
        feature_rPPG = self.rPPG_x_upsample_Linear(feature_rPPG_xF)
        # print("Embedding rPPG shape: ", feature_rPPG.shape)
        # Embedding rPPG shape:  torch.Size([1, 320])
        # CLIP_text
        feature_clip_text_curr = self.text_downsample_linear(class_embeddings)
        if video.shape[0]==1:
            feature_clip_text_curr = feature_clip_text_curr.unsqueeze(0)
        # print("Embedding CLIPtext shape: ", feature_clip_text_curr.shape, feature_clip_text_other.shape)
        # Embedding CLIPtext shape:  torch.Size([1, 320]) torch.Size([1, 320])
        #####################################################################################
        # feature alignment for both embeddings
        # print(feature_clip_text_curr.shape, feature_rPPG.shape, feature_landmark.shape)
        r_text_emb_curr, r_rppg_emb_curr, r_lmk_emb_curr, sim_tr_curr, sim_tl_curr, sim_rl_curr = \
            self.custome_feature_alignment(feature_clip_text_curr, feature_rPPG, feature_landmark)
        #####################################################################################
        # classifier process
        if len(r_text_emb_curr.shape)!=2:
            r_text_emb_curr = r_text_emb_curr.squeeze(0)
        if self.args_set.classifier_module=='mlp': #type: ignore
            final_class_feature_curr = torch.cat([r_text_emb_curr, r_rppg_emb_curr, r_lmk_emb_curr], dim=1)
        elif self.args_set.classifier_module=='trans_mlp': #type: ignore
            multimodal_features_curr = torch.stack([r_text_emb_curr, r_rppg_emb_curr, r_lmk_emb_curr], dim=0)  # Shape: [3, 1, 320]
            transformer_output_curr = self.transformer_classifier(multimodal_features_curr)
            final_class_feature_curr = transformer_output_curr.mean(dim=0)  # Shape: [1, 320]
        final_output_curr = self.final_classifier(final_class_feature_curr)

        if self.args_set.twin_dissim_sim_prompt_set: #type: ignore
            feature_clip_text_other = self.text_downsample_linear(class_embeddings_other)
            if video.shape[0]==1:
                feature_clip_text_other = feature_clip_text_other.unsqueeze(0)
            r_text_emb_other, r_rppg_emb_other, r_lmk_emb_other, sim_tr_other, sim_tl_other, sim_rl_other = \
            self.custome_feature_alignment(feature_clip_text_other, feature_rPPG, feature_landmark)

            if self.args_set.classifier_module=='mlp': #type: ignore
                final_class_feature_other = torch.cat([r_text_emb_other, r_rppg_emb_other, r_lmk_emb_other], dim=1)
            elif self.args_set.classifier_module=='trans_mlp': #type: ignore
                multimodal_features_other = torch.stack([r_text_emb_other, r_rppg_emb_other, r_lmk_emb_other], dim=0)  # Shape: [3, 1, 320]
                transformer_output_other = self.transformer_classifier(multimodal_features_other)
                final_class_feature_other = transformer_output_other.mean(dim=0)  # Shape: [1, 320]
            final_output_other = self.final_classifier(final_class_feature_other)    
            
            return final_output_curr, final_output_other, rppg_x, \
                sim_tr_curr, sim_tl_curr, sim_rl_curr, \
                sim_tr_other, sim_tl_other, sim_rl_other, \
                r_text_emb_curr, r_rppg_emb_curr, r_lmk_emb_curr, \
                r_text_emb_other, r_rppg_emb_other, r_lmk_emb_other
                
        return final_output_curr, rppg_x, \
            sim_tr_curr, sim_tl_curr, sim_rl_curr, \
            r_text_emb_curr, r_rppg_emb_curr, r_lmk_emb_curr
        # return final_class_feature, rppg_x, sim_tr, sim_tl, sim_rl, r_text_emb, r_rppg_emb, r_lmk_emb
        # return final_class_feature, rppg_x, sim_tr, sim_tl, sim_rl
        #####################################################################################

class FeatureAlignmentModule(nn.Module):
    """
    三個模態: text, rppg, landmark
    各自輸出特徵之後，先映射到一個共享空間再做顯式對齊 (Affinity)。
    同時也內建 Masking Learning 的機制。
    """
    # def __init__(self,
    #              dim_text=768,
    #              dim_rppg=256,
    #              dim_landmark=128,
    #              embed_dim=256,
    #              mask_prob=0.3):
    def __init__(self,
                 training = False,
                 mask_prob=0.5):

        super(FeatureAlignmentModule, self).__init__()
        
        # # 投影層: 把每個模態都映射到同樣的維度 (embed_dim)
        # self.proj_text = nn.Linear(dim_text, embed_dim)
        # self.proj_rppg = nn.Linear(dim_rppg, embed_dim)
        # self.proj_landmark = nn.Linear(dim_landmark, embed_dim)
        self.training = training
        
        # 可選的非線性
        self.act = nn.ReLU()
        
        # 可學習參數 (若要對 "cosine相似度" 做 affine 轉換)
        # 例如只示範三個 pair: (text-rppg), (text-landmark), (rppg-landmark)
        self.w_tr = nn.Parameter(torch.tensor(1.0))
        self.b_tr = nn.Parameter(torch.tensor(0.0))
        
        self.w_tl = nn.Parameter(torch.tensor(1.0))
        self.b_tl = nn.Parameter(torch.tensor(0.0))
        
        self.w_rl = nn.Parameter(torch.tensor(1.0))
        self.b_rl = nn.Parameter(torch.tensor(0.0))
        
        # Masking 的機率
        self.mask_prob = mask_prob

    # def forward(self, feat_text, feat_rppg, feat_landmark):
    def forward(self, feat_text, feat_rppg, feat_landmark):
        """
        feat_text: shape [B, dim_text]
        feat_rppg: shape [B, dim_rppg]
        feat_landmark: shape [B, dim_landmark]
        回傳對齊後的 embedding 以及模態間的相似度
        """
        B = feat_text.shape[0]
        
        # 1) 投影到共享空間
        # text_emb = self.act(self.proj_text(feat_text))      # [B, embed_dim]
        text_emb = self.act(feat_text)     # [B, embed_dim]
        # rppg_emb = self.act(self.proj_rppg(feat_rppg))      # [B, embed_dim]
        rppg_emb = self.act(feat_rppg)      # [B, embed_dim]
        # lmk_emb  = self.act(self.proj_landmark(feat_landmark)) # [B, embed_dim]
        lmk_emb  = self.act(feat_landmark) # [B, embed_dim]
        sim_tr = F.cosine_similarity(text_emb, rppg_emb, dim=1)
        sim_tr = torch.sigmoid(self.w_tr * sim_tr + self.b_tr)
        
        sim_tl = F.cosine_similarity(text_emb, lmk_emb, dim=1)
        sim_tl = torch.sigmoid(self.w_tl * sim_tl + self.b_tl)
        
        sim_rl = F.cosine_similarity(rppg_emb, lmk_emb, dim=1)
        sim_rl = torch.sigmoid(self.w_rl * sim_rl + self.b_rl)
        
        # 最終我們可以回傳
        # 1) 對齊後的 embedding (text_emb, rppg_emb, lmk_emb)
        # 2) 各 pair 的相似度 sim_tr, sim_tl, sim_rl
        # print("sim_tr, sim_tl, sim_rl: ", sim_tr.shape, sim_tl.shape, sim_rl.shape)
        # print(sim_tr, sim_tl, sim_rl)
        return text_emb, rppg_emb, lmk_emb, sim_tr, sim_tl, sim_rl