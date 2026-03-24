import torch
from torch import nn


class LandmarkDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(LandmarkDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def generate_mask(self, landmark, frame):
        position_p = torch.bernoulli(torch.Tensor([1 - self.p]*(landmark//2)))
        return position_p.repeat(1, frame, 2)

    def forward(self, x: torch.Tensor):
        if self.training:
            _, frame, landmark = x.size()
            landmark_mask = self.generate_mask(landmark, frame)
            scale = 1/(1-self.p)
            return x*landmark_mask.to(x.device)*scale
        else:
            return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class LRNet(nn.Module):
    def __init__(self, feature_size=136, lm_dropout_rate=0.1, rnn_unit=32,
                 num_layers=1, rnn_dropout_rate=0,
                 fc_dropout_rate=0.5, res_hidden=64):
        super(LRNet, self).__init__()
        self.hidden_size = rnn_unit
        self.hidden_state = nn.Parameter(torch.randn(2 * num_layers, 1, rnn_unit))
        self.dropout_landmark = LandmarkDropout(lm_dropout_rate)
        self.gru = nn.GRU(input_size=feature_size, hidden_size=rnn_unit,
                          num_layers=num_layers, dropout=rnn_dropout_rate,
                          batch_first=True, bidirectional=True)

        self.dense = nn.Sequential(
            nn.Dropout(fc_dropout_rate),
            Residual(FeedForward(rnn_unit * 2 * num_layers, res_hidden, fc_dropout_rate)),
            nn.Dropout(fc_dropout_rate),

            # MLP-Head
            nn.Linear(rnn_unit * 2 * num_layers, 2)
        )
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout_landmark(x)
        _, hidden = self.gru(x, self.hidden_state.repeat(1, x.shape[0], 1))
        x = torch.cat(list(hidden), dim=1)
        
        # x = self.dense(x)
        
        # exclude the last layer
        for layer in self.dense[0:-1]:
            x = layer(x)
            
            
        # x = self.output(x)
        return x



class DualLRNet(nn.Module):
    def __init__(self, feature_size=136, lm_dropout_rate=0.1, rnn_unit=32,
                 num_layers=1, rnn_dropout_rate=0,
                 fc_dropout_rate=0.5, res_hidden=64,
                 load_pretrained=False, pretrained_comp="", device="cuda:0"):
        super(DualLRNet, self).__init__()
        
        self.g1 = LRNet(feature_size, lm_dropout_rate, rnn_unit, num_layers, rnn_dropout_rate, fc_dropout_rate, res_hidden)
        self.g2 = LRNet(feature_size, lm_dropout_rate, rnn_unit, num_layers, rnn_dropout_rate, fc_dropout_rate, res_hidden)
        if load_pretrained:
            if pretrained_comp == "":
                raise ValueError("pretrained_comp must be specified")
            
            
            # print("Loading LRNet pretrained weights ->", pretrained_comp)
            print("Loading LRNet pretrained weights ->", 'git org weight')
            # self.g1.load_state_dict(torch.load(f"LRNet_weights/g1_{pretrained_comp}.pth", map_location=device)) 
            # self.g2.load_state_dict(torch.load(f"LRNet_weights/g2_{pretrained_comp}.pth", map_location=device))
            self.g1.load_state_dict(torch.load("module/LRNet_org_weights/g1.pth", map_location=device))
            self.g2.load_state_dict(torch.load("module/LRNet_org_weights/g2.pth", map_location=device))
        else:
            print("No pretrained LRNet model is loaded")
       
    
    def forward(self, x1, x2):
        x1 = self.g1(x1)
        # print(f"x1 ={x1.shape}")
        # print(f"x1 ={x1}")
        # 假设 features 是你的特征张量
        std_dev = 0.1  # 较小的标准差值
        noise = torch.randn(x1.size()) * std_dev  # 生成符合标准正态分布的噪声
        noise = noise.to("cuda:0")
        # print(f"noise ={noise.shape}")
        # print(f"noise ={noise}")
        x1_noise = x1 + noise
        # print(f"x1_noise ={x1_noise.shape}")
        # print(f"x1_noise ={x1_noise}")
        x2 = self.g2(x2)
        # print(f"x2 ={x2.shape}")
        noise_1 = torch.randn(x1.size()) * std_dev  # 生成符合标准正态分布的噪声
        noise_1 = noise_1.to("cuda:0")
        x2_noise = x2 + noise_1

        return x1_noise, x2_noise
    


if __name__ == "__main__":
    model = DualLRNet()
    x = torch.randn(16, 300, 136)
    x_diff = torch.randn(16, 299, 136)
    y1, y2 = model(x, x_diff)
    # print(y)
    print(y1.shape)
    print(y2.shape)
    
