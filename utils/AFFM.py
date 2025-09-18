import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFrequencyFilter(nn.Module):
    """
    自适应频率滤波模块 (AFFM)
    为每个输入实例生成一个自适应的频率域滤波器
    """
    def __init__(self, seq_len, enc_in, hidden_dim=64, rfft=True):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.rfft = rfft
        self.n_freq = seq_len // 2 + 1 if rfft else seq_len

        # 轻量级元网络：从时域信号生成频率滤波器
        # 输入：时域信号 (B, T, N) -> 全局特征 (B, N, hidden_dim)
        self.meta_net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 为每个频率单元预测一个滤波系数（范围[0,1]）
        self.filter_generator = nn.Linear(hidden_dim, self.n_freq)

    def forward(self, x):
        """
        x: input time series (B, T, N)
        returns:
            filtered_x: 滤波后的时域信号（非平稳部分）
            residual_x: 残差信号（平稳部分）
        """
        batch_size, _, n_channels = x.shape
        
        # 1. 计算频域表示
        if self.rfft:
            x_freq = torch.fft.rfft(x, dim=1) # (B, F, N)
        else:
            x_freq = torch.fft.fft(x, dim=1)
        
        # 2. 为每个instance和channel生成自适应滤波器
        # 首先提取时域信号的全局特征
        global_features = self.meta_net(x.transpose(1,2)) # (B, N, hidden_dim)
        
        # 为每个频率生成滤波系数 (B, N, F)
        filter_coeff = torch.sigmoid(self.filter_generator(global_features)).transpose(1,2) # (B, F, N)
        
        # 3. 应用软滤波：不是完全去除，而是衰减
        x_freq_filtered = x_freq * filter_coeff
        
        # 4. 变换回时域
        if self.rfft:
            x_filtered = torch.fft.irfft(x_freq_filtered, dim=1, n=self.seq_len)
        else:
            x_filtered = torch.fft.ifft(x_freq_filtered, dim=1).real
        
        # 5. 计算残差（平稳部分）
        residual_x = x - x_filtered
        
        return residual_x, x_filtered, filter_coeff