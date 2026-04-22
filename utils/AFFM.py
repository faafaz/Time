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
        # x: (B, T, N)
        batch_size, seq_len, num_nodes = x.shape
        
        # 1. 提取时域全局特征
        global_features = self.meta_net(x.transpose(1, 2))  # (B, N, filter_dim)
        
        # 2. 生成基础高通滤波器（保留高频，衰减低频）
        # 频率长度 = (seq_len // 2) + 1 (rfft的输出长度)
        freq_length = (seq_len // 2) + 1
        base_filter = torch.ones(batch_size, freq_length, num_nodes, device=x.device)
        # 低频部分设为0（衰减），高频部分设为1（保留）
        base_filter[:, :freq_length//2, :] = 0  # 低频部分衰减
        
        # 3. 通过时域特征生成自适应权重
        filter_coeff = torch.sigmoid(self.filter_generator(global_features)).transpose(1, 2)  # (B, F, N)
        
        # 4. 应用自适应高通滤波
        # 确保整体是高通，但允许自适应调整
        filter_coeff = base_filter * filter_coeff
        
        # 5. 频域转换
        x_freq = torch.fft.rfft(x, dim=1)  # (B, F, N)
        
        # 6. 应用滤波
        x_freq_filtered = x_freq * filter_coeff  # 保留高频
        x_high_freq = torch.fft.irfft(x_freq_filtered, dim=1)  # (B, T, N)
        
        # 7. 低频部分 = 原始信号 - 高频部分
        x_low_freq = x - x_high_freq
        
        return x_high_freq, x_low_freq, filter_coeff