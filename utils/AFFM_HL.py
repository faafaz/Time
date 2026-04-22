import torch
import torch.nn as nn
import torch.nn.functional as F

class AFFM_HL(nn.Module):
    """
    改进的AFFM模块，专门用于高频-低频分离
    """
    def __init__(self, seq_len, enc_in, hidden_dim=64, rfft=True, 
                 low_freq_ratio=0.2, dynamic_cutoff=True):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.rfft = rfft
        self.n_freq = seq_len // 2 + 1 if rfft else seq_len
        self.low_freq_ratio = low_freq_ratio  # 低频占比的初始值
        self.dynamic_cutoff = dynamic_cutoff  # 是否动态调整截止频率
        
        # 元网络：分析信号特征，生成频率分割参数
        self.meta_net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        if dynamic_cutoff:
            # 动态生成截止频率
            self.cutoff_generator = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # 输出0-1之间的值，表示截止频率的位置
            )
        else:
            # 固定截止频率
            self.register_buffer('fixed_cutoff', 
                                torch.tensor([low_freq_ratio]))
        
        # 低频和高频滤波器的平滑参数
        self.smoothness = nn.Parameter(torch.tensor(5.0))
        
    def create_filters(self, cutoff_ratio, batch_size, n_channels):
        """
        创建低频和高频滤波器
        """
        # 创建频率轴 [0, 1, 2, ..., n_freq-1]
        freq_axis = torch.linspace(0, 1, self.n_freq, 
                                  device=cutoff_ratio.device)
        freq_axis = freq_axis.view(1, self.n_freq, 1).expand(
            batch_size, self.n_freq, n_channels)
        
        # 计算低频滤波器（低通）
        low_pass_filter = torch.sigmoid(
            self.smoothness * (cutoff_ratio.unsqueeze(1) - freq_axis))
        
        # 高频滤波器是高通的，即1减去低通滤波器
        high_pass_filter = 1 - low_pass_filter
        
        return low_pass_filter, high_pass_filter
        
    def forward(self, x):
        batch_size, _, n_channels = x.shape
        
        # 1. 计算频域表示
        if self.rfft:
            x_freq = torch.fft.rfft(x, dim=1)
        else:
            x_freq = torch.fft.fft(x, dim=1)
        
        # 2. 提取全局特征并生成截止频率
        global_features = self.meta_net(x.transpose(1, 2))  # (B, N, hidden_dim)
        
        if self.dynamic_cutoff:
            # 动态生成截止频率比例
            cutoff_ratio = self.cutoff_generator(global_features.mean(dim=1))  # (B, 1)
        else:
            # 使用固定的截止频率比例
            cutoff_ratio = self.fixed_cutoff.expand(batch_size, 1)
        
        # 3. 创建低频和高频滤波器
        low_pass_filter, high_pass_filter = self.create_filters(
            cutoff_ratio, batch_size, n_channels)
        
        # 4. 应用滤波器
        x_low_freq = x_freq * low_pass_filter
        x_high_freq = x_freq * high_pass_filter
        
        # 5. 逆变换回时域
        if self.rfft:
            x_low = torch.fft.irfft(x_low_freq, dim=1, n=self.seq_len)
            x_high = torch.fft.irfft(x_high_freq, dim=1, n=self.seq_len)
        else:
            x_low = torch.fft.ifft(x_low_freq, dim=1).real
            x_high = torch.fft.ifft(x_high_freq, dim=1).real
        
        return x_low, x_high, cutoff_ratio