import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseDetector1D(nn.Module):
    """
    一维噪声检测器 (类比论文中的 Noise Detection Module)
    功能：利用傅里叶变换(FFT)分析输入序列的频域特性，判断其是否为“噪声主导”。
    """
    def __init__(self, seq_len, hidden_dim=16):
        super().__init__()
        # 频域上的可学习滤波器
        self.freq_conv_real = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.freq_conv_imag = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        
        # 轻量级分类器
        self.classifier = nn.Sequential(
            nn.Linear(seq_len * hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 输出一个0-1之间的概率值，表示“含噪”的可能性
        )
        self.activation = nn.GELU()

    def forward(self, x): # x: [B, 1, L]
        # 1. 傅里叶变换到频域
        x_fft = torch.fft.fft(x, dim=-1)
        x_real, x_imag = x_fft.real, x_fft.imag

        # 2. 在频域应用可学习的滤波器
        filtered_real = self.activation(self.freq_conv_real(x_real))
        filtered_imag = self.activation(self.freq_conv_imag(x_imag))

        # 3. 逆傅里叶变换回时域 (以整合相位和振幅信息)
        filtered_fft = torch.complex(filtered_real, filtered_imag)
        x_reconstructed = torch.fft.ifft(filtered_fft, dim=-1).real

        # 4. 分类器判断是否含噪
        x_flat = x_reconstructed.flatten(start_dim=1)
        noise_prob = self.classifier(x_flat) # [B, 1]
        
        return noise_prob


class FrequencyTemporalDenoiser1D(nn.Module):
    """
    频率-时间双路去噪器 (类比论文中的 Frequency-Spatial Denoising Module)
    功能：结合频域滤波和时域重建，对被判定为“含噪”的序列进行去噪。
    """
    def __init__(self, seq_len, hidden_dim=32):
        super().__init__()
        # --- 频率分支 (用于生成时域注意力掩码) ---
        self.freq_filter_real = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.freq_filter_imag = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)

        # # --- 时间分支 (使用轻量级卷积进行重建) ---
        # self.temporal_conv = nn.Sequential(
        #     nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.GELU(),
        #     nn.Conv1d(hidden_dim, 1, kernel_size=7, padding=3),
        #     nn.BatchNorm1d(1),
        # )

    def forward(self, x): # x: [B, 1, L]
        # --- 频率分支处理 ---
        x_fft = torch.fft.fft(x, dim=-1)
        filtered_real = self.freq_filter_real(x_fft.real)
        filtered_imag = self.freq_filter_imag(x_fft.imag)
        filtered_fft = torch.complex(filtered_real, filtered_imag)
        temporal_mask = torch.fft.ifft(filtered_fft, dim=-1).real
        temporal_mask = torch.sigmoid(temporal_mask) # 生成时域注意力掩码

        # --- 时间分支处理 ---
        # x_reconstructed = self.temporal_conv(x)

        # --- 跨域融合 ---
        # 使用频率分支得到的掩码，来调制时间分支的重建结果
        denoised_output = x * temporal_mask
        
        return denoised_output


class TargetedDenoising1D(nn.Module):
    """
    核心模块：一维定向去噪模块 (TFD for 1D)
    功能：集成噪声检测和双路去噪，实现对输入序列的智能降噪。
    """
    def __init__(self, seq_len, noise_threshold=0.5):
        super().__init__()
        self.seq_len = seq_len
        self.noise_threshold = noise_threshold
        
        self.detector = NoiseDetector1D(seq_len)
        self.denoiser = FrequencyTemporalDenoiser1D(seq_len)

    def forward(self, x): # x: [B, 1, L]
        # 1. 检测噪声概率
        noise_prob = self.detector(x) # [B, 1]

        # 2. 执行去噪 (无论是否含噪都计算，以便于梯度传播)
        x_denoised = self.denoiser(x)

        # 3. 定向应用：根据噪声概率决定是否使用去噪结果
        is_noisy = (noise_prob > self.noise_threshold).view(-1, 1, 1)
        output = torch.where(is_noisy, x_denoised, x) # 如果含噪，用去噪结果；否则用原始输入
        
        return output