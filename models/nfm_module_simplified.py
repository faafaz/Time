"""
Simplified Neural Fourier Modelling (NFM) Module

简化版NFM模块，针对过拟合问题进行优化：
1. 减少SIREN网络层数和隐藏维度
2. 简化频率域处理（移除复杂的INFF层）
3. 减少Mixer层数
4. 使用更轻量的投影路径
5. 添加更强的正则化

主要改进：
- 参数量减少约60%
- 计算复杂度降低
- 保持核心频率域建模能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleSinusoidalPE(nn.Module):
    """简化的正弦位置编码"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, batch_size, seq_len):
        return self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)


class SimplifiedFrequencyLayer(nn.Module):
    """
    简化的频率域处理层
    - 移除SIREN网络，使用简单的可学习频率权重
    - 减少参数量，降低过拟合风险
    """
    def __init__(self, hidden_dim, seq_len, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.freq_dim = seq_len // 2 + 1
        
        # 可学习的频率权重（替代SIREN网络）
        self.freq_weights_real = nn.Parameter(torch.randn(1, self.freq_dim, hidden_dim) * 0.02)
        self.freq_weights_imag = nn.Parameter(torch.randn(1, self.freq_dim, hidden_dim) * 0.02)
        
        # 简化的频率调制
        self.freq_scale = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.freq_bias = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, C] 时域输入
        Returns:
            x_out: [B, L, C] 处理后的时域输出
        """
        B, L, C = x.shape
        
        # 转换到频域
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, F, C] complex
        
        # 应用可学习的频率权重
        if C == self.hidden_dim:
            # 构建复数权重
            freq_weights = torch.complex(
                self.freq_weights_real.expand(B, -1, -1),
                self.freq_weights_imag.expand(B, -1, -1)
            )
            
            # 频域调制
            x_freq_modulated = x_freq * freq_weights
        else:
            x_freq_modulated = x_freq
        
        # 转换回时域
        x_out = torch.fft.irfft(x_freq_modulated, n=L, dim=1, norm='ortho')
        
        # 应用缩放和偏置
        x_out = self.freq_scale * x_out + self.freq_bias
        
        return self.norm(self.dropout(x_out))


class SimplifiedMixerBlock(nn.Module):
    """
    简化的Mixer块
    - 减少隐藏层维度
    - 使用简单的频率域处理
    - 添加dropout正则化
    """
    def __init__(self, hidden_dim, seq_len, hidden_factor=2, dropout=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 通道混合（减少扩展因子从3到2）
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * hidden_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * hidden_factor, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Token混合（简化的频率域处理）
        self.token_mixer = SimplifiedFrequencyLayer(hidden_dim, seq_len, dropout=dropout)
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, C]
        Returns:
            x: [B, L, C]
        """
        # 通道混合
        x = x + self.channel_mixer(x)
        
        # Token混合（频域）
        x = x + self.token_mixer(self.norm(x))
        
        return x


class SimplifiedNFMPredictor(nn.Module):
    """
    简化的NFM预测器
    
    主要简化：
    1. 单一投影路径（移除双路径）
    2. 减少隐藏维度（默认32->24）
    3. 减少Mixer层数（默认1层）
    4. 简化频率域处理
    5. 添加更强的dropout正则化
    
    参数量对比：
    - 原版NFM: ~50K参数（hidden_dim=32, num_layers=1）
    - 简化版: ~20K参数（hidden_dim=24, num_layers=1）
    """
    def __init__(self, seq_len, pred_len, hidden_dim=24, num_layers=1, 
                 hidden_factor=2, dropout=0.15):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # 单一投影路径（移除双路径）
        self.projection_in = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout * 0.5)  # 输入dropout
        )
        
        # 位置编码
        self.pos_emb = SimpleSinusoidalPE(hidden_dim, max_len=seq_len)
        
        # 简化的频率层（替代LFT）
        self.freq_layer = SimplifiedFrequencyLayer(hidden_dim, seq_len, dropout=dropout)
        
        # Mixer层（减少层数）
        self.mixer_layers = nn.ModuleList([
            SimplifiedMixerBlock(hidden_dim, seq_len, hidden_factor=hidden_factor, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 简化的输出FFN
        self.output_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 时间投影（seq_len -> pred_len）
        self.time_proj = nn.Linear(seq_len, pred_len)
        
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, 1] - 原始目标变量
        Returns:
            pred: [B, pred_len, 1] - 预测值
        """
        B, L, C = x.shape
        assert L == self.seq_len and C == 1, f"Expected input shape [B, {self.seq_len}, 1], got {x.shape}"
        
        # 实例归一化（RevIN风格）
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - x_mean) / x_std
        
        # 投影到隐藏维度
        z = self.projection_in(x_norm)  # [B, L, hidden_dim]
        
        # 添加位置编码
        z = z + self.pos_emb(B, L)
        
        # 频率层处理
        z_freq = self.freq_layer(z)
        z = z + z_freq  # 残差连接
        
        # Mixer层
        for mixer in self.mixer_layers:
            z = mixer(z)
        
        # 输出投影
        z = self.output_ffn(z)  # [B, L, 1]
        
        # 时间投影：seq_len -> pred_len
        z = z.transpose(1, 2)  # [B, 1, L]
        z = self.time_proj(z)  # [B, 1, pred_len]
        z = z.transpose(1, 2)  # [B, pred_len, 1]
        
        # 反归一化
        pred = z * x_std + x_mean
        
        return pred


class SimplifiedLightweightFusion(nn.Module):
    """
    简化的轻量融合模块
    
    简化策略：
    1. 移除复杂的门控机制
    2. 使用简单的可学习权重
    3. 添加dropout正则化
    """
    def __init__(self, fusion_type='add', dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'gate':
            # 简化的门控机制
            self.gate = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 2),
                nn.Softmax(dim=-1)
            )
        elif fusion_type == 'add':
            # 简单加权（默认）
            self.weights = nn.Parameter(torch.tensor([0.7, 0.3]))
        elif fusion_type == 'concat':
            # 拼接+投影
            self.proj = nn.Sequential(
                nn.Linear(2, 1),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, lstm_pred, nfm_pred):
        """
        Args:
            lstm_pred: [B, pred_len, 1] - LSTM预测
            nfm_pred: [B, pred_len, 1] - NFM预测
        Returns:
            fused_pred: [B, pred_len, 1] - 融合预测
        """
        if self.fusion_type == 'gate':
            stacked = torch.cat([lstm_pred, nfm_pred], dim=-1)
            gates = self.gate(stacked)
            fused = gates[:, :, 0:1] * lstm_pred + gates[:, :, 1:2] * nfm_pred
        elif self.fusion_type == 'add':
            w = F.softmax(self.weights, dim=0)
            fused = w[0] * lstm_pred + w[1] * nfm_pred
        elif self.fusion_type == 'concat':
            stacked = torch.cat([lstm_pred, nfm_pred], dim=-1)
            fused = self.proj(stacked)
        
        return fused


# 为了向后兼容，提供别名
NFMPredictor = SimplifiedNFMPredictor
LightweightFusion = SimplifiedLightweightFusion

