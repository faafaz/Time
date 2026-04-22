import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. 基础组件 (保持 DLinear 原有的分解逻辑)
# =============================================================================

class moving_avg(nn.Module):
    """
    移动平均模块：用于提取趋势
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding logic to keep sequence length unchanged
        padding_length = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding_length, 1)
        end = x[:, -1:, :].repeat(1, padding_length, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    序列分解模块：Input -> Trend + Seasonal
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # Trend
        res = x - moving_mean             # Seasonal
        return res, moving_mean

# =============================================================================
# 2. 核心创新：增强型 3D Mixer 模块 (Patch + Channel Mixing)
# =============================================================================

class MLP(nn.Module):
    """
    非线性变换单元：Norm -> FC -> GELU -> Dropout -> FC -> Dropout
    引入 expansion_factor 提升模型拟合复杂波动的能力
    """
    def __init__(self, dim, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion_factor)
        self.fc2 = nn.Linear(dim * expansion_factor, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [..., dim]
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x  # Pre-Norm Residual Connection

class MixerBlock(nn.Module):
    """
    单层混合块：依次处理 Z轴(变量)、X轴(局部)、Y轴(全局)
    """
    def __init__(self, n_vars, num_patches, patch_len, dropout=0.1):
        super().__init__()
        
        # 1. Z-Mixer: 变量间关系 (Channel)
        # 维度不宜过大，Expansion=2 即可
        self.z_mlp = MLP(dim=n_vars, expansion_factor=2, dropout=dropout)
        
        # 2. X-Mixer: 子序列内部 (Intra-patch, Temporal Local)
        # 时间局部性很重要，可以适当增加复杂度
        self.x_mlp = MLP(dim=patch_len, expansion_factor=2, dropout=dropout)
        
        # 3. Y-Mixer: 子序列之间 (Inter-patch, Temporal Global)
        # 捕捉长程依赖
        self.y_mlp = MLP(dim=num_patches, expansion_factor=2, dropout=dropout)

    def forward(self, x):
        # Input: [Batch, Channels, Num_Patches, Patch_Len] -> [B, C, P, S]
        
        # --- Z轴混合 (Channel Mixing) ---
        # 目标维度: C (dim=1)
        # Permute: [B, P, S, C] -> MLP acts on C
        z = x.permute(0, 2, 3, 1)
        z = self.z_mlp(z)
        x = z.permute(0, 3, 1, 2) # 还原回 [B, C, P, S]
        
        # --- X轴混合 (Intra-patch Mixing) ---
        # 目标维度: S (dim=3)
        # Tensor 默认最后维度，直接传入
        x = self.x_mlp(x)
        
        # --- Y轴混合 (Inter-patch Mixing) ---
        # 目标维度: P (dim=2)
        # Permute: [B, C, S, P] -> MLP acts on P
        y = x.permute(0, 1, 3, 2)
        y = self.y_mlp(y)
        x = y.permute(0, 1, 3, 2) # 还原回 [B, C, P, S]
        
        return x

class ThreeDimMixer(nn.Module):
    """
    完整的多层 3D 混合网络
    """
    def __init__(self, n_vars, seq_len, pred_len, patch_len=24, e_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        
        # 检查能否整除
        assert seq_len % patch_len == 0, f"序列长度 {seq_len} 必须能被 Patch长度 {patch_len} 整除"

        # 堆叠多层 MixerBlock，增加网络深度
        self.blocks = nn.ModuleList([
            MixerBlock(n_vars, self.num_patches, patch_len, dropout)
            for _ in range(e_layers)
        ])

        # 最终投影层：将 seq_len 映射到 pred_len
        self.projection = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x Input: [Batch, Channels, Seq_Len]
        B, C, L = x.shape
        
        # 1. Tensorization (重塑为 4D 张量)
        # [B, C, 96] -> [B, C, 4, 24]
        x = x.reshape(B, C, self.num_patches, self.patch_len)
        
        # 2. Deep Mixing (多层特征交互)
        for block in self.blocks:
            x = block(x)
            
        # 3. Flatten (展平)
        # [B, C, 4, 24] -> [B, C, 96]
        x = x.reshape(B, C, L)
        
        # 4. Projection (预测未来)
        # [B, C, 96] -> [B, C, 16]
        x = self.projection(x)
        
        return x

# =============================================================================
# 3. 主模型 Model
# =============================================================================

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # 能够容错处理 configs 中可能缺少的参数
        self.moving_avg_size = getattr(configs, 'moving_avg', 25)
        self.patch_len = getattr(configs, 'patch_len', 24) # 默认子序列长度
        self.e_layers = getattr(configs, 'e_layers', 2)    # 默认堆叠2层
        # self.e_layers = 1    # 默认堆叠2层
        self.dropout = getattr(configs, 'dropout', 0.3)
        self.individual = configs.individual
        self.channels = configs.enc_in
        
        # 分解模块
        self.decompsition = series_decomp(self.moving_avg_size)

        # -----------------------------------------------------------
        # Seasonal 分支：使用增强版 3D Mixer
        # -----------------------------------------------------------
        self.Mixer_Seasonal = ThreeDimMixer(
            n_vars=self.channels,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            patch_len=self.patch_len,
            e_layers=self.e_layers,
            dropout=self.dropout
        )

        self.Mixer_Trend = ThreeDimMixer(
            n_vars=self.channels,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            patch_len=self.patch_len,
            e_layers=self.e_layers,
            dropout=self.dropout
        )

        # -----------------------------------------------------------
        # Trend 分支：保持 DLinear 简单的线性层
        # -----------------------------------------------------------
        if self.individual:
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x Input: [Batch, Seq_Len, Channels]
        
        # === 1. RevIN Normalization (统计归一化) ===
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # === 2. Decomposition (序列分解) ===
        # seasonal_init: [B, L, C]
        # trend_init:    [B, L, C]
        seasonal_init, trend_init = self.decompsition(x)

        # 变换维度为 [B, C, L] 以适配 Linear 和 Mixer
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # === 3. Seasonal Prediction (复杂分支) ===
        # 通过 3D Mixer 处理复杂的季节项/残差项
        seasonal_output = self.Mixer_Seasonal(seasonal_init)

        # === 4. Trend Prediction (简单分支) ===
        # if self.individual:
        #     trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
        #                              dtype=trend_init.dtype).to(trend_init.device)
        #     for i in range(self.channels):
        #         trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        # else:
        #     trend_output = self.Linear_Trend(trend_init)
        trend_output = self.Mixer_Trend(trend_init)

        # === 5. Add & Output ===
        x = seasonal_output + trend_output
        
        # 变回 [B, Pred_Len, Channels]
        x = x.permute(0, 2, 1)

        # === 6. RevIN Inverse (反归一化) ===
        x = x * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        x = x + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        # 只返回目标预测变量（通常最后一列是 Power）
        return x[:, :, self.channels - 1:self.channels]

# =============================================================================
# 4. 调试/测试代码块
# =============================================================================
# if __name__ == '__main__':
#     class Configs:
#         seq_len = 96
#         pred_len = 16
#         enc_in = 7       # 假设有7个变量 (风速、风向、温度等)
#         individual = False
#         moving_avg = 25
#         patch_len = 24   # 切片长度
#         e_layers = 2     # Mixer层数
#         dropout = 0.1

#     configs = Configs()
#     model = Model(configs)

#     # 模拟输入数据 [Batch=32, Seq_Len=96, Channels=7]
#     input_tensor = torch.randn(32, 96, 7)
    
#     # 模拟其他参数 (DLinear 接口通常还需要这些)
#     x_mark_enc = torch.randn(32, 96, 4)
#     x_dec = torch.randn(32, 16, 7)
#     x_mark_dec = torch.randn(32, 16, 4)

#     output = model(input_tensor, x_mark_enc, x_dec, x_mark_dec)
    
#     print("模型构建成功！")
#     print(f"输入尺寸: {input_tensor.shape}")
#     print(f"输出尺寸: {output.shape}") 
#     # 预期输出: [32, 16, 1] (如果最后一列是目标)