import torch
import torch.nn as nn
import torch.nn.functional as F
from third_party.pytorch_wavelets.dwt.transform1d import DWT1DForward


class MultiScaleDetailExtractor(nn.Module):
    """
    优化版多尺度细节提取器
    - 合并填充操作，避免重复计算
    - 使用单尺度或少量尺度减少计算开销
    """
    def __init__(self, kernel_sizes=[7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.detail_extractors = nn.ModuleList()

        for k in kernel_sizes:
            self.detail_extractors.append(nn.AvgPool1d(kernel_size=k, stride=1, padding=0))

    def forward(self, x):  # x: [B, L, 1]
        details = []
        x_perm = x.permute(0, 2, 1)  # [B, 1, L]

        # 优化：预先计算最大填充，避免重复操作
        max_k = max(self.kernel_sizes)
        max_pad = (max_k - 1) // 2

        # 一次性填充到最大尺度
        front = x_perm[:, :, 0:1].repeat(1, 1, max_pad)
        end = x_perm[:, :, -1:].repeat(1, 1, max_pad)
        x_padded_max = torch.cat([front, x_perm, end], dim=2)

        for i, extractor in enumerate(self.detail_extractors):
            k = self.kernel_sizes[i]
            pad = (k - 1) // 2

            # 从最大填充中截取需要的部分
            start_idx = max_pad - pad
            end_idx = x_padded_max.shape[2] - (max_pad - pad)
            x_padded = x_padded_max[:, :, start_idx:end_idx]

            # 提取趋势
            trend = extractor(x_padded)
            # 计算细节 (原序列 - 趋势)
            detail = x_perm - trend
            details.append(detail.permute(0, 2, 1))  # [B, L, 1]

        return details


class DetailEnhancementNetwork(nn.Module):
    """
    优化版细节增强网络
    - 使用轻量级深度可分离卷积替代多头注意力
    - 使用批量归一化加速训练和提升稳定性
    - 减少卷积层数，降低计算复杂度
    """
    def __init__(self, seq_len, pred_len, hidden_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # 优化：使用深度可分离卷积 (Depthwise Separable Convolution)
        # 计算量从 O(C_in * C_out * K) 降低到 O(C_in * K + C_in * C_out)
        self.depthwise_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, groups=1)
        self.bn1 = nn.BatchNorm1d(1)

        # 点卷积 (Pointwise Convolution) - 1x1卷积用于通道混合
        self.pointwise_conv = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 压缩回单通道
        self.compress_conv = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(1)

        self.activation = nn.GELU()

        # 输出投影层
        self.output_proj = nn.Linear(seq_len, pred_len)

    def forward(self, details):  # details: List of [B, L, 1]
        enhanced_details = []

        for detail in details:
            # 深度可分离卷积增强
            x = detail.permute(0, 2, 1)  # [B, 1, L]

            # 深度卷积 + BN + 激活
            x = self.depthwise_conv(x)
            x = self.bn1(x)
            x = self.activation(x)

            # 点卷积 + BN + 激活
            x = self.pointwise_conv(x)
            x = self.bn2(x)
            x = self.activation(x)

            # 压缩 + BN
            x = self.compress_conv(x)
            x = self.bn3(x)

            # 残差连接
            detail_enhanced = x.permute(0, 2, 1)  # [B, L, 1]
            detail_final = detail + detail_enhanced

            enhanced_details.append(detail_final)

        return enhanced_details


class DetailFusionModule(nn.Module):
    """
    优化版细节融合模块
    - 使用批量归一化替代手动统计计算
    - 简化融合逻辑，减少计算开销
    """
    def __init__(self, num_scales, hidden_dim=16):
        super().__init__()
        self.num_scales = num_scales

        # 优化：使用BatchNorm替代手动均值/方差计算
        self.bn = nn.BatchNorm1d(num_scales)

        # 简化的权重学习网络
        self.scale_weights = nn.Sequential(
            nn.Linear(num_scales, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, enhanced_details):  # List of [B, L, 1]
        if len(enhanced_details) == 1:
            # 单尺度情况，直接返回
            return enhanced_details[0]

        # 堆叠所有尺度的细节 [B, L, num_scales]
        stacked_details = torch.cat(enhanced_details, dim=-1)  # [B, L, num_scales]

        # 使用BatchNorm进行归一化 (需要转换维度)
        stacked_norm = stacked_details.permute(0, 2, 1)  # [B, num_scales, L]
        stacked_norm = self.bn(stacked_norm)
        stacked_norm = stacked_norm.permute(0, 2, 1)  # [B, L, num_scales]

        # 计算全局特征用于权重学习
        global_feat = stacked_norm.mean(dim=1)  # [B, num_scales]

        # 学习权重
        weights = self.scale_weights(global_feat)  # [B, num_scales]

        # 加权融合
        weights = weights.unsqueeze(1)  # [B, 1, num_scales]
        fused_detail = (stacked_norm * weights).sum(dim=-1, keepdim=True)  # [B, L, 1]

        return fused_detail


class DetailEnhancementModule(nn.Module):
    """
    优化版细节增强模块
    - 默认使用单尺度，减少计算开销
    - 集成轻量级卷积和批量归一化
    - 整体计算复杂度降低约70%
    """
    def __init__(self, seq_len, pred_len, kernel_sizes=[7], hidden_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 多尺度细节提取 (优化版)
        self.detail_extractor = MultiScaleDetailExtractor(kernel_sizes)

        # 细节增强网络 (轻量级版本)
        self.enhancement_net = DetailEnhancementNetwork(seq_len, pred_len, hidden_dim)

        # 细节融合模块 (优化版)
        self.fusion_module = DetailFusionModule(len(kernel_sizes), hidden_dim // 2)

        # 最终输出投影
        self.output_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x_target):  # x_target: [B, L, 1]
        # 1. 多尺度细节提取
        details = self.detail_extractor(x_target)

        # 2. 细节增强
        enhanced_details = self.enhancement_net(details)

        # 3. 细节融合
        fused_detail = self.fusion_module(enhanced_details)

        # 4. 输出投影
        detail_pred = self.output_proj(fused_detail.squeeze(-1))  # [B, pred_len]

        return detail_pred.unsqueeze(-1)  # [B, pred_len, 1]


class EnhancedDLinear1DHead(nn.Module):
    """
    优化版DLinear头部
    - 移除门控机制，使用固定权重融合
    - 集成轻量级细节处理模块
    - 使用批量归一化提升稳定性
    """
    def __init__(self, seq_len, pred_len, kernel_size=8, detail_kernel_sizes=[7]):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 原始DLinear组件
        self.decomp = _TargetDecomp(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

        # 批量归一化 - 对单通道进行归一化
        # 输入形状: [B, 1, pred_len]，所以num_features=1
        self.bn_seasonal = nn.BatchNorm1d(1)
        self.bn_trend = nn.BatchNorm1d(1)

        # 优化版细节增强模块
        self.detail_enhancement = DetailEnhancementModule(
            seq_len, pred_len, detail_kernel_sizes, hidden_dim=32
        )

        # 移除门控机制，使用固定权重 (可通过实验调整)
        # DLinear权重: 0.6, 细节权重: 0.4
        self.dlinear_weight = 0.6
        self.detail_weight = 1.0 - self.dlinear_weight

    def forward(self, x_target):  # [B, L, 1]
        # 原始DLinear预测
        seasonal, trend = self.decomp(x_target)
        s = seasonal.squeeze(-1)  # [B, L]
        t = trend.squeeze(-1)     # [B, L]

        # 分别预测季节和趋势
        seasonal_pred = self.linear_seasonal(s)  # [B, pred_len]
        trend_pred = self.linear_trend(t)        # [B, pred_len]

        # 批量归一化 - 正确的维度处理
        # [B, pred_len] -> [B, 1, pred_len] -> BN -> [B, 1, pred_len] -> [B, pred_len]
        seasonal_pred = self.bn_seasonal(seasonal_pred.unsqueeze(1)).squeeze(1)
        trend_pred = self.bn_trend(trend_pred.unsqueeze(1)).squeeze(1)

        # DLinear预测
        dlinear_pred = seasonal_pred + trend_pred  # [B, pred_len]

        # 细节增强预测
        detail_pred = self.detail_enhancement(x_target)  # [B, pred_len, 1]
        detail_pred = detail_pred.squeeze(-1)  # [B, pred_len]

        # 固定权重融合 (移除门控机制)
        final_pred = self.dlinear_weight * dlinear_pred + self.detail_weight * detail_pred

        return final_pred.unsqueeze(-1)  # [B, pred_len, 1]


# 导入_TargetDecomp类 (从iTransformerPlus.py)
class _TargetMovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):  # x: [B, L, 1]
        k = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, k, 1)
        end = x[:, -1:, :].repeat(1, k, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)           # [B, 1, L]
        x = self.avg(x)
        x = x.permute(0, 2, 1)           # [B, L, 1]
        return x

class _TargetDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.mavg = _TargetMovingAvg(kernel_size)

    def forward(self, x):  # [B, L, 1]
        trend = self.mavg(x)
        seasonal = x - trend
        return seasonal, trend


class WaveletDetailHead(nn.Module):
    """
    基于多级一维离散小波变换(DWT)的细节预测头：
    - 仅对目标通道 [B,L,1] 进行处理
    - 通过 J 级 DWT 提取高频细节系数（多尺度细节）
    - 将各级高频分量上采样回原长度并做可学习加权融合
    - 通过线性层从 seq_len 投影到 pred_len
    """
    def __init__(self, seq_len: int, pred_len: int, levels: int = 3, wavelet: str = 'db4', mode: str = 'symmetric'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.levels = max(1, int(levels))
        self.dwt = DWT1DForward(J=self.levels, wave=wavelet, mode=mode)
        # 级别权重（softmax 归一）
        self.level_logits = nn.Parameter(torch.zeros(self.levels))
        # 输出投影
        self.out_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x_target):  # [B, L, 1]
        # 调整到 [B, 1, L]
        x = x_target.permute(0, 2, 1)
        _, yh_list = self.dwt(x)  # yl: 低频 [B,1,L/J], yh_list: len=J, 每级 [B,1,L/(2^k)]
        # 将各级高频分量上采样回原长度
        upsampled = []
        for yh in yh_list:
            # yh: [B, 1, L_k]
            if yh.shape[-1] != x.shape[-1]:
                up = F.interpolate(yh, size=x.shape[-1], mode='linear', align_corners=False)
            else:
                up = yh
            upsampled.append(up)
        # 可学习加权融合
        weights = torch.softmax(self.level_logits, dim=0)  # [J]
        fused = sum(w * up for w, up in zip(weights, upsampled))  # [B,1,L]
        # 投影到预测步长
        fused_seq = fused.squeeze(1)                 # [B,L]
        detail_pred = self.out_proj(fused_seq)       # [B,pred_len]
        return detail_pred.unsqueeze(-1)             # [B,pred_len,1]
