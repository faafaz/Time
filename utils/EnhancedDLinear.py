import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedDecompositionModule(nn.Module):
    """
    统一的分解模块
    深度融合细节处理与季节/趋势分解，避免分离处理
    """
    def __init__(self, kernel_sizes=[3, 7, 15], dropout=0.1):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout

        # 多尺度移动平均层
        self.moving_avg_layers = nn.ModuleList()
        for k in kernel_sizes:
            self.moving_avg_layers.append(nn.AvgPool1d(kernel_size=k, stride=1, padding=0))

        # 统一的特征提取器 - 使用轻量级卷积
        self.unified_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(8, 1, kernel_size=1)
        )

        # 自适应权重学习 - 简化版本
        self.adaptive_weights = nn.Sequential(
            nn.Linear(len(kernel_sizes), 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, len(kernel_sizes)),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):  # x: [B, L, 1]
        x_perm = x.permute(0, 2, 1)  # [B, 1, L]

        # 多尺度分解
        trends = []
        seasonals = []
        details = []

        for i, mavg in enumerate(self.moving_avg_layers):
            k = (self.kernel_sizes[i] - 1) // 2

            # 边界填充
            front = x_perm[:, :, 0:1].repeat(1, 1, k)
            end = x_perm[:, :, -1:].repeat(1, 1, k)
            x_padded = torch.cat([front, x_perm, end], dim=2)

            # 提取趋势
            trend = mavg(x_padded)
            trends.append(trend)

            # 计算季节性和细节
            seasonal = x_perm - trend
            seasonals.append(seasonal)

            # 细节 = 季节性 - 平滑后的季节性
            detail = seasonal - F.avg_pool1d(seasonal, kernel_size=3, stride=1, padding=1)
            details.append(detail)

        # 统一特征提取
        unified_features = []
        for detail in details:
            feat = self.unified_extractor(detail)
            unified_features.append(feat)

        # 计算自适应权重
        # 使用每个尺度的方差作为特征
        scale_features = torch.stack([feat.var(dim=2) for feat in unified_features], dim=-1)  # [B, 1, num_scales]
        weights = self.adaptive_weights(scale_features.squeeze(1))  # [B, num_scales]

        # 加权融合
        fused_trend = torch.zeros_like(trends[0])
        fused_seasonal = torch.zeros_like(seasonals[0])
        fused_detail = torch.zeros_like(details[0])

        for i, (trend, seasonal, detail) in enumerate(zip(trends, seasonals, details)):
            weight = weights[:, i:i+1, None]  # [B, 1, 1]
            fused_trend += weight * trend
            fused_seasonal += weight * seasonal
            fused_detail += weight * detail

        return fused_trend.permute(0, 2, 1), fused_seasonal.permute(0, 2, 1), fused_detail.permute(0, 2, 1)


class LightweightEnhancementModule(nn.Module):
    """
    轻量级增强模块
    使用简化的注意力机制和残差连接
    """
    def __init__(self, seq_len, pred_len, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 轻量级特征增强
        self.feature_enhancer = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 简化的自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=1,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len // 2, pred_len)
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(1)

    def forward(self, x):  # x: [B, L, 1]
        # 特征增强
        enhanced = self.feature_enhancer(x)

        # 残差连接
        x_enhanced = x + enhanced

        # 自注意力
        attended, _ = self.self_attention(x_enhanced, x_enhanced, x_enhanced)

        # 残差连接 + 层归一化
        x_final = self.layer_norm(x_enhanced + attended)

        # 输出投影
        output = self.output_proj(x_final.squeeze(-1))  # [B, pred_len]

        return output.unsqueeze(-1)  # [B, pred_len, 1]


class OptimizedEnhancedDLinear1DHead(nn.Module):
    """
    优化版增强DLinear头部
    深度融合细节处理与季节/趋势分解，添加dropout防止过拟合
    """
    def __init__(self, seq_len, pred_len, kernel_size=8, detail_kernel_sizes=[3, 7, 15], dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout = dropout

        # 统一分解模块
        self.unified_decomp = UnifiedDecompositionModule(detail_kernel_sizes, dropout)

        # 传统DLinear组件（用于主要趋势和季节性）
        self.main_decomp = self._create_moving_avg(kernel_size)
        self.linear_seasonal = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len // 2, pred_len)
        )
        self.linear_trend = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len // 2, pred_len)
        )

        # 细节增强模块
        self.detail_enhancement = LightweightEnhancementModule(seq_len, pred_len, dropout=dropout)

        # 融合网络 - 学习如何组合不同组件
        self.fusion_net = nn.Sequential(
            nn.Linear(3, 8),  # 3个组件：trend, seasonal, detail
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 3),
            nn.Softmax(dim=-1)
        )

        # 最终输出层
        self.final_proj = nn.Sequential(
            nn.Linear(pred_len, pred_len // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len // 2, pred_len)
        )

    def _create_moving_avg(self, kernel_size):
        """创建移动平均层"""
        class MovingAvg(nn.Module):
            def __init__(self, kernel_size):
                super().__init__()
                self.kernel_size = kernel_size
                self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

            def forward(self, x):  # x: [B, L, 1]
                k = (self.kernel_size - 1) // 2
                front = x[:, 0:1, :].repeat(1, k, 1)
                end = x[:, -1:, :].repeat(1, k, 1)
                x = torch.cat([front, x, end], dim=1)
                x = x.permute(0, 2, 1)  # [B, 1, L]
                x = self.avg(x)
                x = x.permute(0, 2, 1)  # [B, L, 1]
                return x

        return MovingAvg(kernel_size)

    def forward(self, x_target):  # [B, L, 1]
        # 1. 统一分解（深度融合细节处理）
        unified_trend, unified_seasonal, unified_detail = self.unified_decomp(x_target)

        # 2. 传统DLinear分解（用于主要趋势和季节性）
        main_trend = self.main_decomp(x_target)
        main_seasonal = x_target - main_trend

        # 3. 线性投影
        trend_pred = self.linear_trend(main_trend.squeeze(-1))  # [B, pred_len]
        seasonal_pred = self.linear_seasonal(main_seasonal.squeeze(-1))  # [B, pred_len]

        # 4. 细节增强
        detail_pred = self.detail_enhancement(unified_detail)  # [B, pred_len, 1]
        detail_pred = detail_pred.squeeze(-1)  # [B, pred_len]

        # 5. 自适应融合
        # 计算每个组件的统计特征用于融合权重
        trend_feat = trend_pred.mean(dim=1, keepdim=True)  # [B, 1]
        seasonal_feat = seasonal_pred.mean(dim=1, keepdim=True)  # [B, 1]
        detail_feat = detail_pred.mean(dim=1, keepdim=True)  # [B, 1]

        fusion_features = torch.cat([trend_feat, seasonal_feat, detail_feat], dim=-1)  # [B, 3]
        fusion_weights = self.fusion_net(fusion_features)  # [B, 3]

        # 加权融合
        w_trend = fusion_weights[:, 0:1]  # [B, 1]
        w_seasonal = fusion_weights[:, 1:2]  # [B, 1]
        w_detail = fusion_weights[:, 2:3]  # [B, 1]

        fused_pred = w_trend * trend_pred + w_seasonal * seasonal_pred + w_detail * detail_pred

        # 6. 最终输出处理
        final_pred = self.final_proj(fused_pred)

        return final_pred.unsqueeze(-1)  # [B, pred_len, 1]


# 为了兼容性，保留原有的类名
class EnhancedDLinear1DHead(OptimizedEnhancedDLinear1DHead):
    """兼容性别名"""
    pass


class EnhancedDLinear(nn.Module):
    """
    增强版DLinear模型
    深度融合细节处理与季节趋势分解，适用于多变量时间序列预测
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.n_input_features
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 统一的分解模块
        detail_kernel_sizes = getattr(configs, 'detail_kernel_sizes', [3, 7, 15])
        self.unified_decomp = UnifiedDecompositionModule(detail_kernel_sizes, self.dropout)

        # 传统DLinear组件
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.main_decomp = self._create_moving_avg(kernel_size)

        # 多变量线性投影
        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.seq_len // 2, self.pred_len)
        )
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.seq_len // 2, self.pred_len)
        )

        # 细节增强模块
        self.detail_enhancement = LightweightEnhancementModule(
            self.seq_len, self.pred_len, dropout=self.dropout
        )

        # 多变量融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(3 * self.channels, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 3 * self.channels),
            nn.Softmax(dim=-1)
        )

        # 最终输出投影
        self.final_proj = nn.Sequential(
            nn.Linear(self.pred_len, self.pred_len // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.pred_len // 2, self.pred_len)
        )

    def _create_moving_avg(self, kernel_size):
        """创建移动平均层"""
        class MovingAvg(nn.Module):
            def __init__(self, kernel_size):
                super().__init__()
                self.kernel_size = kernel_size
                self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

            def forward(self, x):  # x: [B, L, C]
                k = (self.kernel_size - 1) // 2
                front = x[:, 0:1, :].repeat(1, k, 1)
                end = x[:, -1:, :].repeat(1, k, 1)
                x = torch.cat([front, x, end], dim=1)
                x = x.permute(0, 2, 1)  # [B, C, L]
                x = self.avg(x)
                x = x.permute(0, 2, 1)  # [B, L, C]
                return x

        return MovingAvg(kernel_size)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [B, L, C]
        batch_size, seq_len, channels = x.shape

        # 对每个通道分别处理
        trend_preds = []
        seasonal_preds = []
        detail_preds = []

        for c in range(channels):
            x_channel = x[:, :, c:c+1]  # [B, L, 1]

            # 1. 统一分解
            unified_trend, unified_seasonal, unified_detail = self.unified_decomp(x_channel)

            # 2. 传统DLinear分解
            main_trend = self.main_decomp(x_channel)
            main_seasonal = x_channel - main_trend

            # 3. 线性投影
            trend_pred = self.linear_trend(main_trend.squeeze(-1))  # [B, pred_len]
            seasonal_pred = self.linear_seasonal(main_seasonal.squeeze(-1))  # [B, pred_len]

            # 4. 细节增强
            detail_pred = self.detail_enhancement(unified_detail)  # [B, pred_len, 1]
            detail_pred = detail_pred.squeeze(-1)  # [B, pred_len]

            trend_preds.append(trend_pred.unsqueeze(-1))
            seasonal_preds.append(seasonal_pred.unsqueeze(-1))
            detail_preds.append(detail_pred.unsqueeze(-1))

        # 堆叠所有通道的预测
        trend_preds = torch.cat(trend_preds, dim=-1)  # [B, pred_len, C]
        seasonal_preds = torch.cat(seasonal_preds, dim=-1)  # [B, pred_len, C]
        detail_preds = torch.cat(detail_preds, dim=-1)  # [B, pred_len, C]

        # 多变量自适应融合
        # 计算每个通道每个组件的统计特征
        trend_feat = trend_preds.mean(dim=1)  # [B, C]
        seasonal_feat = seasonal_preds.mean(dim=1)  # [B, C]
        detail_feat = detail_preds.mean(dim=1)  # [B, C]

        fusion_features = torch.cat([trend_feat, seasonal_feat, detail_feat], dim=-1)  # [B, 3*C]
        fusion_weights = self.fusion_net(fusion_features)  # [B, 3*C]

        # 重塑权重
        weights = fusion_weights.view(batch_size, 3, channels)  # [B, 3, C]

        # 加权融合
        w_trend = weights[:, 0:1, :]  # [B, 1, C]
        w_seasonal = weights[:, 1:2, :]  # [B, 1, C]
        w_detail = weights[:, 2:3, :]  # [B, 1, C]

        fused_pred = w_trend * trend_preds + w_seasonal * seasonal_preds + w_detail * detail_preds

        # 最终输出处理
        final_pred = self.final_proj(fused_pred)

        return final_pred  # [B, pred_len, C]