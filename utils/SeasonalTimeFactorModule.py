import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from data_provider.timefeatures import time_features_from_frequency_str, TimeFeature


class SeasonalTimeFactorModule(nn.Module):
    """
    季节性时间因子模块
    基于论文中的季节性时间因子理论，构建多层次的季节性特征提取和融合模块
    
    主要功能：
    1. 多层次季节性特征提取（日、周、月、年）
    2. 自适应季节性权重学习
    3. 季节性模式识别和增强
    4. 与时间序列数据的深度融合
    """
    
    def __init__(self, 
                 seq_len: int,
                 pred_len: int,
                 freq: str = '60min',
                 seasonal_levels: List[str] = ['daily', 'weekly', 'monthly', 'yearly'],
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.freq = freq
        self.seasonal_levels = seasonal_levels
        self.hidden_dim = hidden_dim
        
        # 基础时间特征提取器
        self.time_feature_extractors = self._build_time_feature_extractors()
        
        # 季节性特征编码器
        self.seasonal_encoders = nn.ModuleDict()
        for level in seasonal_levels:
            self.seasonal_encoders[level] = SeasonalEncoder(
                input_dim=self._get_feature_dim(level),
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        # 季节性注意力融合
        self.seasonal_attention = SeasonalAttentionFusion(
            num_levels=len(seasonal_levels),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 时间序列融合层
        self.temporal_fusion = TemporalFusionLayer(
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def _build_time_feature_extractors(self) -> Dict[str, List[TimeFeature]]:
        """构建不同季节性级别的时间特征提取器"""
        extractors = {}
        
        # 日季节性特征
        extractors['daily'] = [
            HourOfDay(),
            MinuteOfHour(),
            SecondOfMinute()
        ]
        
        # 周季节性特征
        extractors['weekly'] = [
            DayOfWeek(),
            HourOfDay(),
            MinuteOfHour()
        ]
        
        # 月季节性特征
        extractors['monthly'] = [
            DayOfMonth(),
            DayOfWeek(),
            HourOfDay()
        ]
        
        # 年季节性特征
        extractors['yearly'] = [
            MonthOfYear(),
            DayOfYear(),
            WeekOfYear(),
            DayOfMonth()
        ]
        
        return extractors
    
    def _get_feature_dim(self, level: str) -> int:
        """获取不同季节性级别的特征维度"""
        return len(self.time_feature_extractors[level])
    
    def extract_time_features(self, timestamps: pd.DatetimeIndex) -> Dict[str, torch.Tensor]:
        """提取多层次时间特征"""
        features = {}
        
        for level in self.seasonal_levels:
            level_features = []
            for extractor in self.time_feature_extractors[level]:
                feat_data = extractor(timestamps)
                level_features.append(feat_data)
            
            # 堆叠特征 [num_features, seq_len]
            level_tensor = torch.tensor(np.vstack(level_features), dtype=torch.float32)
            features[level] = level_tensor.T  # [seq_len, num_features]
        
        return features
    
    def forward(self, x: torch.Tensor, timestamps: pd.DatetimeIndex) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入时间序列 [batch_size, seq_len, features]
            timestamps: 时间戳序列
            
        Returns:
            增强后的时间序列特征 [batch_size, pred_len, 1]
        """
        batch_size = x.shape[0]
        
        # 1. 提取多层次时间特征
        time_features = self.extract_time_features(timestamps)
        
        # 2. 季节性特征编码
        seasonal_embeddings = {}
        for level, encoder in self.seasonal_encoders.items():
            level_feat = time_features[level].unsqueeze(0).repeat(batch_size, 1, 1)  # [B, seq_len, feat_dim]
            seasonal_embeddings[level] = encoder(level_feat)  # [B, seq_len, hidden_dim]
        
        # 3. 季节性注意力融合
        seasonal_fused = self.seasonal_attention(seasonal_embeddings)  # [B, seq_len, hidden_dim]
        
        # 4. 时间序列融合
        temporal_enhanced = self.temporal_fusion(x, seasonal_fused)  # [B, pred_len, hidden_dim]
        
        # 5. 输出投影
        output = self.output_proj(temporal_enhanced)  # [B, pred_len, 1]
        
        return output


class SeasonalEncoder(nn.Module):
    """季节性特征编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SeasonalAttentionFusion(nn.Module):
    """季节性注意力融合模块"""
    
    def __init__(self, num_levels: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        
        # 多尺度注意力
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 季节性权重学习
        self.seasonal_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, seasonal_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        融合不同季节性的特征
        
        Args:
            seasonal_embeddings: 不同季节性级别的特征字典
            
        Returns:
            融合后的特征 [B, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = list(seasonal_embeddings.values())[0].shape
        
        # 堆叠所有季节性特征
        stacked_features = torch.stack(list(seasonal_embeddings.values()), dim=1)  # [B, num_levels, seq_len, hidden_dim]
        
        # 应用季节性权重
        weights = F.softmax(self.seasonal_weights, dim=0)  # [num_levels]
        weighted_features = stacked_features * weights.view(1, -1, 1, 1)  # [B, num_levels, seq_len, hidden_dim]
        
        # 重塑为注意力输入格式
        weighted_features = weighted_features.view(batch_size * self.num_levels, seq_len, hidden_dim)
        
        # 自注意力增强
        attended_features, _ = self.multi_head_attention(
            weighted_features, weighted_features, weighted_features
        )
        
        # 重塑回原始格式
        attended_features = attended_features.view(batch_size, self.num_levels, seq_len, hidden_dim)
        
        # 融合不同季节性特征
        fused_features = attended_features.view(batch_size, seq_len, -1)  # [B, seq_len, num_levels * hidden_dim]
        fused_features = self.fusion_layer(fused_features)  # [B, seq_len, hidden_dim]
        
        return fused_features


class TemporalFusionLayer(nn.Module):
    """时间序列融合层"""
    
    def __init__(self, seq_len: int, pred_len: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 时间序列编码器
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 季节性特征编码器
        self.seasonal_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 融合注意力
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 预测投影层
        self.prediction_proj = nn.Linear(seq_len, pred_len)
        
    def forward(self, x: torch.Tensor, seasonal_features: torch.Tensor) -> torch.Tensor:
        """
        融合时间序列数据和季节性特征
        
        Args:
            x: 输入时间序列 [B, seq_len, features]
            seasonal_features: 季节性特征 [B, seq_len, hidden_dim]
            
        Returns:
            预测特征 [B, pred_len, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # 编码时间序列数据
        temporal_encoded = self.temporal_encoder(x)  # [B, seq_len, hidden_dim]
        
        # 编码季节性特征
        seasonal_encoded = self.seasonal_encoder(seasonal_features)  # [B, seq_len, hidden_dim]
        
        # 融合注意力
        fused_features, _ = self.fusion_attention(
            temporal_encoded, seasonal_encoded, seasonal_encoded
        )
        
        # 投影到预测长度
        # 使用全局平均池化 + 投影
        global_feature = fused_features.mean(dim=1)  # [B, hidden_dim]
        global_feature = global_feature.unsqueeze(1).repeat(1, self.pred_len, 1)  # [B, pred_len, hidden_dim]
        
        return global_feature


class AdaptiveSeasonalWeighting(nn.Module):
    """自适应季节性权重学习模块"""
    
    def __init__(self, num_seasons: int, hidden_dim: int = 32):
        super().__init__()
        
        self.num_seasons = num_seasons
        
        # 权重学习网络
        self.weight_net = nn.Sequential(
            nn.Linear(num_seasons, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_seasons),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, seasonal_features: torch.Tensor) -> torch.Tensor:
        """
        学习自适应季节性权重
        
        Args:
            seasonal_features: 季节性特征 [B, seq_len, num_seasons]
            
        Returns:
            加权后的季节性特征 [B, seq_len, num_seasons]
        """
        # 计算全局季节性统计
        global_stats = seasonal_features.mean(dim=1)  # [B, num_seasons]
        
        # 学习权重
        weights = self.weight_net(global_stats)  # [B, num_seasons]
        
        # 应用权重
        weighted_features = seasonal_features * weights.unsqueeze(1)  # [B, seq_len, num_seasons]
        
        return weighted_features


# 扩展的时间特征类
class HourOfDay(TimeFeature):
    """小时特征，编码为[-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class MinuteOfHour(TimeFeature):
    """分钟特征，编码为[-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class SecondOfMinute(TimeFeature):
    """秒特征，编码为[-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class DayOfWeek(TimeFeature):
    """星期几特征，编码为[-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """月份中的天数特征，编码为[-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """年份中的天数特征，编码为[-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """月份特征，编码为[-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """年份中的周数特征，编码为[-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


# 使用示例和测试函数
def create_seasonal_time_factor_module(seq_len: int = 96, pred_len: int = 8) -> SeasonalTimeFactorModule:
    """创建季节性时间因子模块的便捷函数"""
    return SeasonalTimeFactorModule(
        seq_len=seq_len,
        pred_len=pred_len,
        freq='60min',
        seasonal_levels=['daily', 'weekly', 'monthly', 'yearly'],
        hidden_dim=64,
        num_heads=4,
        dropout=0.1
    )


def test_seasonal_time_factor_module():
    """测试季节性时间因子模块"""
    # 创建测试数据
    batch_size = 32
    seq_len = 96
    pred_len = 8
    
    # 创建时间戳
    timestamps = pd.date_range('2023-01-01', periods=seq_len, freq='60min')
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, 1)
    
    # 创建模块
    module = create_seasonal_time_factor_module(seq_len, pred_len)
    
    # 前向传播
    output = module(x, timestamps)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模块参数量: {sum(p.numel() for p in module.parameters())}")
    
    return output


if __name__ == "__main__":
    # 运行测试
    test_seasonal_time_factor_module()



