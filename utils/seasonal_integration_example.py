"""
季节性时间因子模块使用示例
展示如何在iTransformerPlus中集成季节性时间因子模块
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils.SeasonalTimeFactorModule import SeasonalTimeFactorModule, create_seasonal_time_factor_module


class iTransformerPlusWithSeasonalFactors(nn.Module):
    """
    集成季节性时间因子模块的iTransformerPlus模型
    """
    
    def __init__(self, configs):
        super().__init__()
        self.is_plus_model = True
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.configs = configs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 原始iTransformerPlus组件
        from models.iTransformer import DataEmbedding_inverted
        from layers.Transformer_EncDec import Encoder, EncoderLayer
        from layers.SelfAttention_Family import FullAttention, AttentionLayer
        from utils.SEGateExogenous import SEGateExogenous
        from utils.DetailEnhancementModule import EnhancedDLinear1DHead
        
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=nn.LayerNorm(configs.d_model))
        
        if self.task_name in ['ultra_short_term_forecast', 'short_term_forecast', 'long_term_forecast']:
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
        # SE门控外生特征
        self.se_gate_exogenous = SEGateExogenous(reduction=4, device=self.device, channels=self.configs.n_input_features-1)
        
        # 增强版DLinear头部
        detail_kernel_sizes = getattr(configs, 'detail_kernel_sizes', [3, 7, 15])
        ksize = int(getattr(configs, 'moving_avg', 25))
        self.dlinear_head = EnhancedDLinear1DHead(
            self.seq_len, self.pred_len, 
            kernel_size=ksize, 
            detail_kernel_sizes=detail_kernel_sizes
        )
        
        # 新增：季节性时间因子模块
        self.seasonal_factor_module = create_seasonal_time_factor_module(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len
        )
        
        # 季节性特征融合层
        self.seasonal_fusion = nn.Sequential(
            nn.Linear(2, 1),  # 融合原始预测和季节性预测
            nn.Sigmoid()
        )
        
    def _split_target_exo(self, x_enc):
        """分离目标特征和外生特征"""
        _, _, N = x_enc.shape
        idx = int(getattr(self.configs, 'target_channel_index', 0))
        idx = max(0, min(idx, N - 1))
        x_target = x_enc[:, :, idx:idx + 1]
        if N > 1:
            left, right = x_enc[:, :, :idx], x_enc[:, :, idx + 1:]
            x_feats = torch.cat([left, right], dim=2) if (left.shape[2] + right.shape[2]) > 0 else None
        else:
            x_feats = None
        return x_target, x_feats
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """预测函数，集成季节性时间因子"""
        # RevIN标准化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        # 分离目标/外生特征
        x_target, x_feats = self._split_target_exo(x_enc)
        
        # 原始iTransformerPlus预测
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        _, _, N = x_enc.shape
        dec_out = dec_out[:, :, :N]
        
        # DLinear头部预测
        dlin = self.dlinear_head(x_target)
        dec_out[:, :, 0:1] = dec_out[:, :, 0:1] + dlin
        
        # 新增：季节性时间因子预测
        # 从时间标记中提取时间戳
        timestamps = self._extract_timestamps_from_mark(x_mark_enc)
        seasonal_pred = self.seasonal_factor_module(x_target, timestamps)
        
        # 融合原始预测和季节性预测
        # 使用自适应权重融合
        fusion_input = torch.cat([dec_out[:, :, 0:1], seasonal_pred], dim=-1)  # [B, pred_len, 2]
        fusion_weight = self.seasonal_fusion(fusion_input)  # [B, pred_len, 1]
        
        # 加权融合
        final_pred = fusion_weight * dec_out[:, :, 0:1] + (1 - fusion_weight) * seasonal_pred
        
        # 更新最终输出
        dec_out[:, :, 0:1] = final_pred
        
        # 反RevIN标准化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out
    
    def _extract_timestamps_from_mark(self, x_mark_enc):
        """从时间标记中提取时间戳"""
        # 这里需要根据实际的时间标记格式来实现
        # 假设x_mark_enc包含时间特征，我们需要重构时间戳
        batch_size, seq_len, _ = x_mark_enc.shape
        
        # 创建一个示例时间戳序列
        # 在实际应用中，这里应该根据x_mark_enc的内容来重构真实的时间戳
        base_time = pd.Timestamp('2023-01-01')
        timestamps = pd.date_range(base_time, periods=seq_len, freq='60min')
        
        return timestamps
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """前向传播"""
        if self.task_name in ['ultra_short_term_forecast', 'short_term_forecast', 'long_term_forecast']:
            if x_enc.shape[2] > 1:
                x_enc = x_enc[:, :, 2:]
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, :, 0:1]
        return None


def test_integration():
    """测试集成后的模型"""
    # 创建测试配置
    class Config:
        def __init__(self):
            self.task_name = 'ultra_short_term_forecast'
            self.seq_len = 96
            self.pred_len = 8
            self.d_model = 128
            self.dropout = 0.1
            self.e_layers = 1
            self.n_heads = 8
            self.d_ff = 128
            self.activation = 'gelu'
            self.n_input_features = 13
            self.moving_avg = 25
            self.detail_kernel_sizes = [3, 7, 15]
    
    config = Config()
    
    # 创建模型
    model = iTransformerPlusWithSeasonalFactors(config)
    
    # 创建测试数据
    batch_size = 32
    x_enc = torch.randn(batch_size, config.seq_len, 1)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # 时间特征
    x_dec = torch.randn(batch_size, config.pred_len, 1)
    x_mark_dec = torch.randn(batch_size, config.pred_len, 4)
    
    # 前向传播
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"输入形状: {x_enc.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    return output


def compare_performance():
    """比较原始模型和集成季节性因子的模型性能"""
    print("=== 性能对比测试 ===")
    
    # 测试原始季节性时间因子模块
    print("\n1. 测试季节性时间因子模块:")
    from utils.SeasonalTimeFactorModule import test_seasonal_time_factor_module
    test_seasonal_time_factor_module()
    
    # 测试集成模型
    print("\n2. 测试集成模型:")
    test_integration()
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    # 运行测试
    compare_performance()



