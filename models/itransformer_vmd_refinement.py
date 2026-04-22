"""
iTransformer + VMD + Future Refinement Head。

特殊架构变体:
- 对目标变量和风速都进行VMD分解
- 使用FutureRefinementHead利用未来风速信息做残差修正
- 基础预测 + 残差修正两阶段
"""

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.iTransformer import DataEmbedding_inverted
from models.vmd_decomposer import StandardVMD


class FutureRefinementHead(nn.Module):
    """
    方案B：利用历史隐变量和未来预测风速进行残差修正。
    """
    def __init__(self, d_model, n_future_vars, dropout=0.1):
        super().__init__()
        self.future_encoder = nn.Linear(n_future_vars, d_model)
        self.refine_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, history_latent, x_dec):
        # history_latent: [B, d_model]
        # x_dec: [B, pred_len, C]
        f_feat = self.future_encoder(x_dec)  # [B, pred_len, d_model]
        h_feat = history_latent.unsqueeze(1).repeat(1, x_dec.size(1), 1)

        combined = torch.cat([h_feat, f_feat], dim=-1)
        return self.refine_mlp(combined)


class Model(nn.Module):
    """
    VMD分解 + iTransformer + 未来修正头。

    特点:
    - 对目标变量(功率)和条件变量(风速)都做VMD分解
    - 提取目标变量的深层特征做基础预测
    - 利用未来风速信息做残差修正
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in

        # 1. VMD分解模块
        vmd_k = getattr(configs, 'vmd_k', 8)
        vmd_impl = getattr(configs, 'vmd_impl', 'fftbank')
        vmd_alpha = getattr(configs, 'vmd_alpha', 2000.0)

        self.vmd_k = vmd_k
        self.vmd_decomposer = StandardVMD(
            K=vmd_k,
            impl=vmd_impl,
            alpha=vmd_alpha
        )

        # 2. iTransformer核心架构
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 3. 基础预测投影 (路径A)
        self.base_projection = nn.Linear(configs.d_model, configs.pred_len)

        # 4. 未来修正头 (路径B)
        self.refinement_head = FutureRefinementHead(
            configs.d_model,
            configs.enc_in,
            configs.dropout
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 归一化 (RevIN)
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # VMD分解功率信号和风速信号
        target_var = x[:, :, -1:]  # 目标功率 (最后一列)
        wind_var = x[:, :, -2:-1]  # 风速 (倒数第二列)
        imfs = self.vmd_decomposer(target_var)
        wind_imfs = self.vmd_decomposer(wind_var)

        # 特征拼接: 原始特征 + 多尺度IMFs
        x_concat = torch.cat([x, imfs], dim=2)

        # iTransformer编码
        enc_out = self.enc_embedding(x_concat, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)

        # 提取目标变量(功率)的深度特征
        # iTransformer倒置结构中，最后一个变量包含最丰富的功率耦合信息
        target_latent = enc_out[:, self.in_num_features - 1, :]

        # 路径A: 基准预测
        base_output = self.base_projection(target_latent).unsqueeze(-1)

        # 路径B: 未来修正 (使用x_dec未来风速)
        x_dec_norm = (x_dec - means[:, :1, :]) / stdev[:, :1, :]
        residual = self.refinement_head(target_latent, x_dec_norm)

        # 融合
        final_output = base_output + residual

        # 反归一化
        final_output = final_output * stdev[:, :, -1:] + means[:, :, -1:]
        return final_output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
