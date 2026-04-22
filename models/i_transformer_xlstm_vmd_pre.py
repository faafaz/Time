"""
iTransformer_xLSTM with VMD preprocessing (统一版本)。

架构:
1. VMD分解 (预处理阶段): 对目标变量进行VMD分解得到K个IMF分量
2. 特征拼接: 将分解得到的IMF分量与原始特征拼接
3. iTransformer: 建模变量之间的依赖关系 (variables as tokens)
4. xLSTM: 在时间维度进行时序建模
5. 预测头: 线性投影得到最终预测

支持通过配置切换不同变体:
- vmd_mode: 'standard' (固定K) | 'sparse_adaptive' (自适应稀疏，自动选K)
- xlstm_layers: xLSTM堆叠层数 (1 或 2)
- rnn_type: 'gru' | 'slstm'
- enable_vmd_preprocessing: 是否启用VMD预处理
- enable_xlstm: 是否启用xLSTM
- simplify_itransformer: 是否启用简化iTransformer (减小d_model/d_ff)

消融实验通过配置控制，不需要修改代码:
- mask_high_freq: bool = False - 屏蔽最高频2个模态做消融
- mask_low_freq: bool = False - 屏蔽最低频2个模态做消融
- mask_mid_freq: bool = False - 屏蔽中间4个模态做消融

所有变种通过配置实现，无需多份代码拷贝。
"""

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.iTransformer import DataEmbedding_inverted
from models.iTransformer_xLSTM import xLSTMStack
from models.vmd_decomposer import VMDDecomposer


class Model(nn.Module):
    """
    统一的iTransformer_xLSTM + VMD预处理模型。

    所有之前的变种 (Preprocessed, Preprocessed1, Preprocessed4, Preprocessed5)
    都可以通过配置参数在此模型中实现，消除了代码重复。
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in
        self.configs = configs

        # ========== VMD预处理配置 ==========
        self.enable_vmd_preprocessing = getattr(configs, 'enable_vmd_preprocessing', True)
        if self.enable_vmd_preprocessing:
            # 使用统一VMD分解器，根据配置选择模式
            self.vmd_decomposer = VMDDecomposer(configs)
            self.vmd_k = self.vmd_decomposer.vmd_k
        else:
            self.vmd_decomposer = None
            self.vmd_k = 0

        # ========== 消融实验配置 (通过配置开关，无需改代码) ==========
        self.mask_high_freq = getattr(configs, 'mask_high_freq', False)
        self.mask_low_freq = getattr(configs, 'mask_low_freq', False)
        self.mask_mid_freq = getattr(configs, 'mask_mid_freq', False)

        # ========== iTransformer配置 ==========
        self.simplify_itransformer = getattr(configs, 'simplify_itransformer', True)
        if self.simplify_itransformer:
            d_model_actual = getattr(configs, 'simplified_d_model', 64)
            d_ff_actual = getattr(configs, 'simplified_d_ff', 64)
        else:
            d_model_actual = configs.d_model
            d_ff_actual = configs.d_ff

        # 计算VMD拼接后的输入维度
        # 原始特征: in_num_features
        # VMD分量: vmd_k (仅对目标变量分解)
        # 总分量: in_num_features + vmd_k
        # +5 是为了兼容原有代码维度计算，保持兼容性
        self.input_dim_after_vmd = self.in_num_features + self.vmd_k + 5

        # iTransformer倒置嵌入和编码器
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            d_model_actual,
            configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        d_model_actual,
                        configs.n_heads
                    ),
                    d_model_actual,
                    d_ff_actual,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model_actual)
        )
        self.back_projection = nn.Linear(d_model_actual, configs.seq_len)

        # ========== xLSTM时序建模配置 ==========
        self.enable_xlstm = getattr(configs, 'enable_xlstm', True)
        xlstm_hidden = getattr(configs, 'xlstm_hidden', 64)
        enable_longconv = getattr(configs, 'enable_xlstm_longconv', True)
        xlstm_kernels = tuple(getattr(configs, 'xlstm_kernels', [7, 15, 25]))
        xlstm_layers = getattr(configs, 'xlstm_layers', 2)
        rnn_type = getattr(configs, 'rnn_type', 'slstm')

        # xLSTM输入维度
        in_dim_xlstm = self.input_dim_after_vmd

        self.xlstm = xLSTMStack(
            in_dim_xlstm,
            xlstm_hidden,
            layers=xlstm_layers,
            dropout=configs.dropout,
            enable_longconv=enable_longconv,
            kernels=xlstm_kernels,
            rnn_type=rnn_type,
            share_params=True
        )

        # ========== 预测头 ==========
        self.time_proj = nn.Linear(self.seq_len, self.pred_len)
        self.head = nn.Linear(xlstm_hidden, 1)

        # 缓存分解结果用于可视化/分析
        self.modes = None

    def get_vmd_sparsity_loss(self):
        """获取VMD稀疏损失 (仅sparse_adaptive模式有效)"""
        if self.enable_vmd_preprocessing and self.vmd_decomposer is not None:
            return self.vmd_decomposer.get_sparsity_loss()
        return torch.tensor(0.0, device=self.head.weight.device)

    def get_effective_vmd_k(self):
        """获取有效VMD模态数量 (仅sparse_adaptive模式有效)"""
        if self.enable_vmd_preprocessing and self.vmd_decomposer is not None:
            return self.vmd_decomposer.get_effective_k()
        return self.vmd_k

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        预测前向传播。

        Args:
            x_enc: [B, L, C] 编码器输入
            x_mark_enc: [B, L, D] 编码器时间特征
            x_dec: [B, pred_len, C] 解码器输入 (未使用)
            x_mark_dec: [B, pred_len, D] 解码器时间特征 (未使用)

        Returns:
            output: [B, pred_len, 1] 预测结果
        """
        # 1) RevIN归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # 2) VMD预处理: 分解目标变量并拼接
        if self.enable_vmd_preprocessing and self.vmd_decomposer is not None:
            # 提取目标变量 (最后一个特征)
            target_var = x[:, :, -1:]  # [B, L, 1]

            # 分解得到IMF分量
            imf_components = self.vmd_decomposer(target_var)  # [B, L, K]

            # ========== 消融实验: 屏蔽特定频率分量 ==========
            if self.mask_high_freq:
                # 屏蔽最高频2个模态
                imf_components[:, :, -2:] = 0.0
            if self.mask_low_freq:
                # 屏蔽最低频2个模态
                imf_components[:, :, 0:2] = 0.0
            if self.mask_mid_freq:
                # 屏蔽中间4个模态
                K = imf_components.shape[-1]
                if K >= 8:
                    imf_components[:, :, 2:6] = 0.0

            # 缓存分解结果
            self.modes = imf_components

            # 将IMF分量与原始特征拼接
            # x: [B, L, C], imf_components: [B, L, K]
            x_concat = torch.cat([x, imf_components], dim=2)  # [B, L, C + K]
        else:
            x_concat = x

        # 3) iTransformer编码 (变量作为token)
        enc_out = self.enc_embedding(x_concat, x_mark_enc)  # [B, n_vars, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [B, n_vars, d_model]

        # 4) 反向投影回时间维度
        time_feat = self.back_projection(enc_out)  # [B, n_vars, L]
        time_feat = time_feat.permute(0, 2, 1)  # [B, L, n_vars]

        # 5) xLSTM时序建模
        if self.enable_xlstm:
            self.xlstm = self.xlstm.to(time_feat.device)
            h = self.xlstm(time_feat)  # [B, L, H]
        else:
            h = nn.functional.gelu(self.xlstm.input_proj(time_feat))

        # 6) 时间投影到预测范围
        h_t = h.transpose(1, 2)  # [B, H, L]
        h_h = self.time_proj(h_t)  # [B, H, pred_len]
        h_h = h_h.transpose(1, 2)  # [B, pred_len, H]
        output = self.head(h_h)  # [B, pred_len, 1]

        # 7) 反归一化输出
        tgt_idx = self.in_num_features - 1
        output = output * stdev[:, :, tgt_idx:tgt_idx+1] + means[:, :, tgt_idx:tgt_idx+1]

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in [
            'ultra_short_term_forecast',
            'short_term_forecast',
            'long_term_forecast'
        ]:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None
