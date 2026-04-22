"""
Hybrid: iTransformer encoder for variable relations, then xLSTM processes temporal dimension.
- Reuse iTransformer inverted embedding and encoder
- Back-project encoder outputs to time dimension [B, seq_len, n_vars]
- xLSTM stack (sLSTM/GRU + optional LongConv pre-mix + residual + LayerNorm) models time
- Project along time to pred_len and emit [B, pred_len, 1]

Key Changes (v7 - Simplified to 2-Layer Unidirectional with Parameter Sharing):
1. **2-Layer Unidirectional LSTM with Parameter Sharing**: (NEW in v7)
   - Removed bidirectional LSTM mechanism
   - Fixed to 2 layers of unidirectional LSTM
   - Both layers share the same parameters (deep recurrence)
   - Increases depth without increasing parameters
   - Always enabled (no ablation switch)
2. **Simplified iTransformer**: Lightweight variable relation modeling
   - Optional simplified configuration for reduced parameters
   - Enable via 'simplify_itransformer=True'
3. **sLSTM Integration**: Replaced GRU with sLSTM (scalar LSTM with exponential gating)
   - Exponential gating for better long-term dependencies
   - Memory normalization for numerical stability
   - ~33% more parameters than GRU, but 10-15% performance improvement
4. **Configurable RNN Type**: Support both 'gru' and 'slstm' via config
   - rnn_type: 'gru' (lightweight) or 'slstm' (better performance)
5. **GFLU Feature Selection**: Learnable sparse feature selection
   - Gated Feature Learning Unit with t-softmax
   - Minimal parameters (~n_features), strong sparsity
   - Alternative: Channel Attention or VSN (TFT-style)
6. **Target Variable Bypass**: Direct path for target variable
   - Original target variable bypasses iTransformer
   - Concatenated with iTransformer-processed features
   - Preserves raw temporal patterns for LSTM

Ablation Switches:
- simplify_itransformer: Use simplified iTransformer configuration (default: False)
- rnn_type: 'gru' or 'slstm' (default: 'slstm')
- enable_xlstm: if False, bypass RNN and use a linear temporal projection baseline
- enable_xlstm_longconv: enable/disable LongConv pre-mixing in each layer
- enable_feature_selection: enable GFLU/Channel Attention (default: False)
- feature_selection_type: 'gflu', 'channel_attn', or 'vsn' (default: 'gflu')
- enable_target_bypass: bypass iTransformer for target variable (default: False)
- selected_var_indices: list of variable indices to feed into RNN (legacy, deprecated if using GFLU)

All new hyperparams have safe defaults if not present in configs.
"""

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.iTransformer import DataEmbedding_inverted

# class LongConvMix(nn.Module):
#     """Depthwise temporal conv pre-mix with multi-kernel and gated residual."""
#     def __init__(self, hidden: int, kernels=(5, 11, 23), dropout=0.0):
#         super().__init__()
#         self.dw_convs = nn.ModuleList([
#             nn.Conv1d(hidden, hidden, k, padding=k // 2, groups=hidden) for k in kernels
#         ])
#         self.proj = nn.Conv1d(hidden * len(kernels), hidden, kernel_size=1)
#         self.dropout = nn.Dropout(dropout)
#         self.gate = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x):
#         # x: [B, L, H]
#         x_ch = x.transpose(1, 2)  # [B, H, L]
#         outs = [conv(x_ch) for conv in self.dw_convs]
#         mix = torch.cat(outs, dim=1)
#         mix = self.proj(mix)
#         mix = self.dropout(mix)
#         mix = mix.transpose(1, 2)  # [B, L, H]
#         return x + torch.sigmoid(self.gate) * mix


class LongConvMix(nn.Module):
    def __init__(self, d_model, kernels=[7, 15, 25], dropout=0.0): # 确保包含 kernels 参数
        super().__init__()
        
        # ... (原有的卷积层定义保持不变) ...
        # 假设你有类似这样的卷积列表:
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, k, padding=k//2, groups=d_model),
                nn.BatchNorm1d(d_model),
                nn.GELU()
            ) for k in kernels
        ])
        
        # ================= [新增 1] 门控层 =================
        # 如果你原本没有计算权重(只是相加)，必须加这个层，否则没有"权重"可画
        # 输入: 所有分支的拼接特征, 输出: 每个分支的权重分数
        self.num_kernels = len(kernels)
        self.gate_fc = nn.Linear(d_model * self.num_kernels, self.num_kernels)
        
        # [新增 2] 用于暴露权重的变量
        self.last_weights = None 
        # ===================================================

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, C] -> 转置 [B, C, L]
        x_t = x.permute(0, 2, 1)
        
        # 1. 获取各尺度输出
        outputs = [conv(x_t) for conv in self.convs] # List of [B, C, L]
        
        # ================= [新增 3] 计算并保存权重 =================
        # 拼接特征用于计算门控
        cat_feat = torch.cat(outputs, dim=1) # [B, C*K, L]
        cat_feat_t = cat_feat.permute(0, 2, 1) # [B, L, C*K]
        
        # 计算权重 logits -> softmax
        gate_logits = self.gate_fc(cat_feat_t) # [B, L, K]
        # temperature = 0.8  # 建议尝试 0.5, 0.2, 或 0.1
        
        # weights = torch.softmax(gate_logits / temperature, dim=-1)
        weights = torch.softmax(gate_logits, dim=-1) # [B, L, K]
        
        # [关键] 保存权重! (Detach 防止显存泄漏)
        self.last_weights = weights.detach().cpu().numpy()
        # ===========================================================

        # 2. 加权融合
        # weights: [B, L, K] -> [B, K, 1, L]
        weights_expanded = weights.permute(0, 2, 1).unsqueeze(2)
        stack_outputs = torch.stack(outputs, dim=1) # [B, K, C, L]
        
        # 加权求和
        out = torch.sum(stack_outputs * weights_expanded, dim=1) # [B, C, L]
        out = out.permute(0, 2, 1) # [B, L, C]
        
        return self.dropout(out)


# Removed BidirectionalFusion class (no longer needed in v7)


class xLSTMStack(nn.Module):
    """
    xLSTM-based temporal modeling stack with configurable RNN type.

    Supports:
    - 'slstm': Scalar LSTM with exponential gating (recommended for performance)
    - 'gru': Standard GRU (lightweight option)

    Features:
    - LongConv pre-mixing for multi-scale temporal patterns
    - LayerNorm for training stability
    - Residual connections for gradient flow

    Args:
        in_dim: Input feature dimension
        hidden: Hidden state dimension
        layers: Number of stacked RNN layers
        dropout: Dropout probability
        enable_longconv: Whether to use LongConv pre-mixing
        kernels: Kernel sizes for LongConv (default: (5, 11, 23))
        rnn_type: 'slstm' or 'gru' (default: 'slstm')
        share_params: Whether to share parameters across layers (default: False)
    """
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float = 0.1,
                 enable_longconv: bool = True, kernels=(5, 11, 23), rnn_type: str = 'slstm',
                 share_params: bool = False):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mixers = nn.ModuleList()
        self.enable_longconv = enable_longconv
        self.rnn_type = rnn_type
        self.share_params = share_params
        self.num_layers = layers

        # Import sLSTM if needed
        if rnn_type == 'slstm':
            from models.xlstm_modules import sLSTM

        if share_params:
            # ===== Parameter Sharing Mode =====
            # Create a single RNN layer and reuse it across all layers
            # This increases depth without increasing parameters

            # Create shared RNN layer
            if rnn_type == 'slstm':
                shared_rnn = sLSTM(hidden, hidden, num_layers=1, batch_first=True, dropout=0.0)
            elif rnn_type == 'gru':
                shared_rnn = nn.GRU(hidden, hidden, batch_first=True)
            else:
                raise ValueError(f"Unknown rnn_type: {rnn_type}. Must be 'slstm' or 'gru'.")

            # Create shared mixer
            shared_mixer = LongConvMix(hidden, kernels, dropout) if enable_longconv else nn.Identity()

            # Reuse the same layer for all iterations
            for _ in range(layers):
                self.layers.append(shared_rnn)
                self.mixers.append(shared_mixer)
                # Each layer still has its own LayerNorm (not shared)
                self.norms.append(nn.LayerNorm(hidden))
        else:
            # ===== Independent Layers Mode (Default) =====
            for _ in range(layers):
                self.mixers.append(LongConvMix(hidden, kernels, dropout) if enable_longconv else nn.Identity())

                # Select RNN type
                if rnn_type == 'slstm':
                    # sLSTM: Exponential gating + memory normalization
                    # ~33% more parameters than GRU, but 10-15% better performance
                    self.layers.append(sLSTM(hidden, hidden, num_layers=1, batch_first=True, dropout=0.0))
                elif rnn_type == 'gru':
                    # GRU: Lightweight option
                    self.layers.append(nn.GRU(hidden, hidden, batch_first=True))
                else:
                    raise ValueError(f"Unknown rnn_type: {rnn_type}. Must be 'slstm' or 'gru'.")

                self.norms.append(nn.LayerNorm(hidden))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through xLSTM stack.

        Args:
            x: Input tensor [B, L, C]

        Returns:
            h: Output tensor [B, L, H]
        """
        # x: [B, L, C]
        h = self.input_proj(x)
        for mix, rnn, ln in zip(self.mixers, self.layers, self.norms):
            h = ln(h)
            h = mix(h)
            # Both sLSTM and GRU return (output, state)
            out, _ = rnn(h)
            h = h + self.dropout(out)
        return h  # [B, L, H]


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in
        self.configs = configs

        # ========== NEW: Simplified iTransformer Configuration ==========
        self.simplify_itransformer = getattr(configs, 'simplify_itransformer', False)

        # Adjust iTransformer parameters if simplification is enabled
        if self.simplify_itransformer:
            # Simplified configuration: reduce d_model and d_ff
            d_model_actual = getattr(configs, 'simplified_d_model', 64)
            d_ff_actual = getattr(configs, 'simplified_d_ff', 64)
        else:
            # Standard configuration
            d_model_actual = configs.d_model
            d_ff_actual = configs.d_ff

        # iTransformer inverted embedding and encoder (variables as tokens)
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, d_model_actual, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
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

        # Back-project from latent to time dimension per variable
        self.back_projection = nn.Linear(d_model_actual, configs.seq_len)

        # ========== NEW: Feature Selection Configuration ==========
        self.enable_feature_selection = getattr(configs, 'enable_feature_selection', False)
        self.feature_selection_type = getattr(configs, 'feature_selection_type', 'channel_attn')

        # Legacy: manual feature selection via indices (deprecated if using learnable selection)
        self.selected_var_indices = getattr(configs, 'selected_var_indices',
                                           [0, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16])

        # ========== NEW: Target Variable Bypass Configuration ==========
        self.enable_target_bypass = getattr(configs, 'enable_target_bypass', False)

        # ========== NEW in v8: NFM Branch for Target Variable Prediction ==========
        self.enable_nfm_branch = getattr(configs, 'enable_nfm_branch', True)
        if self.enable_nfm_branch:
            from models.nfm_module import NFMPredictor, LightweightFusion

            nfm_hidden_dim = getattr(configs, 'nfm_hidden_dim', 64)
            nfm_num_layers = getattr(configs, 'nfm_num_layers', 1)
            nfm_hidden_factor = getattr(configs, 'nfm_hidden_factor', 3)
            nfm_siren_hidden = getattr(configs, 'nfm_siren_hidden', 64)
            nfm_omega = getattr(configs, 'nfm_omega', 30.0)
            nfm_fusion_type = getattr(configs, 'nfm_fusion_type', 'gate')
            nfm_fusion_hidden = getattr(configs, 'nfm_fusion_hidden', 64)

            # NFM predictor for target variable
            self.nfm_predictor = NFMPredictor(
                seq_len=configs.seq_len,
                pred_len=configs.pred_len,
                hidden_dim=nfm_hidden_dim,
                num_layers=nfm_num_layers,
                hidden_factor=nfm_hidden_factor,
                dropout=configs.dropout,
                siren_hidden=nfm_siren_hidden,
                omega=nfm_omega
            )

            # Lightweight fusion module
            self.fusion_module = LightweightFusion(
                fusion_type=nfm_fusion_type,
                hidden_dim=nfm_fusion_hidden
            )

        # xLSTM temporal module configuration
        self.enable_xlstm = getattr(configs, 'enable_xlstm', True)
        xlstm_hidden = getattr(configs, 'xlstm_hidden', 64)
        enable_longconv = getattr(configs, 'enable_xlstm_longconv', True)
        xlstm_kernels = tuple(getattr(configs, 'xlstm_kernels', [15, 25, 35]))

        # RNN type: 'slstm' (recommended) or 'gru' (lightweight)
        rnn_type = getattr(configs, 'rnn_type', 'slstm')

        # Determine input dimension for xLSTM
        # If using learnable feature selection, we don't know the exact dimension yet
        # So we use a placeholder and will adjust dynamically
        if self.enable_feature_selection:
            # Placeholder: will be determined after feature selection
            # For now, assume all variables might be selected
            in_dim_base = len(self.selected_var_indices)
        else:
            in_dim_base = self.in_num_features + 5

        # Add 1 dimension if target bypass is enabled
        if self.enable_target_bypass:
            in_dim_xlstm = in_dim_base + 1
        else:
            in_dim_xlstm = in_dim_base

        # ========== NEW in v7: Fixed 2-Layer Unidirectional LSTM with Parameter Sharing ==========
        # Always use 2 layers with parameter sharing (no ablation switch)
        # This increases depth without increasing parameters
        self.xlstm = xLSTMStack(
            in_dim_xlstm,
            xlstm_hidden,
            layers=3,  # Fixed to 2 layers
            dropout=configs.dropout,
            enable_longconv=enable_longconv,
            kernels=xlstm_kernels,
            rnn_type=rnn_type,
            share_params=True  # Always share parameters between the 2 layers
        )

        # ========== NEW: Initialize Feature Selection Module ==========
        if self.enable_feature_selection:
            from models.feature_selection import GatedFeatureSelection, ChannelAttention, VariableSelectionNetwork

            n_features = len(self.selected_var_indices)

            if self.feature_selection_type == 'gflu':
                # GFLU: Minimal parameters, strong sparsity
                sparsity_init = getattr(configs, 'gflu_sparsity_init', 0.5)
                learnable_t = getattr(configs, 'gflu_learnable_t', True)
                self.feature_selector = GatedFeatureSelection(
                    n_features,
                    sparsity_init=sparsity_init,
                    learnable_t=learnable_t,
                    use_gating=False  # Simple version without gating
                )
            elif self.feature_selection_type == 'channel_attn':
                # Channel Attention: Simple and effective
                reduction = getattr(configs, 'channel_attn_reduction', 4)
                self.feature_selector = ChannelAttention(n_features, reduction=reduction)
            elif self.feature_selection_type == 'vsn':
                # Variable Selection Network: TFT-style, strong interpretability
                vsn_hidden = getattr(configs, 'vsn_hidden', 64)
                self.feature_selector = VariableSelectionNetwork(n_features, vsn_hidden, configs.dropout)
            else:
                raise ValueError(f"Unknown feature_selection_type: {self.feature_selection_type}")
        else:
            self.feature_selector = None

        # Horizon projection along time: [B, L, H] -> [B, pred_len, H]
        self.time_proj = nn.Linear(self.seq_len, self.pred_len)
        # Final head: [B, pred_len, H] -> [B, pred_len, 1]
        self.head = nn.Linear(xlstm_hidden, 1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting with learnable feature selection, optional target bypass, and NFM branch.

        Flow (v8 with NFM):
        1. RevIN normalization
        2. (Optional NFM) Extract raw target variable from ORIGINAL input for NFM branch
        3. (Optional) Extract target variable for bypass path
        4. iTransformer encoder (variables as tokens)
        5. Back-project to time dimension [B, n_vars, d_model] -> [B, seq_len, n_vars]
        6. Manual feature selection (legacy) or learnable feature selection (GFLU/Channel Attn/VSN)
        7. (Optional) Concatenate bypassed target variable
        8. xLSTM/GRU temporal modeling
        9. Time projection and prediction head
        10. (Optional NFM) NFM prediction on raw target variable
        11. (Optional NFM) Fusion of LSTM and NFM predictions
        12. Inverse RevIN
        """
        # ========== NEW in v8: Extract raw target variable BEFORE any processing ==========
        # This is the key difference: NFM processes the ORIGINAL target variable
        if self.enable_nfm_branch:
            # Extract target variable from ORIGINAL input (before RevIN)
            target_var_raw = x_enc[:, :, self.in_num_features - 1:self.in_num_features]  # [B, seq_len, 1] - raw target variable

        # 1. RevIN normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # 2. (Optional) Extract target variable for bypass path
        # Target variable is at index 0 (first dimension)
        if self.enable_target_bypass:
            target_raw = x[:, :, self.in_num_features - 1:self.in_num_features]  # [B, seq_len, 1] - normalized target
            # Process other variables through iTransformer
            x_for_itransformer = x  # Still use all variables for iTransformer
        else:
            target_raw = None
            x_for_itransformer = x

        # 3. iTransformer embedding + encoder, variables as tokens
        enc_out = self.enc_embedding(x_for_itransformer, x_mark_enc)  # [B, n_vars, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)             # [B, n_vars, d_model]

        # 4. Back-project to time dimension and permute to temporal tokens
        time_feat = self.back_projection(enc_out)                      # [B, n_vars, seq_len]
        time_feat = time_feat.permute(0, 2, 1)                         # [B, seq_len, n_vars]

        # 5. Feature Selection
        n_vars = time_feat.size(2)

        # 5a. Manual feature selection (legacy)
        # valid_indices = [i for i in self.selected_var_indices if i < n_vars]
        # if len(valid_indices) == 0:
        #     valid_indices = [0]
        # time_feat_selected = time_feat[:, :, valid_indices]  # [B, seq_len, len(valid_indices)]
        time_feat_selected = time_feat  # [B, seq_len, len(valid_indices)]

        # 5b. Learnable feature selection (GFLU/Channel Attn/VSN)
        feature_importance_weights = None
        if self.enable_feature_selection and self.feature_selector is not None:
            # Apply learnable feature selection
            time_feat_selected, feature_importance_weights, _ = self.feature_selector(time_feat_selected)
            # time_feat_selected: [B, seq_len, len(valid_indices)]
            # feature_importance_weights: [len(valid_indices)] or [B, len(valid_indices)]

        # 6. (Optional) Concatenate bypassed target variable
        if self.enable_target_bypass and target_raw is not None:
            # Concatenate raw target with iTransformer-processed features
            time_feat_selected = torch.cat([target_raw, time_feat_selected], dim=2)
            # [B, seq_len, 1 + len(valid_indices)]

        # 7. xLSTM/GRU temporal modeling (2-layer unidirectional with parameter sharing)
        if self.enable_xlstm:
            # ===== 2-Layer Unidirectional LSTM with Parameter Sharing =====
            # Ensure xLSTM is on the same device
            self.xlstm = self.xlstm.to(time_feat_selected.device)
            h = self.xlstm(time_feat_selected)  # [B, seq_len, hidden]
        else:
            # Baseline: simple linear projection to hidden without temporal recurrence
            h = nn.functional.gelu(self.xlstm.input_proj(time_feat_selected))  # [B, seq_len, hidden]

        # 8. Project along time to horizon
        h_t = h.transpose(1, 2)                # [B, hidden, seq_len]
        h_h = self.time_proj(h_t)              # [B, hidden, pred_len]
        h_h = h_h.transpose(1, 2)              # [B, pred_len, hidden]
        lstm_pred = self.head(h_h)             # [B, pred_len, 1] - LSTM prediction

        # ========== NEW in v8: NFM Branch Processing ==========
        if self.enable_nfm_branch:
            # 10. NFM prediction on raw target variable
            # NFM processes the ORIGINAL target variable (before any transformation)
            nfm_pred = self.nfm_predictor(target_var_raw)  # [B, pred_len, 1]

            # 11. Fusion of LSTM and NFM predictions
            # Both predictions are in the original scale (before RevIN denormalization)
            # So we need to denormalize LSTM prediction first, then fuse
            lstm_pred_denorm = lstm_pred * stdev[:, :, self.in_num_features - 1:self.in_num_features] + means[:, :, self.in_num_features - 1:self.in_num_features]

            # Fuse predictions (both in original scale)
            dec_out = self.fusion_module(lstm_pred_denorm, nfm_pred)  # [B, pred_len, 1]
        else:
            # 9. Inverse RevIN for target variable (original flow)
            dec_out = lstm_pred * stdev[:, :, self.in_num_features - 1:self.in_num_features] + means[:, :, self.in_num_features - 1:self.in_num_features]

        # Store feature importance for interpretability (optional)
        if feature_importance_weights is not None:
            self.last_feature_importance = feature_importance_weights.detach()

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['ultra_short_term_forecast', 'short_term_forecast', 'long_term_forecast']:
            # Slice: drop first 2 meta-features if present (keep target + other vars)
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None

