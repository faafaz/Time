"""
Variant of iTransformer_xLSTM that replaces the NFM branch with a VMD-based branch.

Key points:
- Keeps the same input/output interface as models.iTransformer_xLSTM.Model
- Uses VMD (or a VMD-inspired band decomposition fallback) to decompose the target into K modes
- Processes each mode separately and fuses mode-wise predictions
- Fuses the VMD branch with the iTransformer+xLSTM branch via a lightweight fusion module
- End-to-end trainable except the VMD decomposition itself (non-differentiable), which is treated as a fixed transform

Configs (all optional, with defaults):
- enable_vmd_branch: bool = True
- vmd_k: int = 4
- vmd_impl: str = 'auto'   # 'vmdpy' | 'fftbank' | 'auto'
- vmd_alpha: float = 2000.0
- vmd_tau: float = 0.0
- vmd_dc: int = 0
- vmd_init: int = 1
- vmd_tol: float = 1e-7
- vmd_max_iter: int = 500
- vmd_mode_heads: str = 'shared'   # 'shared' or 'separate'
- vmd_mode_fusion: str = 'gate'    # 'gate' or 'add'
- vmd_final_fusion_type: str = 'gate'  # uses LightweightFusion: 'gate' | 'add' | 'concat'

Note on preprocessing: if you prefer offline VMD to save compute, set enable_vmd_branch=True
and replace VMDDecomposer.forward() with cached/precomputed modes (not provided here).
"""

from typing import Optional
import math
import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.iTransformer import DataEmbedding_inverted
from models.iTransformer_xLSTM import xLSTMStack  # reuse the same temporal stack
from models.nfm_module import LightweightFusion   # reuse fusion for branch merging


class VMDDecomposer(nn.Module):
    """Variational Mode Decomposition (VMD) wrapper with FFT-band fallback.

    forward(x):
      x: [B, L, 1] (raw target)
      returns modes: [B, L, K]
    """
    def __init__(self, K: int = 4, impl: str = 'fftbank', alpha: float = 2000.0,
                 tau: float = 0.0, DC: int = 0, init: int = 1, tol: float = 1e-7,
                 max_iter: int = 500):
        super().__init__()
        self.K = K
        self.impl = impl
        # Make VMD parameters trainable
        self.vmd_alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=True)
        self.vmd_k = nn.Parameter(torch.tensor(float(K)), requires_grad=True)
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter

        # Try to detect vmdpy if impl is 'auto'
        self._use_vmdpy = False
        if impl in ('auto', 'vmdpy'):
            try:
                import vmdpy  # noqa: F401
                self._use_vmdpy = True
            except Exception:
                self._use_vmdpy = False
                if impl == 'vmdpy':
                    # If explicitly requested but not available, still fall back silently
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 1]
        B, L, C = x.shape
        assert C == 1, f"VMDDecomposer expects [B, L, 1], got {x.shape}"

        if self._use_vmdpy:
            # CPU-side vmdpy per-sample, then stack back to torch on original device
            modes_list = []
            import numpy as np
            import vmdpy
            x_np = x.detach().cpu().numpy()
            for b in range(B):
                signal = x_np[b, :, 0].astype(np.float64)
                u, _, _ = vmdpy.VMD(signal,
                                     alpha=float(self.vmd_alpha.item()),
                                     tau=self.tau,
                                     K=self.K,
                                     DC=self.DC,
                                     init=self.init,
                                     tol=self.tol)
                # u: [K, L]
                modes_list.append(torch.from_numpy(u.T.astype(np.float32)))  # [L, K]
            modes = torch.stack(modes_list, dim=0).to(x.device)  # [B, L, K]
            return modes
        else:
            # FFT soft filter bank fallback (differentiable w.r.t vmd_alpha, vmd_k)
            x_freq = torch.fft.rfft(x.squeeze(-1), dim=1, norm='ortho')  # [B, Nf]
            B, n_freq = x_freq.shape
            K_fixed = self.K
            # Frequency indices
            freq_idx = torch.arange(n_freq, device=x.device).float().unsqueeze(-1)  # [Nf, 1]
            # Trainable scale for partition (derived from vmd_k), clamp to reasonable range
            k_scale = 0.5 + torch.sigmoid(self.vmd_k)  # (0.5, 1.5)
            # Trainable sharpness via vmd_alpha (positive)
            alpha_eff = torch.nn.functional.softplus(self.vmd_alpha) + 1e-6
            # Convert alpha to bandwidth (larger alpha -> narrower band)
            sigma = torch.clamp(n_freq / (K_fixed * alpha_eff), min=1.0)
            # Centers across frequency axis
            centers = (torch.arange(K_fixed, device=x.device).float() + 0.5) * (n_freq / K_fixed) * k_scale  # [K]
            # Gaussian weights per frequency bin and mode
            diffs = (freq_idx - centers) / sigma  # [Nf, K]
            w = torch.exp(-0.5 * (diffs ** 2))  # [Nf, K]
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # normalize across K at each freq
            # Apply soft masks in frequency domain
            x_freq_expanded = x_freq.unsqueeze(-1) * w.unsqueeze(0)  # [B, Nf, K]
            x_freq_flat = x_freq_expanded.permute(0, 2, 1).reshape(B * K_fixed, n_freq)  # [B*K, Nf]
            x_time_flat = torch.fft.irfft(x_freq_flat, n=L, dim=1, norm='ortho')  # [B*K, L]
            modes = x_time_flat.reshape(B, K_fixed, L).permute(0, 2, 1)  # [B, L, K]
            return modes


class VMDBranch(nn.Module):
    """Process VMD modes and produce a prediction with independent linear layers per mode.

    Architecture:
    - Decomposer: VMDDecomposer -> modes [B, L, K]
    - Per-mode independent linear layers: each mode has its own Linear(seq_len -> pred_len)
    - Mode fusion: element-wise addition of all mode predictions -> [B, pred_len, 1]
    - Final fusion with xLSTM output via LightweightFusion

    Key changes:
    - Each VMD mode gets its own dedicated linear layer (no weight sharing)
    - All mode predictions are summed (element-wise addition)
    - Output is [B, pred_len, 1] for fusion with xLSTM branch
    """
    def __init__(self, seq_len: int, pred_len: int, K: int = 4,
                 decomposer: Optional[VMDDecomposer] = None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.K = K

        self.decomposer = decomposer if decomposer is not None else VMDDecomposer(K=K)

        # Create independent linear layers for each VMD mode
        # Each layer: input [B, 1, seq_len] -> output [B, 1, pred_len]
        self.mode_linear_layers = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(K)
        ])

    def forward(self, target_raw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VMD branch with independent linear layers.

        Args:
            target_raw: [B, L, 1] (original scale)

        Returns:
            fused_pred: [B, pred_len, 1] (sum of all mode predictions)
        """
        # Decompose target into K modes
        modes = self.decomposer(target_raw)  # [B, L, K]

        # Process each mode with its own independent linear layer
        mode_preds = []
        for k in range(self.K):
            # Extract k-th mode
            mk = modes[:, :, k:k+1]             # [B, L, 1]

            # Reshape for linear layer: [B, L, 1] -> [B, 1, L]
            mk_t = mk.transpose(1, 2)           # [B, 1, L]

            # Apply independent linear layer for this mode
            # Input: [B, 1, L], Output: [B, 1, pred_len]
            pk = self.mode_linear_layers[k](mk_t)  # [B, 1, pred_len]

            # Reshape back: [B, 1, pred_len] -> [B, pred_len, 1]
            pk = pk.transpose(1, 2)             # [B, pred_len, 1]

            mode_preds.append(pk)

        # Fuse all mode predictions via element-wise addition
        # Stack all predictions: [B, pred_len, 1] x K
        # Sum across modes: [B, pred_len, 1]
        fused_pred = torch.stack(mode_preds, dim=0).sum(dim=0)  # [B, pred_len, 1]

        return fused_pred


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in
        self.configs = configs

        # iTransformer configuration (reuse simplified option)
        self.simplify_itransformer = getattr(configs, 'simplify_itransformer', False)
        if self.simplify_itransformer:
            d_model_actual = getattr(configs, 'simplified_d_model', 96)
            d_ff_actual = getattr(configs, 'simplified_d_ff', 96)
        else:
            d_model_actual = configs.d_model
            d_ff_actual = configs.d_ff

        # iTransformer inverted embedding and encoder
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
        self.back_projection = nn.Linear(d_model_actual, configs.seq_len)

        # Feature selection (optional, reuse existing module if enabled)
        self.enable_feature_selection = getattr(configs, 'enable_feature_selection', False)
        self.feature_selection_type = getattr(configs, 'feature_selection_type', 'channel_attn')
        self.selected_var_indices = getattr(configs, 'selected_var_indices', list(range(max(1, self.in_num_features - 1))))
        if self.enable_feature_selection:
            from models.feature_selection import GatedFeatureSelection, ChannelAttention, VariableSelectionNetwork
            n_features = len(self.selected_var_indices)
            if self.feature_selection_type == 'gflu':
                sparsity_init = getattr(configs, 'gflu_sparsity_init', 0.5)
                learnable_t = getattr(configs, 'gflu_learnable_t', True)
                self.feature_selector = GatedFeatureSelection(n_features, sparsity_init=sparsity_init, learnable_t=learnable_t, use_gating=False)
            elif self.feature_selection_type == 'channel_attn':
                reduction = getattr(configs, 'channel_attn_reduction', 4)
                self.feature_selector = ChannelAttention(n_features, reduction=reduction)
            elif self.feature_selection_type == 'vsn':
                vsn_hidden = getattr(configs, 'vsn_hidden', 64)
                self.feature_selector = VariableSelectionNetwork(n_features, vsn_hidden, configs.dropout)
            else:
                raise ValueError(f"Unknown feature_selection_type: {self.feature_selection_type}")
        else:
            self.feature_selector = None

        # Target bypass
        self.enable_target_bypass = getattr(configs, 'enable_target_bypass', False)

        # VMD branch configuration
        self.enable_vmd_branch = getattr(configs, 'enable_vmd_branch', True)
        if self.enable_vmd_branch:
            vmd_k = getattr(configs, 'vmd_k', 12)
            vmd_impl = getattr(configs, 'vmd_impl', 'fftbank')
            vmd_alpha = getattr(configs, 'vmd_alpha', 2000.0)
            vmd_tau = getattr(configs, 'vmd_tau', 0.0)
            vmd_dc = getattr(configs, 'vmd_dc', 0)
            vmd_init = getattr(configs, 'vmd_init', 1)
            vmd_tol = getattr(configs, 'vmd_tol', 1e-7)
            vmd_max_iter = getattr(configs, 'vmd_max_iter', 500)

            decomposer = VMDDecomposer(K=vmd_k, impl=vmd_impl, alpha=vmd_alpha, tau=vmd_tau,
                                       DC=vmd_dc, init=vmd_init, tol=vmd_tol, max_iter=vmd_max_iter)
            # New VMDBranch with independent linear layers per mode and element-wise addition fusion
            self.vmd_branch = VMDBranch(seq_len=self.seq_len, pred_len=self.pred_len, K=vmd_k,
                                        decomposer=decomposer)

            # Final branch fusion (reuse LightweightFusion to fuse VMD output with xLSTM output)
            vmd_final_fusion_type = getattr(configs, 'vmd_final_fusion_type', getattr(configs, 'nfm_fusion_type', 'add'))
            vmd_fusion_hidden = getattr(configs, 'vmd_fusion_hidden', getattr(configs, 'nfm_fusion_hidden', 64))
            self.fusion_module = LightweightFusion(fusion_type=vmd_final_fusion_type, hidden_dim=vmd_fusion_hidden)

        # xLSTM temporal modeling
        self.enable_xlstm = getattr(configs, 'enable_xlstm', True)
        xlstm_hidden = getattr(configs, 'xlstm_hidden', 64)
        enable_longconv = getattr(configs, 'enable_xlstm_longconv', True)
        xlstm_kernels = tuple(getattr(configs, 'xlstm_kernels', [7, 15, 25]))
        rnn_type = getattr(configs, 'rnn_type', 'slstm')

        # Estimate input dim to xLSTM
        if self.enable_feature_selection:
            in_dim_base = len(self.selected_var_indices)
        else:
            in_dim_base = self.in_num_features + 5
        in_dim_xlstm = in_dim_base + 1 if self.enable_target_bypass else in_dim_base

        self.xlstm = xLSTMStack(
            in_dim_xlstm, xlstm_hidden, layers=2, dropout=configs.dropout,
            enable_longconv=enable_longconv, kernels=xlstm_kernels,
            rnn_type=rnn_type, share_params=True
        )

        # Time projection and final head for LSTM branch
        self.time_proj = nn.Linear(self.seq_len, self.pred_len)
        self.head = nn.Linear(xlstm_hidden, 1)

        self.last_feature_importance = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 0) Raw target for VMD branch (before normalization)
        target_var_raw = x_enc[:, :, self.in_num_features - 1:self.in_num_features]  # [B, L, 1]

        # 1) RevIN normalization for main branch inputs
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # 2) Optional target bypass
        if self.enable_target_bypass:
            target_norm = x[:, :, self.in_num_features - 1:self.in_num_features]  # [B, L, 1]
            x_for_itransformer = x
        else:
            target_norm = None
            x_for_itransformer = x

        # 3) iTransformer encoding (variables as tokens)
        enc_out = self.enc_embedding(x_for_itransformer, x_mark_enc)  # [B, n_vars, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)             # [B, n_vars, d_model]

        # 4) Back-project to time dimension
        time_feat = self.back_projection(enc_out)                      # [B, n_vars, L]
        time_feat = time_feat.permute(0, 2, 1)                         # [B, L, n_vars]

        # 5) Feature selection
        time_feat_selected = time_feat
        feature_importance_weights = None
        if self.enable_feature_selection and self.feature_selector is not None:
            time_feat_selected, feature_importance_weights, _ = self.feature_selector(time_feat_selected)

        # 6) (Optional) concat bypassed target
        if self.enable_target_bypass and target_norm is not None:
            time_feat_selected = torch.cat([target_norm, time_feat_selected], dim=2)

        # 7) Temporal modeling with xLSTM
        if self.enable_xlstm:
            self.xlstm = self.xlstm.to(time_feat_selected.device)
            h = self.xlstm(time_feat_selected)         # [B, L, H]
        else:
            h = torch.nn.functional.gelu(self.xlstm.input_proj(time_feat_selected))

        # 8) Project along time to horizon
        h_t = h.transpose(1, 2)                        # [B, H, L]
        h_h = self.time_proj(h_t)                      # [B, H, pred_len]
        h_h = h_h.transpose(1, 2)                      # [B, pred_len, H]
        lstm_pred = self.head(h_h)                     # [B, pred_len, 1]

        # 9) VMD branch and fusion
        if self.enable_vmd_branch:
            vmd_pred = self.vmd_branch(target_var_raw)  # [B, pred_len, 1] (original scale)
            # Denorm LSTM to original scale, then fuse
            tgt_idx = self.in_num_features - 1
            lstm_pred_denorm = lstm_pred * stdev[:, :, tgt_idx:tgt_idx+1] + means[:, :, tgt_idx:tgt_idx+1]
            dec_out = self.fusion_module(lstm_pred_denorm, vmd_pred)
        else:
            tgt_idx = self.in_num_features - 1
            dec_out = lstm_pred * stdev[:, :, tgt_idx:tgt_idx+1] + means[:, :, tgt_idx:tgt_idx+1]

        if feature_importance_weights is not None:
            self.last_feature_importance = feature_importance_weights.detach()
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['ultra_short_term_forecast', 'short_term_forecast', 'long_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None

