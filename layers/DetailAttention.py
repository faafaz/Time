import torch
import torch.nn as nn
from math import sqrt


class LocalDetailAttention(nn.Module):
    """
    Local window self-attention focused on short-range dependencies for detail enhancement.
    - Non-causal by default; set mask_flag=True to enable causal masking in addition to windowing.
    - Applies a fixed window mask |i - j| <= window_size around each query position.
    """

    def __init__(self, mask_flag: bool = False, window_size: int = 8, scale=None,
                 attention_dropout: float = 0.1, output_attention: bool = False,
                 efficient: bool = True, dilation: int = 1):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.window_size = max(1, int(window_size))
        self.dilation = max(1, int(dilation))
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.efficient = bool(efficient)

    def _build_window_mask(self, L_q: int, L_k: int, device: torch.device):
        # mask True means to be masked as -inf, False means keep
        # shape: [L_q, L_k]
        idx_q = torch.arange(L_q, device=device).unsqueeze(1)
        idx_k = torch.arange(L_k, device=device).unsqueeze(0)
        dist = torch.abs(idx_q - idx_k)
        keep = dist <= self.window_size
        return ~keep  # True for positions outside window

    def _build_indices(self, L: int, device: torch.device):
        # 生成每个位置 i 的局部窗口索引，带 dilation：i + t*d，t in [-w, w]
        radius = self.window_size
        offsets = torch.arange(-radius, radius + 1, device=device) * self.dilation  # [win]
        base = torch.arange(L, device=device).unsqueeze(1)  # [L,1]
        idx = base + offsets.unsqueeze(0)  # [L, win]
        idx = idx.clamp(0, L - 1)
        return idx  # [L, win]

    def forward(self, queries, keys, values, attn_mask=None):
        # queries/keys/values: [B, L, H, E]
        B, L_q, H, E = queries.shape
        _, L_k, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        if not self.efficient:
            # 原始路径：先算全量，再窗口 mask
            scores = torch.einsum("blhe,bshe->bhls", queries, keys)
            window_mask = self._build_window_mask(L_q, L_k, device=queries.device)
            scores.masked_fill_(window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            if self.mask_flag and attn_mask is not None:
                scores.masked_fill_(attn_mask.mask, float('-inf'))
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)
            return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)

        # 高效路径：仅计算窗口内 scores，复杂度 O(B·H·L·w)
        idx = self._build_indices(L_k, queries.device)  # [L, win]
        win = idx.shape[1]
        # 扩展到批次与头，并在时间维 gather K/V（确保 index 维度与 input 一致）
        idx_bh = idx.view(1, L_k, 1, win, 1).expand(B, L_k, H, win, 1)  # [B,L,H,win,1]
        keys_e = keys.unsqueeze(3).expand(B, L_k, H, win, E)            # [B,L,H,win,E]
        values_e = values.unsqueeze(3).expand(B, L_k, H, win, D)        # [B,L,H,win,D]
        K_win = torch.gather(keys_e, dim=1, index=idx_bh.expand(B, L_k, H, win, E))    # [B,L,H,win,E]
        V_win = torch.gather(values_e, dim=1, index=idx_bh.expand(B, L_k, H, win, D))  # [B,L,H,win,D]
        # 调整维度顺序以匹配爱因斯坦求和公式（blhe, blwhe -> blhw）
        K_win = K_win.permute(0, 1, 3, 2, 4)  # [B,L,win,H,E]
        V_win = V_win.permute(0, 1, 3, 2, 4)  # [B,L,win,H,D]

        # 计算局部 scores：Q[i] 与其窗口 K_win[i,:]
        scores = torch.einsum('blhe,blwhe->blhw', queries, K_win) * scale  # [B,L,H,win]

        # 可选：叠加因果掩码（仅在窗口内）
        if self.mask_flag and attn_mask is not None:
            # 将全局下三角 mask 投影到窗口索引上（近似：对窗口内违反因果的元素置 -inf）
            # 简化：当 j > i 时屏蔽（基于 idx 与 i 的关系）
            pos_i = torch.arange(L_q, device=queries.device).unsqueeze(1).expand(L_q, win)  # [L,win]
            causal_mask = (idx > pos_i).unsqueeze(0).unsqueeze(2)  # [1,L,1,win]
            scores = scores.masked_fill(causal_mask, float('-inf'))

        A = self.dropout(torch.softmax(scores, dim=-1))  # [B,L,H,win]
        V = torch.einsum('blhw,blwhe->blhe', A, V_win)  # [B,L,H,E]
        return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)


class FusedFullLocalAttention(nn.Module):
    """
    融合全局 FullAttention 与 局部 LocalDetailAttention 的注意力：
    - 先分别计算 Full 与 Local 的权重并得到各自输出
    - 使用可学习的标量门控（sigmoid）进行线性融合：out = a * out_full + (1-a) * out_local
    - 兼容 SelfAttention_Family.AttentionLayer 的接口
    """

    def __init__(self, full_attention_cls, mask_flag: bool = False, window_size: int = 8,
                 attention_dropout: float = 0.1, output_attention: bool = False,
                 only_target_first: bool = True):
        super().__init__()
        # FullAttention 的实例（传入类以避免循环依赖）
        self.full = full_attention_cls(mask_flag=mask_flag, attention_dropout=attention_dropout,
                                       output_attention=output_attention)
        self.local = LocalDetailAttention(mask_flag=mask_flag, window_size=window_size,
                                          attention_dropout=attention_dropout, output_attention=output_attention)
        # 可学习融合门控（初始化偏向 0.5）
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.output_attention = output_attention
        self.only_target_first = only_target_first

    def forward(self, queries, keys, values, attn_mask=None):
        out_full, attn_full = self.full(queries, keys, values, attn_mask)
        out_local, attn_local = self.local(queries, keys, values, attn_mask)
        w = torch.sigmoid(self.alpha)
        # 仅对第一个位置（目标变量）应用局部融合，其余位置使用 FullAttention
        if self.only_target_first:
            B, L, H, E = out_full.shape
            gate_full = out_full.new_ones((L,))
            gate_local = out_full.new_zeros((L,))
            gate_full[0] = w
            gate_local[0] = (1.0 - w)
            gate_full = gate_full.view(1, L, 1, 1)
            gate_local = gate_local.view(1, L, 1, 1)
            out = gate_full * out_full + gate_local * out_local
        else:
            out = w * out_full + (1.0 - w) * out_local
        if self.output_attention:
            # 简单返回加权后的注意力（同权融合）用于可视化；也可返回二者及权重
            if attn_full is not None and attn_local is not None:
                if self.only_target_first:
                    # 仅第一个位置使用融合，其余返回 full 的注意力
                    B, Hh, Lq, Sk = attn_full.shape
                    mask = attn_full.new_zeros((Lq,))
                    mask[0] = 1.0
                    mask = mask.view(1, 1, Lq, 1)
                    attn = mask * (0.5 * (attn_full + attn_local)) + (1.0 - mask) * attn_full
                else:
                    attn = 0.5 * (attn_full + attn_local)
            else:
                attn = None
            return (out.contiguous(), attn)
        else:
            return (out.contiguous(), None)


