import torch
import torch.nn as nn


class SEGateExogenous(nn.Module):
    """SE门控外生特征模块"""

    def __init__(self, reduction: int = 4, channels: int = 12, device: torch.device = None):
        """
        初始化SE门控外生特征模块

        Args:
            reduction: SE模块的压缩比例
        """
        super().__init__()
        self.reduction = reduction

        """初始化SE门控网络"""
        if channels <= 0:
            return

        hidden = max(1, channels // self.reduction)
        self.se_gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

        # 如果指定了设备，将网络移动到该设备
        if device is not None:
            self.se_gate = self.se_gate.to(device)

    def forward(self, x_target: torch.Tensor, x_exogenous: torch.Tensor = None):
        """
        前向传播，应用SE门控到外生特征

        Args:
            x_target: 目标特征张量 [B, L, 1]
            x_exogenous: 外生特征张量 [B, L, C_exo] 或 None

        Returns:
            x_combined: 组合后的特征张量 [B, L, 1 + C_exo]
        """
        # 如果没有外生特征，直接返回目标特征
        if x_exogenous is None or x_exogenous.shape[-1] == 0:
            return x_target

        # 应用SE门控
        if self.se_gate is not None:
            # 计算通道注意力权重
            s = x_exogenous.mean(dim=1)                    # [B, C_exo]
            w = self.se_gate(s)                            # [B, C_exo]
            # 应用权重到外生特征
            x_exogenous_gated = x_exogenous * w.view(x_exogenous.shape[0], 1, -1)  # [B, L, C_exo]
        else:
            x_exogenous_gated = x_exogenous

        # 组合目标特征和外生特征
        x_combined = torch.cat([x_target, x_exogenous_gated], dim=2)  # [B, L, 1 + C_exo]

        return x_combined

    def split_features(self, x_combined: torch.Tensor, target_channel_index: int = 0):
        """
        从组合特征中分离目标特征和外生特征

        Args:
            x_combined: 组合特征张量 [B, L, C_total]
            target_channel_index: 目标特征通道索引

        Returns:
            x_target: 目标特征张量 [B, L, 1]
            x_exogenous: 外生特征张量 [B, L, C_exo] 或 None
        """
        batch_size, seq_len, num_channels = x_combined.shape

        if num_channels <= 1:
            return x_combined, None

        # 确保目标通道索引有效
        target_idx = max(0, min(target_channel_index, num_channels - 1))

        # 提取目标特征
        x_target = x_combined[:, :, target_idx:target_idx + 1]  # [B, L, 1]

        # 提取外生特征
        if num_channels > 1:
            left_channels = x_combined[:, :, :target_idx]
            right_channels = x_combined[:, :, target_idx + 1:]
            x_exogenous = torch.cat([left_channels, right_channels], dim=2) if left_channels.shape[2] + right_channels.shape[2] > 0 else None
        else:
            x_exogenous = None

        return x_target, x_exogenous