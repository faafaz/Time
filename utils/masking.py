import torch


class TriangularCausalMask:
    """
    三角因果掩码（Triangular Causal Mask）
    用于Transformer中防止未来信息泄露
    """
    def __init__(self, B, L, device="cpu"):
        """
        Args:
            B: batch size
            L: 序列长度
            device: 设备 (cpu/cuda)
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # 创建下三角掩码（对角线及以下为True）
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    
    @property
    def mask(self):
        return self._mask
