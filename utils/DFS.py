import torch
import torch.nn as nn


class DifferentiableFeatureSelector(nn.Module):
    def __init__(self, input_dim, temperature=1.0, init_value=-0.5):
        super(DifferentiableFeatureSelector, self).__init__()
        self.input_dim = input_dim
        self.temperature = temperature
        # 初始化掩码参数，初始值设为init_value
        self.mask_logits = nn.Parameter(torch.full((input_dim,), init_value))

    def forward(self, x, hard: bool = False):
        # x: [B, L, C]
        C = x.shape[-1]
        assert C == self.input_dim, f"DFS input_dim={self.input_dim} but got x with C={C}"
        if self.training:
            # 训练阶段：使用 Gumbel-Sigmoid 采样近似二值掩码
            uniform = torch.rand_like(self.mask_logits)
            gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
            noisy_logits = (self.mask_logits + gumbel) / max(self.temperature, 1e-6)
            mask = torch.sigmoid(noisy_logits)
        else:
            # 推理阶段：使用确定性掩码（无噪声）
            if hard:
                mask = (self.mask_logits > 0).float()
            else:
                mask = torch.sigmoid(self.mask_logits)
        # 广播到通道维 [C] -> [1,1,C]
        mask = mask.view(1, 1, C)
        return x * mask

    def get_feature_importance(self):
        return torch.sigmoid(self.mask_logits).detach().cpu().numpy()

    def get_selected_features(self, threshold=0.5):
        importance = self.get_feature_importance()

    def l1_regularization(self):
        """
        对掩码概率(sigmoid(logits))做 L1 惩罚，鼓励稀疏。
        返回一个标量张量。
        """
        probs = torch.sigmoid(self.mask_logits)
        return torch.sum(torch.abs(probs))

    def anneal(self, rate: float = 0.95, min_temperature: float = 0.1):
        """温度退火: T <- max(min_T, T * rate)"""
        self.temperature = float(max(min_temperature, self.temperature * rate))
        return None


        return [i for i, imp in enumerate(importance) if imp > threshold]