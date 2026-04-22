import torch
import torch.nn as nn


class DifferentiableFeatureSelector(nn.Module):
    def __init__(self, input_dim, temperature=1.0, init_value=1.0):
        super(DifferentiableFeatureSelector, self).__init__()
        self.input_dim = input_dim
        self.temperature = temperature
        # 初始化掩码参数，初始值设为init_value
        self.mask_logits = nn.Parameter(torch.full((input_dim,), init_value))
        
    def forward(self, x, hard=False):
        # 训练时使用Gumbel-Softmax松弛，推理时可以使用硬掩码
        # if self.run_type=="0":
            # 使用Gumbel-Softmax技巧获得近似二值的掩码
        uniform = torch.rand_like(self.mask_logits)
        gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
        noisy_logits = (self.mask_logits + gumbel) / self.temperature
        mask = torch.sigmoid(noisy_logits)
        # else:
        #     if hard:
        #         # 在推理时，我们可以使用硬阈值来得到二值掩码
        #         mask = (self.mask_logits > 0).float()
        #     else:
        #         mask = torch.sigmoid(self.mask_logits)
        
        # 应用特征选择掩码
        masked_x = x * mask
        return masked_x

    def get_feature_importance(self):
        return torch.sigmoid(self.mask_logits).detach().cpu().numpy()

    def get_selected_features(self, threshold=0.5):
        importance = self.get_feature_importance()
        return [i for i, imp in enumerate(importance) if imp > threshold]