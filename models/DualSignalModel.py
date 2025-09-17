import torch
import torch.nn as nn
from utils.FAN import main_freq_part
from models.iTransformer import Model as iTransformer


class Model(nn.Module):
    """双信号分离模型: 将输入信号分离为主频率成分和残差信号, 分别用iTransformer建模, 最后加权融合"""

    def __init__(self, configs, freq_topk=20, model_name="iTransformer"):
        super(Model, self).__init__()
        self.configs = configs
        self.freq_topk = freq_topk
        self.model_name = model_name

        # 信号分离参数
        self.rfft = True

        # 创建两个iTransformer模型
        self.main_freq_model = self._create_model()
        self.residual_model = self._create_model()

        # 可学习的权重参数
        self.weight_main = nn.Parameter(torch.tensor(0.5))
        self.weight_residual = nn.Parameter(torch.tensor(0.5))

    def _create_model(self):
        """根据模型名称创建对应的模型"""
        if self.model_name == "iTransformer":
            return iTransformer(self.configs)
        else:
            # 这里可以扩展支持其他模型
            raise ValueError(f"不支持的模型类型: {self.model_name}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播
        Args:
            x_enc: 输入序列 [batch_size, seq_len, n_vars]
            x_mark_enc: 编码器时间标记
            x_dec: 解码器输入
            x_mark_dec: 解码器时间标记
        Returns:
            融合后的预测结果 [batch_size, pred_len, n_vars]
        """
        batch_size, seq_len, n_vars = x_enc.shape

        # 1. 信号分离: 提取主频率成分和残差信号
        residual, main_freq = main_freq_part(x_enc, self.freq_topk, self.rfft)

        # 2. 分别用两个模型进行预测
        # 主频率成分预测 - 添加小噪声避免标准差为0
        main_freq_safe = main_freq + torch.randn_like(main_freq) * 1e-6
        pred_main = self.main_freq_model(main_freq_safe, x_mark_enc, x_dec, x_mark_dec, mask)

        # 残差信号预测 - 添加小噪声避免标准差为0
        residual_safe = residual + torch.randn_like(residual) * 1e-6
        pred_residual = self.residual_model(residual_safe, x_mark_enc, x_dec, x_mark_dec, mask)

        # 3. 加权融合
        weight_main = torch.sigmoid(self.weight_main)
        weight_residual = torch.sigmoid(self.weight_residual)

        # 归一化权重
        total_weight = weight_main + weight_residual
        weight_main = weight_main / total_weight
        weight_residual = weight_residual / total_weight

        # 融合预测结果
        fused_pred = weight_main * pred_main + weight_residual * pred_residual

        return fused_pred

    def get_model_info(self):
        """获取模型信息"""
        return {
            "model_type": "DualSignalModel",
            "base_model": self.model_name,
            "freq_topk": self.freq_topk,
            "main_weight": torch.sigmoid(self.weight_main).item(),
            "residual_weight": torch.sigmoid(self.weight_residual).item()
        }