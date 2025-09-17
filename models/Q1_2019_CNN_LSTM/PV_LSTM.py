import torch
import torch.nn as nn


class Model(nn.Module):
    """
    基于论文最佳配置(3Y)的LSTM模型 - 适配15分钟分辨率数据
    论文: "A comparison of day-ahead photovoltaic power forecasting models based on deep learning neural network"
    """

    def __init__(self, input_sequence_length, input_features=2, output_length=1):
        """
        初始化LSTM模型 - 使用论文中效果最好的3Y配置

        Args:
            input_sequence_length: 输入序列长度（时间步数）
            input_features: 输入特征数量（默认2个特征：ACTIVEPOWER, TOTALRADIATION）
            output_length: 输出长度（预测的时间步数，默认1）
        """
        super(Model, self).__init__()

        self.input_sequence_length = input_sequence_length
        self.input_features = input_features
        self.output_length = output_length

        # 论文Table 4中3Y配置的LSTM参数
        hidden_units = [80, 300, 500]  # 3层LSTM的隐藏单元数
        dropout_rate = 0.5

        # 多层LSTM
        self.lstm_layers = nn.ModuleList()

        # 第一层LSTM
        self.lstm_layers.append(
            nn.LSTM(input_features, hidden_units[0], batch_first=True, dropout=dropout_rate)
        )

        # 后续LSTM层
        for i in range(1, len(hidden_units)):
            self.lstm_layers.append(
                nn.LSTM(hidden_units[i - 1], hidden_units[i], batch_first=True, dropout=dropout_rate)
            )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_units[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_length)
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_features)

        Returns:
            output: 预测结果，形状为 (batch_size, output_length, 16)
        """
        # 通过多层LSTM
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)

        # 取最后一个时间步的输出
        x = x[:, -1, :]  # (batch_size, hidden_units[-1])

        # 全连接层
        output = self.fc_layers(x)  # (batch_size, output_length)

        # 输出格式调整: (batch_size, output_length) -> (batch_size, output_length, 1)
        output = output.unsqueeze(-1)

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'configuration': '3Y (best performance)',
            'lstm_units': [80, 300, 500],
            'dropout': 0.5,
            'input_shape': (self.input_sequence_length, self.input_features),
            'output_shape': (self.output_length, 16)
        }
        return info


# 使用示例
if __name__ == "__main__":
    # 创建模型
    # 假设输入序列长度为96（24小时*4个15分钟间隔）
    model = PV_LSTM_Model(
        input_sequence_length=96,  # 24小时的15分钟数据: 24*60/15=96
        input_features=2,  # 您的2个特征：ACTIVEPOWER, TOTALRADIATION
        output_length=16  # 预测1个时间步
    )

    # 打印模型信息
    print("=== LSTM模型信息 (3Y配置, 2特征, 15分钟数据) ===")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    # 测试前向传播
    print("\n=== 测试前向传播 ===")
    batch_size = 32
    test_input = torch.randn(batch_size, 96, 2)  # (batch_size, sequence_length, features)

    with torch.no_grad():
        output = model(test_input)
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print(f"\n模型总参数量: {info['total_parameters']:,}")
    print("\n特征说明:")
    print("- 特征1: ACTIVEPOWER (有功功率)")
    print("- 特征2: TOTALRADIATION (总辐射)")

    print("\n=== LSTM模型特点 ===")
    print("- 3层LSTM架构: 80 -> 300 -> 500 隐藏单元")
    print("- 适合捕捉时间序列的长期依赖关系")
    print("- 相比CNN，更专注于时间特征提取")
    print("- 训练时间相对较短（论文中提到）")
