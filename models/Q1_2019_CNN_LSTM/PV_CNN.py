import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    基于论文最佳配置(3Y)的CNN模型 - 适配15分钟分辨率数据
    论文: "A comparison of day-ahead photovoltaic power forecasting models based on deep learning neural network"
    """

    def __init__(self, configs, input_sequence_length=96, input_features=2, output_length=1):
        """
        初始化CNN模型 - 使用论文中效果最好的3Y配置

        Args:
            input_sequence_length: 输入序列长度（时间步数）
            input_features: 输入特征数量（默认2个特征：ACTIVEPOWER, TOTALRADIATION）
            output_length: 输出长度（预测的时间步数，默认1）
        """
        super(Model, self).__init__()

        self.input_sequence_length = input_sequence_length
        self.input_features = input_features
        self.output_length = output_length

        # 论文Table 4中3Y配置的参数
        filters = 4096
        kernel_size = 3
        stride = 2
        dropout_rate = 0.5

        # 激活函数（需要先定义，因为_calculate_conv_output_size会用到）
        self.relu = nn.ReLU()

        # 卷积层
        self.conv1 = nn.Conv1d(
            in_channels=input_features,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2  # 手动计算padding，替代'same'
        )

        # 池化层
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        # 计算卷积层输出的特征维度
        self.conv_output_size = self._calculate_conv_output_size()

        # 全连接层
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_length)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def _calculate_conv_output_size(self):
        """计算卷积层输出的特征维度"""
        with torch.no_grad():
            # 创建测试输入: (batch_size, sequence_length, input_features)
            x = torch.randn(1, self.input_sequence_length, self.input_features)
            # 转换为CNN需要的格式: (batch_size, input_features, sequence_length)
            x = x.transpose(1, 2)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return x.numel()

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_features)

        Returns:
            output: 预测结果，形状为 (batch_size, output_length, 1)
        """
        # 输入格式转换: (batch_size, sequence_length, input_features) -> (batch_size, input_features, sequence_length)
        x = x.transpose(1, 2)  # 交换维度1和维度2

        # 卷积层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 输出层
        output = self.fc4(x)

        # 输出格式调整: (batch_size, output_length) -> (batch_size, output_length, 1)
        output = output.unsqueeze(-1)  # 在最后一个维度添加一个维度

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'configuration': '3Y (best performance)',
            'filters': 4096,
            'kernel_size': 3,
            'stride': 2,
            'dropout': 0.5,
            'input_shape': (self.input_sequence_length, self.input_features),
            'output_shape': (self.output_length, 1)
        }
        return info


# 使用示例
if __name__ == "__main__":
    # 创建模型
    # 假设输入序列长度为96（24小时*4个15分钟间隔）
    model = Model(
        input_sequence_length=96,  # 24小时的15分钟数据: 24*60/15=96
        input_features=2,  # 您的2个特征：ACTIVEPOWER, TOTALRADIATION
        output_length=16  # 预测16个时间步
    )

    # 打印模型信息
    print("=== CNN模型信息 (3Y配置, 2特征, 15分钟数据) ===")
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

    print("\n=== 数据格式说明 ===")
    print("输入数据应该是形状为 (batch_size, sequence_length, 2) 的张量")
    print("输出数据形状为 (batch_size, output_length, 1)")
    print("其中:")
    print("- batch_size: 批次大小")
    print("- sequence_length: 时间序列长度（例如96表示24小时的15分钟数据）")
    print("- 2: 两个特征 [ACTIVEPOWER, TOTALRADIATION]")
    print("- output_length: 预测的时间步数")
    print("\n15分钟分辨率时间计算:")
    print("- 1小时 = 4个数据点")
    print("- 6小时 = 24个数据点")
    print("- 12小时 = 48个数据点")
    print("- 24小时 = 96个数据点")
    print("\n示例:")
    print("data = torch.tensor([")
    print("    [[power_t1, rad_t1],      # 时间步1的特征")
    print("     [power_t2, rad_t2],      # 时间步2的特征")
    print("     ...,")
    print("     [power_t96, rad_t96]]    # 时间步96的特征")
    print("])")
    print("# 数据形状: (1, 96, 2)")
    print("# 预测输出形状: (1, 1, 1)")
    print("\n注意: 通常ACTIVEPOWER作为预测目标，TOTALRADIATION作为辅助特征")

    print("\n=== 15分钟数据的常用序列长度 ===")
    print("根据预测需求选择合适的输入序列长度:")
    print("- 短期预测: 24-48个点 (6-12小时历史数据)")
    print("- 中期预测: 96个点 (24小时历史数据)")
    print("- 长期预测: 192-288个点 (2-3天历史数据)")
    print("注意: 序列长度需要根据您的具体数据量和预测目标调整")
