import torch
import torch.nn as nn


# 原始实现：多留一位的方式
class OriginalTemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # 多留一位的设计
        hour_size = 24  # 0-23，刚好对应
        weekday_size = 7  # 0-6，刚好对应
        day_size = 32  # 1-31，多留一位（索引0不用）
        month_size = 13  # 1-12，多留一位（索引0不用）

        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)

    def forward(self, x):
        x = x.long()

        # 直接使用原始值作为索引
        hour_x = self.hour_embed(x[:, :, 3])  # 0-23
        weekday_x = self.weekday_embed(x[:, :, 2])  # 0-6
        day_x = self.day_embed(x[:, :, 1])  # 1-31（直接用）
        month_x = self.month_embed(x[:, :, 0])  # 1-12（直接用）

        return hour_x + weekday_x + day_x + month_x

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


# 优化实现：索引转换的方式
class OptimizedTemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # 精确对应实际范围
        hour_size = 24  # 0-23
        weekday_size = 7  # 0-6
        day_size = 31  # 1-31 -> 0-30（节省一位）
        month_size = 12  # 1-12 -> 0-11（节省一位）

        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)

    def forward(self, x):
        x = x.long()

        # 需要手动转换索引
        hour_x = self.hour_embed(x[:, :, 3])  # 0-23（不变）
        weekday_x = self.weekday_embed(x[:, :, 2])  # 0-6（不变）
        day_x = self.day_embed(x[:, :, 1] - 1)  # 1-31 -> 0-30
        month_x = self.month_embed(x[:, :, 0] - 1)  # 1-12 -> 0-11

        return hour_x + weekday_x + day_x + month_x

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


def test_temporal_embeddings():
    """测试和对比两种实现方式"""

    d_model = 64

    # 创建两个模型
    original_model = OriginalTemporalEmbedding(d_model)
    optimized_model = OptimizedTemporalEmbedding(d_model)

    # 测试数据：[month, day, weekday, hour]
    # 3月15日周二14:00, 4月1日周五8:00
    test_data = torch.tensor([
        [[3, 15, 2, 14],  # 3月15日周二14:00
         [4, 1, 5, 8]],  # 4月1日周五8:00
        [[12, 31, 0, 23],  # 12月31日周日23:00
         [1, 1, 1, 0]]  # 1月1日周一0:00
    ])

    print("=== 输入数据 ===")
    print(f"数据形状: {test_data.shape}")
    print(f"测试数据:\n{test_data}")
    print()

    # 前向传播
    print("=== 前向传播测试 ===")

    with torch.no_grad():
        original_output = original_model(test_data)
        optimized_output = optimized_model(test_data)

    print(f"原始模型输出形状: {original_output.shape}")
    print(f"优化模型输出形状: {optimized_output.shape}")
    print()

    # 参数量对比
    print("=== 参数量对比 ===")
    original_params = original_model.get_total_params()
    optimized_params = optimized_model.get_total_params()

    print(f"原始模型参数量: {original_params:,}")
    print(f"优化模型参数量: {optimized_params:,}")
    print(
        f"节省参数: {original_params - optimized_params} ({(original_params - optimized_params) / original_params * 100:.1f}%)")
    print()

    # 详细的嵌入层参数对比
    print("=== 各嵌入层参数详情 ===")

    def print_embedding_info(model, name):
        print(f"\n{name}:")
        print(f"  month_embed: {model.month_embed.weight.shape} = {model.month_embed.weight.numel()} 参数")
        print(f"  day_embed:   {model.day_embed.weight.shape} = {model.day_embed.weight.numel()} 参数")
        print(f"  weekday_embed: {model.weekday_embed.weight.shape} = {model.weekday_embed.weight.numel()} 参数")
        print(f"  hour_embed:  {model.hour_embed.weight.shape} = {model.hour_embed.weight.numel()} 参数")

    print_embedding_info(original_model, "原始模型")
    print_embedding_info(optimized_model, "优化模型")

    # 测试边界情况
    print("\n=== 边界情况测试 ===")

    # 测试最小值
    min_data = torch.tensor([[[1, 1, 0, 0]]])  # 1月1日周日0:00
    print("测试最小值 (1月1日周日0:00):")

    try:
        original_min = original_model(min_data)
        print("  原始模型: ✓ 成功")
    except Exception as e:
        print(f"  原始模型: ✗ 错误 {e}")

    try:
        optimized_min = optimized_model(min_data)
        print("  优化模型: ✓ 成功")
    except Exception as e:
        print(f"  优化模型: ✗ 错误 {e}")

    # 测试最大值
    max_data = torch.tensor([[[12, 31, 6, 23]]])  # 12月31日周六23:00
    print("\n测试最大值 (12月31日周六23:00):")

    try:
        original_max = original_model(max_data)
        print("  原始模型: ✓ 成功")
    except Exception as e:
        print(f"  原始模型: ✗ 错误 {e}")

    try:
        optimized_max = optimized_model(max_data)
        print("  优化模型: ✓ 成功")
    except Exception as e:
        print(f"  优化模型: ✗ 错误 {e}")

    print("\n=== 检查未使用的参数 ===")

    # 检查原始模型中未使用的参数
    with torch.no_grad():
        print("原始模型中索引0的参数（永远不会被使用）:")
        print(f"  month_embed[0]: {original_model.month_embed.weight[0][:5]}...")  # 显示前5个值
        print(f"  day_embed[0]:   {original_model.day_embed.weight[0][:5]}...")

        print("优化模型中所有参数都会被使用")


if __name__ == "__main__":
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)

    test_temporal_embeddings()