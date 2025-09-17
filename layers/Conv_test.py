import torch
import torch.nn as nn

print("=== 卷积核参数量固定性验证 ===")

# 定义Inception_Block_V2中的几种卷积核
in_channels = 64
out_channels = 128

# V2中的不同卷积核
convs = {
    "1x3": nn.Conv2d(in_channels, out_channels, kernel_size=[1, 3], padding=[0, 1]),
    "3x1": nn.Conv2d(in_channels, out_channels, kernel_size=[3, 1], padding=[1, 0]),
    "1x5": nn.Conv2d(in_channels, out_channels, kernel_size=[1, 5], padding=[0, 2]),
    "5x1": nn.Conv2d(in_channels, out_channels, kernel_size=[5, 1], padding=[2, 0]),
    "1x1": nn.Conv2d(in_channels, out_channels, kernel_size=1)
}

print("🔧 各种卷积核的参数量:")
total_params = 0
for name, conv in convs.items():
    params = sum(p.numel() for p in conv.parameters())
    total_params += params
    print(f"{name}卷积: {params:,} 参数")

print(f"\n总参数量: {total_params:,}")

print("\n" + "=" * 60)
print("📏 测试不同序列长度的输入")

# 测试不同长度的时序数据
test_sizes = [
    (32, 32),  # 短序列
    (64, 64),  # 中等序列
    (128, 128),  # 长序列
    (256, 256)  # 很长序列
]

print("\n对于1×3卷积核:")
conv_1x3 = convs["1x3"]
original_params = sum(p.numel() for p in conv_1x3.parameters())

for h, w in test_sizes:
    # 创建不同大小的输入
    x = torch.randn(1, in_channels, h, w)

    # 前向传播
    output = conv_1x3(x)

    # 检查参数量
    current_params = sum(p.numel() for p in conv_1x3.parameters())

    print(f"  输入{h}×{w} → 输出{output.shape[2]}×{output.shape[3]}, 参数量: {current_params:,}")

print(f"\n✅ 结论: 无论输入多大，参数量始终是 {original_params:,}")

print("\n" + "=" * 60)
print("🧮 卷积参数量计算公式")

print("\n卷积层参数量 = 输入通道数 × 输出通道数 × 卷积核高 × 卷积核宽 + 偏置项")
print("\n各卷积核详细计算:")

for name, conv in convs.items():
    kernel_size = conv.kernel_size
    if isinstance(kernel_size, int):
        k_h = k_w = kernel_size
    else:
        k_h, k_w = kernel_size

    weight_params = in_channels * out_channels * k_h * k_w
    bias_params = out_channels if conv.bias is not None else 0
    total = weight_params + bias_params

    print(f"{name}: {in_channels}×{out_channels}×{k_h}×{k_w} + {bias_params} = {total:,}")

print("\n" + "=" * 60)
print("💡 关键理解")

print("\n🔒 参数量只取决于:")
print("- ✅ 输入通道数 (in_channels)")
print("- ✅ 输出通道数 (out_channels)")
print("- ✅ 卷积核大小 (kernel_size)")

print("\n🚫 参数量与这些无关:")
print("- ❌ 输入的高度 (序列长度)")
print("- ❌ 输入的宽度 (时间维度)")
print("- ❌ 批次大小 (batch_size)")

print("\n🎯 实际意义:")
print("- 模型大小固定，不随数据长度变化")
print("- 可以处理任意长度的时序数据")
print("- 内存需求主要来自激活值，不是参数")

print("\n⚡ 为什么这样设计?")
print("- 权重共享: 所有时间位置用相同规律")
print("- 参数效率: 避免参数量随序列长度爆炸")
print("- 泛化能力: 学到的模式可应用于不同长度序列")

print("\n" + "=" * 60)
print("📊 与全连接层对比")

seq_lengths = [100, 500, 1000]
print("\n如果用全连接层处理不同长度序列:")

for seq_len in seq_lengths:
    fc_params = seq_len * in_channels * out_channels
    conv_params = original_params  # 卷积参数固定

    print(f"序列长度{seq_len}: 全连接{fc_params:,}参数 vs 卷积{conv_params:,}参数")

print("\n💥 全连接层参数会随序列长度线性增长!")
print("🌟 卷积层参数始终固定!")