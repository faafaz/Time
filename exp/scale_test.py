import numpy as np
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("统一标准化 vs 分离标准化测试")
print("=" * 50)

# 1. 准备简单的测试数据
print("\n1. 原始数据:")
radiation = np.array([100, 120, 90, 110]).reshape(-1, 1)
power = np.array([50, 60, 45, 55]).reshape(-1, 1)

# 组合成完整特征矩阵
all_features = np.column_stack([radiation.flatten(), power.flatten()])

print("辐射量:", radiation.flatten())
print("功率:", power.flatten())
print("完整特征矩阵 shape:", all_features.shape)
print("完整特征矩阵:")
print(all_features)

# 2. 统一标准化方法
print("\n" + "=" * 30)
print("方法1: 统一标准化")
print("=" * 30)

unified_scaler = StandardScaler()
unified_scaler.fit(all_features)

print("统一scaler的均值:", unified_scaler.mean_)
print("统一scaler的标准差:", unified_scaler.scale_)

# 标准化所有特征
all_features_scaled = unified_scaler.transform(all_features)
print("\n标准化后的完整特征:")
print(all_features_scaled)

# 取出功率列（索引1）
power_scaled_unified = all_features_scaled[:, 1].reshape(-1, 1)
print("\n统一标准化的功率值:")
print(power_scaled_unified.flatten())

# 3. 分离标准化方法
print("\n" + "=" * 30)
print("方法2: 分离标准化")
print("=" * 30)

# 为输入特征创建scaler
feature_scaler = StandardScaler()
feature_scaler.fit(all_features)

# 为目标变量创建独立的scaler
target_scaler = StandardScaler()
target_scaler.fit(power)

print("特征scaler的均值:", feature_scaler.mean_)
print("特征scaler的标准差:", feature_scaler.scale_)
print("目标scaler的均值:", target_scaler.mean_)
print("目标scaler的标准差:", target_scaler.scale_)

# 分别标准化
features_scaled = feature_scaler.transform(all_features)
power_scaled_separate = target_scaler.transform(power)

print("\n分离标准化的功率值:")
print(power_scaled_separate.flatten())

# 4. 对比结果
print("\n" + "=" * 30)
print("结果对比")
print("=" * 30)
print("原始功率值:      ", power.flatten())
print("统一标准化功率:  ", power_scaled_unified.flatten().round(3))
print("分离标准化功率:  ", power_scaled_separate.flatten().round(3))

# 5. 反标准化测试
print("\n" + "=" * 30)
print("反标准化测试")
print("=" * 30)

# 模拟预测值
mock_predictions = np.array([[-0.5], [0.5], [-1.0], [1.0]])
print("模拟预测值（标准化）:", mock_predictions.flatten())

# 统一标准化的反标准化（需要构造完整特征矩阵）
dummy_features = np.zeros((4, 2))  # 4个样本，2个特征
dummy_features[:, 1] = mock_predictions.flatten()  # 把预测值放在功率列（索引1）
predictions_unified = unified_scaler.inverse_transform(dummy_features)[:, 1]

# 分离标准化的反标准化（直接用目标scaler）
predictions_separate = target_scaler.inverse_transform(mock_predictions).flatten()

print("统一标准化反标准化结果:", predictions_unified.round(1))
print("分离标准化反标准化结果:", predictions_separate.round(1))

# 6. 总结
print("\n" + "=" * 30)
print("总结")
print("=" * 30)
print("✅ 分离标准化优势:")
print("   - 反标准化简单: target_scaler.inverse_transform(predictions)")
print("   - 目标变量有独立的统计信息")
print("   - 代码逻辑清晰")

print("\n⚠️  统一标准化问题:")
print("   - 反标准化复杂: 需要构造虚拟特征矩阵")
print("   - 目标变量统计信息受其他特征影响")
print("   - 代码容易出错")

print(f"\n🔍 关键发现:")
print(f"   - 两种方法的标准化结果{'相同' if np.allclose(power_scaled_unified.flatten(), power_scaled_separate.flatten()) else '不同'}")
print(f"   - 反标准化结果{'相同' if np.allclose(predictions_unified, predictions_separate) else '不同'}")

if not np.allclose(power_scaled_unified.flatten(), power_scaled_separate.flatten()):
    print(f"   - 差异原因: 使用了不同的统计信息进行标准化")