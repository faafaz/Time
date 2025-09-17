import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from database.mysql_dataloader import query_solar_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def extract_time_features(df):
    """第一步：提取时间特征（已完成）"""
    df = df.copy()
    df['OBSERVETIME'] = pd.to_datetime(df['OBSERVETIME'])
    df['day_of_year'] = df['OBSERVETIME'].dt.dayofyear
    df['time_of_day'] = (df['OBSERVETIME'].dt.hour * 60 +
                         df['OBSERVETIME'].dt.minute)
    return df


def gaussian_kernel_1d(x, xi, bandwidth):
    """第二步：一维高斯核函数（已完成）"""
    distance = np.abs(x - xi)
    weights = np.exp(-0.5 * (distance / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
    return weights


def time_of_day_distance(target_time, time_array):
    """处理time_of_day的周期性距离（已完成）"""
    linear_distance = np.abs(time_array - target_time)
    circular_distance = np.minimum(linear_distance, 1440 - linear_distance)
    return circular_distance


def gaussian_kernel_time_of_day(target_time, time_array, bandwidth):
    """针对time_of_day的特殊高斯核函数（已完成）"""
    distances = time_of_day_distance(target_time, time_array)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
    return weights


def compute_2d_weights(target_day, target_time, df, bandwidth_day=50, bandwidth_time=45):
    """计算二维权重（已完成）"""
    weights_day = gaussian_kernel_1d(target_day, df['day_of_year'].values, bandwidth_day)
    weights_time = gaussian_kernel_time_of_day(target_time, df['time_of_day'].values, bandwidth_time)
    weights_2d = weights_day * weights_time

    if weights_2d.sum() > 0:
        weights_2d = weights_2d / weights_2d.sum()

    return weights_2d, weights_day, weights_time


def weighted_quantile(values, weights, quantile_level):
    """
    第三步：加权分位数计算

    这是Clear Sky Model的核心算法！
    使用加权分位数回归找到指定分位数

    论文中使用0.85分位数来估计晴空功率，因为：
    - 晴天时功率接近最大值
    - 0.85分位数能很好地代表"接近晴天"的条件

    Parameters:
    - values: 功率值数组
    - weights: 对应的权重数组
    - quantile_level: 分位数水平 (0-1)，论文中使用0.85

    Returns:
    - quantile_value: 加权分位数值
    """

    # 检查输入
    if len(values) != len(weights):
        raise ValueError("values和weights长度必须相同")

    if len(values) == 0:
        return 0.0

    # 移除权重为0的点
    valid_mask = weights > 0
    if not np.any(valid_mask):
        return np.median(values)

    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]

    # 按值排序
    sorted_indices = np.argsort(valid_values)
    sorted_values = valid_values[sorted_indices]
    sorted_weights = valid_weights[sorted_indices]

    # 计算累积权重
    cumsum_weights = np.cumsum(sorted_weights)
    total_weight = cumsum_weights[-1]

    # 归一化累积权重到[0,1]
    normalized_cumsum = cumsum_weights / total_weight

    # 找到分位数对应的位置
    target_weight = quantile_level

    # 查找插入位置
    idx = np.searchsorted(normalized_cumsum, target_weight, side='right')

    if idx == 0:
        # 在最小值之前
        return sorted_values[0]
    elif idx >= len(sorted_values):
        # 在最大值之后
        return sorted_values[-1]
    else:
        # 线性插值
        w1 = normalized_cumsum[idx - 1]
        w2 = normalized_cumsum[idx]
        v1 = sorted_values[idx - 1]
        v2 = sorted_values[idx]

        if w2 == w1:  # 避免除零
            return v1

        # 插值权重
        alpha = (target_weight - w1) / (w2 - w1)
        quantile_value = v1 + alpha * (v2 - v1)

        return quantile_value


def estimate_clear_sky_power_single_point(target_day, target_time, df,
                                          quantile_level=0.85,
                                          bandwidth_day=50,
                                          bandwidth_time=45):
    """
    为单个时间点估计晴空功率

    这是论文公式(7)的实现: p_cs = f_max(x, y)
    其中f_max通过加权分位数回归得到

    Parameters:
    - target_day: 目标day_of_year
    - target_time: 目标time_of_day
    - df: 包含ACTIVEPOWER和时间特征的DataFrame
    - quantile_level: 分位数水平，论文建议0.85

    Returns:
    - clear_sky_power: 估计的晴空功率
    - debug_info: 调试信息字典
    """

    # 计算权重
    weights_2d, weights_day, weights_time = compute_2d_weights(
        target_day, target_time, df, bandwidth_day, bandwidth_time
    )

    # 计算加权分位数
    clear_sky_power = weighted_quantile(
        df['ACTIVEPOWER'].values,
        weights_2d,
        quantile_level
    )

    # 收集调试信息
    debug_info = {
        'target_day': target_day,
        'target_time': target_time,
        'target_time_str': f"{target_time // 60:02d}:{target_time % 60:02d}",
        'weights_sum': weights_2d.sum(),
        'weights_max': weights_2d.max(),
        'effective_points': (weights_2d > 1e-6).sum(),
        'clear_sky_power': clear_sky_power,
        'power_range': (df['ACTIVEPOWER'].min(), df['ACTIVEPOWER'].max()),
        'quantile_level': quantile_level
    }

    return clear_sky_power, debug_info


def test_weighted_quantile():
    """
    测试加权分位数计算函数
    """
    print("=== 测试加权分位数计算 ===")

    # 测试用例1：均匀权重，应该等于普通分位数
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    weights = np.ones(len(values))  # 均匀权重

    q50_weighted = weighted_quantile(values, weights, 0.5)
    q50_numpy = np.percentile(values, 50)

    print(f"测试1 - 均匀权重:")
    print(f"  加权中位数: {q50_weighted:.2f}")
    print(f"  numpy中位数: {q50_numpy:.2f}")
    print(f"  差异: {abs(q50_weighted - q50_numpy):.6f}")

    # 测试用例2：集中权重
    weights = np.array([0.1, 0.1, 0.1, 5.0, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1])  # 中间权重大
    q50_weighted = weighted_quantile(values, weights, 0.5)

    print(f"\n测试2 - 集中权重:")
    print(f"  values: {values}")
    print(f"  weights: {weights}")
    print(f"  加权中位数: {q50_weighted:.2f} (应该接近4-5)")

    # 测试用例3：不同分位数
    weights = np.ones(len(values))
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95]

    print(f"\n测试3 - 不同分位数:")
    for q in quantiles:
        q_value = weighted_quantile(values, weights, q)
        print(f"  {q:.2f}分位数: {q_value:.2f}")


def visualize_clear_sky_estimation(df, target_day=180, target_time=720):
    """
    可视化晴空功率估计过程

    Parameters:
    - target_day: 目标日期
    - target_time: 目标时间（720 = 12:00）
    """

    print(f"\n=== 可视化晴空功率估计 ===")
    print(f"目标点: Day {target_day}, Time {target_time // 60:02d}:{target_time % 60:02d}")

    # 估计晴空功率
    clear_sky_power, debug_info = estimate_clear_sky_power_single_point(
        target_day, target_time, df
    )

    # 计算权重
    weights_2d, _, _ = compute_2d_weights(target_day, target_time, df)

    # 找到权重最高的点
    high_weight_mask = weights_2d > np.percentile(weights_2d[weights_2d > 0], 90)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 权重 vs 功率散点图
    scatter = axes[0, 0].scatter(df['ACTIVEPOWER'], weights_2d,
                                 c=df['ACTIVEPOWER'], cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].axvline(clear_sky_power, color='red', linestyle='--', linewidth=2,
                       label=f'估计晴空功率: {clear_sky_power:.0f}W')
    axes[0, 0].set_xlabel('功率 (W)')
    axes[0, 0].set_ylabel('权重')
    axes[0, 0].set_title('功率 vs 权重分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 高权重点的功率分布
    high_weight_powers = df['ACTIVEPOWER'][high_weight_mask]
    if len(high_weight_powers) > 0:
        axes[0, 1].hist(high_weight_powers, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(clear_sky_power, color='red', linestyle='--', linewidth=2,
                           label=f'0.85分位数: {clear_sky_power:.0f}W')
        axes[0, 1].set_xlabel('功率 (W)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('高权重点功率分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. 权重累积分布
    sorted_powers = np.sort(df['ACTIVEPOWER'][weights_2d > 0])
    sorted_weights = weights_2d[weights_2d > 0][np.argsort(df['ACTIVEPOWER'][weights_2d > 0])]
    cumsum_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)

    axes[1, 0].plot(sorted_powers, cumsum_weights, 'b-', linewidth=2)
    axes[1, 0].axhline(0.85, color='red', linestyle='--', alpha=0.7, label='0.85分位数线')
    axes[1, 0].axvline(clear_sky_power, color='red', linestyle='--', alpha=0.7,
                       label=f'估计值: {clear_sky_power:.0f}W')
    axes[1, 0].set_xlabel('功率 (W)')
    axes[1, 0].set_ylabel('累积权重')
    axes[1, 0].set_title('加权累积分布函数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 显示调试信息
    info_text = f"""目标点信息:
日期: 第{debug_info['target_day']}天
时间: {debug_info['target_time_str']}
分位数水平: {debug_info['quantile_level']}

权重统计:
权重总和: {debug_info['weights_sum']:.6f}
最大权重: {debug_info['weights_max']:.6f}
有效点数: {debug_info['effective_points']}

结果:
估计晴空功率: {debug_info['clear_sky_power']:.1f} W
功率范围: {debug_info['power_range'][0]:.0f} - {debug_info['power_range'][1]:.0f} W"""

    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('估计详情')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return clear_sky_power, debug_info


# 使用示例
if __name__ == "__main__":
    print("第三步：加权分位数计算实现完成！")
    print("\n使用方法：")
    print("1. test_weighted_quantile()  # 测试分位数函数")
    print("2. df_with_features = extract_time_features(your_dataframe)")
    print("3. clear_sky_power, info = visualize_clear_sky_estimation(df_with_features)")
    print("\n这一步实现了论文的核心算法：通过加权分位数估计晴空功率！")

    # 查询参数
    farmid = "1"  # 替换为实际的电站ID
    start_time = "2023-01-01 00:00:00"
    end_time = "2023-12-31 23:45:00"

    # 执行查询
    df = query_solar_data(farmid, start_time, end_time)

    test_weighted_quantile()
    df_with_features = extract_time_features(df)
    clear_sky_power, info = visualize_clear_sky_estimation(df_with_features)