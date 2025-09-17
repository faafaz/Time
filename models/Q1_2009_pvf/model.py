import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条
import time

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
    """第三步：加权分位数计算（已完成）"""
    if len(values) != len(weights):
        raise ValueError("values和weights长度必须相同")

    if len(values) == 0:
        return 0.0

    valid_mask = weights > 0
    if not np.any(valid_mask):
        return np.median(values)

    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]

    sorted_indices = np.argsort(valid_values)
    sorted_values = valid_values[sorted_indices]
    sorted_weights = valid_weights[sorted_indices]

    cumsum_weights = np.cumsum(sorted_weights)
    total_weight = cumsum_weights[-1]
    normalized_cumsum = cumsum_weights / total_weight

    target_weight = quantile_level
    idx = np.searchsorted(normalized_cumsum, target_weight, side='right')

    if idx == 0:
        return sorted_values[0]
    elif idx >= len(sorted_values):
        return sorted_values[-1]
    else:
        w1 = normalized_cumsum[idx - 1]
        w2 = normalized_cumsum[idx]
        v1 = sorted_values[idx - 1]
        v2 = sorted_values[idx]

        if w2 == w1:
            return v1

        alpha = (target_weight - w1) / (w2 - w1)
        quantile_value = v1 + alpha * (v2 - v1)

        return quantile_value


def estimate_clear_sky_power_batch(df, quantile_level=0.85, bandwidth_day=50, bandwidth_time=45,
                                   progress_interval=1000):
    """
    第四步：批量估计所有时间点的晴空功率

    这是论文公式(8)的实现: {p̂_cs_t, t = 1, ..., N}
    为每个观测时间点估计对应的晴空功率

    Parameters:
    - df: 包含时间特征的DataFrame
    - quantile_level: 分位数水平，论文建议0.85
    - bandwidth_day/bandwidth_time: 带宽参数
    - progress_interval: 进度显示间隔

    Returns:
    - clear_sky_powers: 晴空功率数组
    """

    print("=== 第四步：批量估计晴空功率 ===")
    print(f"数据点数量: {len(df)}")
    print(f"分位数水平: {quantile_level}")
    print(f"带宽参数: day={bandwidth_day}, time={bandwidth_time}")

    clear_sky_powers = np.zeros(len(df))
    start_time = time.time()

    # 使用tqdm显示进度条
    for i in tqdm(range(len(df)), desc="估计晴空功率"):
        target_day = df.iloc[i]['day_of_year']
        target_time = df.iloc[i]['time_of_day']

        # 计算二维权重
        weights_2d, _, _ = compute_2d_weights(
            target_day, target_time, df, bandwidth_day, bandwidth_time
        )

        # 计算加权分位数
        clear_sky_power = weighted_quantile(
            df['ACTIVEPOWER'].values,
            weights_2d,
            quantile_level
        )

        clear_sky_powers[i] = clear_sky_power

        # 定期显示进度信息
        if (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(df) - i - 1) / rate
            print(f"已完成 {i + 1}/{len(df)}, "
                  f"速度: {rate:.1f} 点/秒, "
                  f"预计剩余: {remaining / 60:.1f} 分钟")

    total_time = time.time() - start_time
    print(f"晴空功率估计完成，耗时: {total_time / 60:.1f} 分钟")

    return clear_sky_powers


def compute_normalized_power(df, clear_sky_powers, power_threshold=0.2):
    """
    计算归一化功率并应用过滤条件

    这是论文公式(9)和(11)的实现:
    - τ_t = p_t / p̂_cs_t  (公式9)
    - 移除 p̂_cs_t / max(p̂_cs_t) < 0.2 的点  (公式11)

    Parameters:
    - df: 包含ACTIVEPOWER的DataFrame
    - clear_sky_powers: 晴空功率数组
    - power_threshold: 功率阈值，论文建议0.2

    Returns:
    - df_processed: 处理后的DataFrame
    - removed_info: 移除点的信息
    """

    print("\n=== 计算归一化功率 ===")

    df_result = df.copy()
    df_result['clear_sky_power'] = clear_sky_powers

    # 计算归一化功率 τ = p / p_cs
    normalized_power = df_result['ACTIVEPOWER'] / df_result['clear_sky_power']
    df_result['normalized_power'] = normalized_power

    # 应用论文中的过滤条件：移除低辐射时段
    # 公式(11): 移除 p̂_cs_t / max(p̂_cs_t) < 0.2 的点
    max_clear_sky = df_result['clear_sky_power'].max()
    relative_clear_sky = df_result['clear_sky_power'] / max_clear_sky
    valid_mask = relative_clear_sky >= power_threshold

    # 统计移除的点
    removed_count = (~valid_mask).sum()
    removed_percentage = removed_count / len(df_result) * 100

    removed_info = {
        'total_points': len(df_result),
        'removed_points': removed_count,
        'removed_percentage': removed_percentage,
        'valid_points': valid_mask.sum(),
        'power_threshold': power_threshold,
        'max_clear_sky_power': max_clear_sky
    }

    print(f"过滤统计:")
    print(f"  总数据点: {removed_info['total_points']}")
    print(f"  移除点数: {removed_info['removed_points']} ({removed_info['removed_percentage']:.1f}%)")
    print(f"  有效点数: {removed_info['valid_points']}")
    print(f"  功率阈值: {power_threshold} (相对于最大晴空功率)")

    # 只保留有效点
    df_processed = df_result[valid_mask].copy().reset_index(drop=True)

    # 归一化功率统计
    norm_stats = df_processed['normalized_power'].describe()
    print(f"\n归一化功率统计:")
    print(f"  均值: {norm_stats['mean']:.3f}")
    print(f"  标准差: {norm_stats['std']:.3f}")
    print(f"  最小值: {norm_stats['min']:.3f}")
    print(f"  最大值: {norm_stats['max']:.3f}")
    print(f"  τ > 1 的比例: {(df_processed['normalized_power'] > 1).sum() / len(df_processed) * 100:.1f}%")

    return df_processed, removed_info


def plot_clear_sky_results(df_processed, sample_days=7):
    """
    可视化Clear Sky Model的结果

    复现论文中的Figure 8和Figure 9效果
    """

    print(f"\n=== 可视化Clear Sky Model结果 ===")

    # 选择几个有代表性的日子进行可视化
    unique_days = df_processed['day_of_year'].unique()

    # 尽量选择夏天的日子（数据更好）
    summer_days = unique_days[(unique_days >= 120) & (unique_days <= 250)]
    if len(summer_days) >= sample_days:
        selected_days = np.random.choice(summer_days, sample_days, replace=False)
    else:
        selected_days = np.random.choice(unique_days, min(sample_days, len(unique_days)), replace=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_days)))

    for i, day in enumerate(selected_days):
        day_data = df_processed[df_processed['day_of_year'] == day].copy()

        if len(day_data) > 0:
            # 按时间排序
            day_data = day_data.sort_values('time_of_day')
            time_hours = day_data['time_of_day'] / 60

            # 上图：实际功率 vs 晴空功率
            ax1.plot(time_hours, day_data['ACTIVEPOWER'],
                     color=colors[i], linewidth=1.5, alpha=0.8,
                     label=f'实际功率 Day{day}' if i < 3 else "")
            ax1.plot(time_hours, day_data['clear_sky_power'],
                     color=colors[i], linewidth=1.5, linestyle='--', alpha=0.8,
                     label=f'晴空功率 Day{day}' if i < 3 else "")

            # 下图：归一化功率
            ax2.plot(time_hours, day_data['normalized_power'],
                     color=colors[i], linewidth=1.5, alpha=0.8,
                     label=f'Day {day}' if i < 5 else "")

    # 设置上图
    ax1.set_ylabel('功率 (W)')
    ax1.set_title('实际功率 vs 晴空功率对比')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(6, 18)  # 只显示白天时段

    # 设置下图
    ax2.axhline(y=1, color='black', linestyle=':', alpha=0.7, linewidth=1)
    ax2.set_ylabel('归一化功率 τ')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_title('归一化功率 τ = p / p_cs')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(6, 18)
    ax2.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.show()


def analyze_normalization_quality(df_processed):
    """
    分析归一化质量

    评估Clear Sky Model的效果
    """

    print(f"\n=== 归一化质量分析 ===")

    # 1. 按时间段分析归一化效果
    df_processed['hour'] = df_processed['time_of_day'] // 60
    hourly_stats = df_processed.groupby('hour')['normalized_power'].agg(['mean', 'std', 'count'])

    print(f"各时段归一化功率统计 (只显示白天时段):")
    daytime_hours = hourly_stats[(hourly_stats.index >= 7) & (hourly_stats.index <= 17)]
    for hour, stats in daytime_hours.iterrows():
        print(f"  {hour:02d}时: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}, 样本数={stats['count']}")

    # 2. 分析原始功率和归一化功率的变异系数
    original_cv = df_processed['ACTIVEPOWER'].std() / df_processed['ACTIVEPOWER'].mean()
    normalized_cv = df_processed['normalized_power'].std() / df_processed['normalized_power'].mean()

    print(f"\n变异系数对比:")
    print(f"  原始功率变异系数: {original_cv:.3f}")
    print(f"  归一化功率变异系数: {normalized_cv:.3f}")
    print(f"  变异系数降低: {(original_cv - normalized_cv) / original_cv * 100:.1f}%")

    # 3. 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始功率分布
    axes[0].hist(df_processed['ACTIVEPOWER'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('原始功率 (W)')
    axes[0].set_ylabel('频次')
    axes[0].set_title('原始功率分布')
    axes[0].grid(True, alpha=0.3)

    # 归一化功率分布
    axes[1].hist(df_processed['normalized_power'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('归一化功率 τ')
    axes[1].set_ylabel('频次')
    axes[1].set_title('归一化功率分布')
    axes[1].grid(True, alpha=0.3)

    # 按小时的归一化功率箱线图
    hour_data = []
    hour_labels = []
    for hour in range(7, 18):
        hour_norm_power = df_processed[df_processed['hour'] == hour]['normalized_power']
        if len(hour_norm_power) > 0:
            hour_data.append(hour_norm_power)
            hour_labels.append(f'{hour}h')

    axes[2].boxplot(hour_data, labels=hour_labels)
    axes[2].set_xlabel('时间')
    axes[2].set_ylabel('归一化功率 τ')
    axes[2].set_title('各时段归一化功率分布')
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# 主函数：完整的Clear Sky Model流程
def run_clear_sky_model(df, quantile_level=0.85, bandwidth_day=50, bandwidth_time=45,
                        power_threshold=0.2, sample_size=None):
    """
    运行完整的Clear Sky Model

    Parameters:
    - df: 原始DataFrame
    - sample_size: 如果指定，则只处理部分数据（用于快速测试）
    """

    print("🌞 开始运行Clear Sky Model 🌞")

    # 如果指定了sample_size，则只处理部分数据
    if sample_size is not None and sample_size < len(df):
        print(f"⚠️  测试模式：只处理 {sample_size} 个数据点")
        df_sample = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_sample = df.copy()

    # 步骤1: 提取时间特征
    df_with_features = extract_time_features(df_sample)

    # 步骤2-3: 批量估计晴空功率
    clear_sky_powers = estimate_clear_sky_power_batch(
        df_with_features, quantile_level, bandwidth_day, bandwidth_time
    )

    # 步骤4: 计算归一化功率
    df_processed, removed_info = compute_normalized_power(
        df_with_features, clear_sky_powers, power_threshold
    )

    # 可视化结果
    plot_clear_sky_results(df_processed)

    # 分析质量
    analyze_normalization_quality(df_processed)

    print(f"\n✅ Clear Sky Model 完成！")
    print(f"   输入数据: {len(df)} 点")
    print(f"   有效数据: {len(df_processed)} 点")
    print(f"   数据利用率: {len(df_processed) / len(df) * 100:.1f}%")

    return df_processed, removed_info


# 使用示例
if __name__ == "__main__":
    print("第四步：批量估计晴空功率实现完成！")
    print("\n使用方法:")
    print("# 完整运行（可能耗时较长）:")
    print("df_processed, info = run_clear_sky_model(your_dataframe)")
    print("\n# 快速测试（推荐先用小样本测试）:")
    print("df_processed, info = run_clear_sky_model(your_dataframe, sample_size=1000)")
    print("\n⚠️  建议先用小样本测试，确认无误后再处理全部数据！")

    # 查询参数
    farmid = "1"  # 替换为实际的电站ID
    start_time = "2023-01-01 00:00:00"
    end_time = "2023-12-31 23:45:00"

    # 执行查询
    df = query_solar_data(farmid, start_time, end_time)
    df_processed, info = run_clear_sky_model(df, sample_size=1000)