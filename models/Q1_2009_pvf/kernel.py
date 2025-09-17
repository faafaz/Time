import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from database.mysql_dataloader import query_solar_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def extract_time_features(df):
    """
    第一步：提取时间特征（已完成）
    """
    print("=== 第一步：提取时间特征 ===")

    df = df.copy()
    df['OBSERVETIME'] = pd.to_datetime(df['OBSERVETIME'])
    df['day_of_year'] = df['OBSERVETIME'].dt.dayofyear
    df['time_of_day'] = (df['OBSERVETIME'].dt.hour * 60 +
                         df['OBSERVETIME'].dt.minute)

    print(f"时间特征提取完成: day_of_year({df['day_of_year'].min()}-{df['day_of_year'].max()}), "
          f"time_of_day({df['time_of_day'].min()}-{df['time_of_day'].max()})")

    return df


def gaussian_kernel_1d(x, xi, bandwidth):
    """
    第二步：一维高斯核函数

    这是论文公式(39): w(xt, xi, hx) = f_std(|xt - xi| / hx)
    其中 f_std 是标准正态分布的概率密度函数

    Parameters:
    - x: 目标点
    - xi: 观测点数组
    - bandwidth: 带宽参数

    Returns:
    - weights: 权重数组
    """
    # 计算距离
    distance = np.abs(x - xi)

    # 高斯核权重
    weights = np.exp(-0.5 * (distance / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

    return weights


def time_of_day_distance(target_time, time_array):
    """
    处理time_of_day的周期性距离

    重要：时间是周期性的！
    - 0分钟和1440分钟（第二天0点）是相邻的
    - 23:59和00:01的距离应该是2分钟，不是1438分钟

    Parameters:
    - target_time: 目标时间（分钟）
    - time_array: 时间数组（分钟）

    Returns:
    - circular_distance: 考虑周期性的距离
    """
    # 计算普通距离
    linear_distance = np.abs(time_array - target_time)

    # 计算周期性距离（通过一天的另一边）
    circular_distance = np.minimum(linear_distance, 1440 - linear_distance)

    return circular_distance


def gaussian_kernel_time_of_day(target_time, time_array, bandwidth):
    """
    针对time_of_day的特殊高斯核函数

    考虑时间的周期性特征
    """
    # 使用周期性距离
    distances = time_of_day_distance(target_time, time_array)

    # 计算高斯权重
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

    return weights


def compute_2d_weights(target_day, target_time, df, bandwidth_day=50, bandwidth_time=45):
    """
    计算二维权重

    这是Clear Sky Model的核心：
    - 将day_of_year和time_of_day的权重相乘
    - 论文公式(38): k(xt, yt, xi, yi) = w(xt, xi, hx) * w(yt, yi, hy)

    Parameters:
    - target_day: 目标day_of_year
    - target_time: 目标time_of_day
    - df: 包含时间特征的DataFrame
    - bandwidth_day: day of year维度的带宽（天）
    - bandwidth_time: time of day维度的带宽（分钟）

    Returns:
    - weights_2d: 二维权重数组
    - weights_day: day维度权重（用于调试）
    - weights_time: time维度权重（用于调试）
    """

    # 计算day_of_year维度的权重
    weights_day = gaussian_kernel_1d(target_day, df['day_of_year'].values, bandwidth_day)

    # 计算time_of_day维度的权重（考虑周期性）
    weights_time = gaussian_kernel_time_of_day(target_time, df['time_of_day'].values, bandwidth_time)

    # 组合二维权重
    weights_2d = weights_day * weights_time

    # 归一化权重
    if weights_2d.sum() > 0:
        weights_2d = weights_2d / weights_2d.sum()

    return weights_2d, weights_day, weights_time


def visualize_kernel_weights(df, target_day=150, target_time=720, bandwidth_day=50, bandwidth_time=45):
    """
    可视化核权重分布

    这有助于理解权重是如何分布的

    Parameters:
    - target_day: 目标日期（一年中的第几天）
    - target_time: 目标时间（分钟，720=12:00）
    """

    print(f"\n=== 可视化权重分布 ===")
    print(f"目标点: Day {target_day}, Time {target_time // 60:02d}:{target_time % 60:02d}")

    # 计算权重
    weights_2d, weights_day, weights_time = compute_2d_weights(
        target_day, target_time, df, bandwidth_day, bandwidth_time
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Day of year 权重分布
    axes[0, 0].plot(df['day_of_year'], weights_day, 'b-', linewidth=2)
    axes[0, 0].axvline(target_day, color='red', linestyle='--', label=f'目标日期: {target_day}')
    axes[0, 0].set_xlabel('Day of Year')
    axes[0, 0].set_ylabel('权重')
    axes[0, 0].set_title(f'Day维度权重分布 (带宽={bandwidth_day}天)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Time of day 权重分布
    time_hours = df['time_of_day'] / 60
    target_hour = target_time / 60
    axes[0, 1].plot(time_hours, weights_time, 'g-', linewidth=2)
    axes[0, 1].axvline(target_hour, color='red', linestyle='--', label=f'目标时间: {target_hour:.1f}h')
    axes[0, 1].set_xlabel('Time of Day (小时)')
    axes[0, 1].set_ylabel('权重')
    axes[0, 1].set_title(f'Time维度权重分布 (带宽={bandwidth_time}分钟)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 二维权重热力图
    # 为了可视化，我们需要创建一个网格
    day_range = np.linspace(df['day_of_year'].min(), df['day_of_year'].max(), 50)
    time_range = np.linspace(0, 1440, 48)  # 每半小时一个点

    weight_grid = np.zeros((len(time_range), len(day_range)))

    for i, day in enumerate(day_range):
        for j, time in enumerate(time_range):
            w_day = gaussian_kernel_1d(target_day, np.array([day]), bandwidth_day)[0]
            w_time = gaussian_kernel_time_of_day(target_time, np.array([time]), bandwidth_time)[0]
            weight_grid[j, i] = w_day * w_time

    im = axes[1, 0].imshow(weight_grid, extent=[day_range[0], day_range[-1], time_range[-1] / 60, time_range[0] / 60],
                           aspect='auto', cmap='viridis', origin='upper')
    axes[1, 0].plot(target_day, target_hour, 'r*', markersize=15, label='目标点')
    axes[1, 0].set_xlabel('Day of Year')
    axes[1, 0].set_ylabel('Time of Day (小时)')
    axes[1, 0].set_title('二维权重热力图')
    axes[1, 0].legend()
    plt.colorbar(im, ax=axes[1, 0])

    # 4. 权重最高的数据点
    top_indices = np.argsort(weights_2d)[-100:]  # 权重最高的100个点
    top_weights = weights_2d[top_indices]
    top_powers = df['ACTIVEPOWER'].iloc[top_indices]

    axes[1, 1].scatter(top_weights, top_powers, alpha=0.6, s=30)
    axes[1, 1].set_xlabel('权重')
    axes[1, 1].set_ylabel('功率 (W)')
    axes[1, 1].set_title('权重最高的数据点')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 输出统计信息
    print(f"权重统计:")
    print(f"  非零权重点数: {(weights_2d > 1e-10).sum()} / {len(weights_2d)}")
    print(f"  最大权重: {weights_2d.max():.6f}")
    print(f"  权重总和: {weights_2d.sum():.6f}")


def test_kernel_functions():
    """
    测试核函数的基本功能
    """
    print("=== 测试核函数 ===")

    # 测试一维高斯核
    x = np.array([1, 2, 3, 4, 5])
    target = 3
    weights = gaussian_kernel_1d(target, x, bandwidth=1.0)
    print(f"一维高斯核测试:")
    print(f"  输入: {x}, 目标: {target}")
    print(f"  权重: {weights}")
    print(f"  最大权重位置: {x[np.argmax(weights)]}")

    # 测试时间周期性距离
    times = np.array([0, 30, 1410, 1439])  # 0:00, 0:30, 23:30, 23:59
    target_time = 0  # 午夜
    distances = time_of_day_distance(target_time, times)
    print(f"\n时间周期性距离测试:")
    print(f"  时间点: {times} (分钟)")
    print(f"  到午夜的距离: {distances}")


# 使用示例
if __name__ == "__main__":
    print("第二步：高斯核函数实现完成！")
    print("\n使用方法：")
    print("1. test_kernel_functions()  # 测试核函数")
    print("2. df_with_features = extract_time_features(your_dataframe)")
    print("3. visualize_kernel_weights(df_with_features)  # 可视化权重分布")
    print("\n下一步将实现加权分位数计算。")

    # 查询参数
    farmid = "1"  # 替换为实际的电站ID
    start_time = "2023-01-01 00:00:00"
    end_time = "2023-12-31 23:45:00"

    # 执行查询
    df = query_solar_data(farmid, start_time, end_time)

    test_kernel_functions()
    df_with_features = extract_time_features(df)
    visualize_kernel_weights(df_with_features)