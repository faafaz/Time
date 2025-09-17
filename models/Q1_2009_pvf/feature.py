import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from database.mysql_dataloader import query_solar_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def extract_time_features(df):
    """
    第一步：提取时间特征

    Bacher论文的核心思想：
    - 将时间分解为两个维度：day of year (x) 和 time of day (y)
    - 这样可以用二维统计平滑来建模晴空功率

    Parameters:
    - df: 包含OBSERVETIME, ACTIVEPOWER的DataFrame

    Returns:
    - df: 添加了day_of_year和time_of_day列的DataFrame
    """

    print("=== 第一步：提取时间特征 ===")

    df = df.copy()

    # 确保OBSERVETIME是datetime类型
    df['OBSERVETIME'] = pd.to_datetime(df['OBSERVETIME'])

    # 提取day of year (1-365/366)
    # 这是论文中的 x 变量
    df['day_of_year'] = df['OBSERVETIME'].dt.dayofyear

    # 提取time of day (以分钟为单位，0-1439)
    # 这是论文中的 y 变量
    df['time_of_day'] = (df['OBSERVETIME'].dt.hour * 60 +
                         df['OBSERVETIME'].dt.minute)

    print(f"数据时间范围：")
    print(f"  开始时间: {df['OBSERVETIME'].min()}")
    print(f"  结束时间: {df['OBSERVETIME'].max()}")
    print(f"  天数跨度: {df['day_of_year'].max() - df['day_of_year'].min() + 1} 天")

    print(f"\n时间特征统计：")
    print(f"  day_of_year: {df['day_of_year'].min()} ~ {df['day_of_year'].max()}")
    print(f"  time_of_day: {df['time_of_day'].min()} ~ {df['time_of_day'].max()} 分钟")
    print(
        f"                ({df['time_of_day'].min() // 60:02d}:{df['time_of_day'].min() % 60:02d} ~ {df['time_of_day'].max() // 60:02d}:{df['time_of_day'].max() % 60:02d})")

    return df


def plot_time_features(df):
    """
    可视化时间特征

    这有助于理解数据的时间分布
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. day_of_year 分布
    axes[0, 0].hist(df['day_of_year'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Day of Year')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('Day of Year 分布')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. time_of_day 分布
    time_hours = df['time_of_day'] / 60  # 转换为小时
    axes[0, 1].hist(time_hours, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Time of Day (小时)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('Time of Day 分布')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 二维散点图：day_of_year vs time_of_day
    scatter = axes[1, 0].scatter(df['day_of_year'], time_hours,
                                 c=df['ACTIVEPOWER'], cmap='viridis',
                                 s=1, alpha=0.6)
    axes[1, 0].set_xlabel('Day of Year')
    axes[1, 0].set_ylabel('Time of Day (小时)')
    axes[1, 0].set_title('时间二维分布（颜色=功率）')
    plt.colorbar(scatter, ax=axes[1, 0], label='Active Power (W)')

    # 4. 功率随时间的变化（选择几天样本）
    sample_days = df['day_of_year'].unique()[:5]  # 选择前5天作为样本
    for day in sample_days:
        day_data = df[df['day_of_year'] == day]
        if len(day_data) > 0:
            day_time_hours = day_data['time_of_day'] / 60
            axes[1, 1].plot(day_time_hours, day_data['ACTIVEPOWER'],
                            alpha=0.7, linewidth=1, label=f'Day {day}')

    axes[1, 1].set_xlabel('Time of Day (小时)')
    axes[1, 1].set_ylabel('Active Power (W)')
    axes[1, 1].set_title('功率日变化模式（样本天数）')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 使用示例和测试
if __name__ == "__main__":
    print("第一步：时间特征提取器已准备就绪！")
    print("\n使用方法：")
    print("1. df_with_features = extract_time_features(your_dataframe)")
    print("2. plot_time_features(df_with_features)")
    print("\n这是Clear Sky Model的第一步，为后续的二维平滑做准备。")

    # 查询参数
    farmid = "1"  # 替换为实际的电站ID
    start_time = "2023-01-01 00:00:00"
    end_time = "2023-12-31 23:45:00"

    # 执行查询
    df = query_solar_data(farmid, start_time, end_time)
    df_with_features = extract_time_features(df)
    plot_time_features(df_with_features)
