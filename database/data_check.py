import pandas as pd
from datetime import datetime, timedelta


def check_missing_dates(df, freq, time_column='OBSERVETIME'):
    """
    检查DataFrame中按日期缺失的数据

    Parameters:
    - df: pandas.DataFrame 输入的数据框
    - freq: int 时间频率（分钟），例如15表示15分钟间隔
    - time_column: str 时间列名，默认为'OBSERVETIME'

    Returns:
    - pandas.DataFrame: 包含缺失数据的日期和缺失条数
        - 'date': 缺少数据的日期
        - 'missing_count': 该日期缺少的数据条数
        - 'expected_count': 该日期应有的数据条数
    """

    if df.empty:
        return pd.DataFrame(columns=['date', 'missing_count', 'expected_count'])

    # 确保时间列是datetime类型
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])

    # 按时间排序并去重
    df_sorted = df.sort_values(time_column).drop_duplicates(subset=[time_column])

    # 获取时间范围
    start_time = df_sorted[time_column].min()
    end_time = df_sorted[time_column].max()

    # 计算每天应有的数据点数（一天1440分钟除以频率）
    points_per_day = 1440 // freq

    # 生成完整的日期范围
    date_range = pd.date_range(
        start=start_time.date(),
        end=end_time.date(),
        freq='D'
    )

    missing_dates = []

    for date in date_range:
        # 获取该日期的数据
        day_data = df_sorted[df_sorted[time_column].dt.date == date.date()]
        actual_count = len(day_data)

        # 如果该日期缺少数据
        if actual_count < points_per_day:
            missing_count = points_per_day - actual_count
            missing_dates.append({
                'date': date.date(),
                'missing_count': missing_count,
                'expected_count': points_per_day,
                'actual_count': actual_count
            })

    result_df = pd.DataFrame(missing_dates)
    return result_df


def print_missing_dates_summary(missing_df):
    """
    打印缺失日期汇总信息

    Parameters:
    - missing_df: check_missing_dates函数返回的DataFrame
    """
    if missing_df.empty:
        print("✓ 所有日期的数据都完整")
        return

    print(f"发现 {len(missing_df)} 个日期缺少数据:")
    print("-" * 50)

    for _, row in missing_df.iterrows():
        print(
            f"{row['date']}: 缺少 {row['missing_count']} 条数据 (应有{row['expected_count']}条，实有{row['actual_count']}条)")

    total_missing = missing_df['missing_count'].sum()
    print("-" * 50)
    print(f"总计缺少: {total_missing} 条数据")


# 使用示例
if __name__ == "__main__":
    # 创建示例数据 - 模拟几天的数据，但故意缺少某些日期的部分数据
    import numpy as np

    base_time = datetime(2024, 1, 1, 0, 0, 0)
    times = []

    # 生成3天的数据，但第2天和第3天缺少一些数据点
    for day in range(3):
        day_start = base_time + timedelta(days=day)

        if day == 0:
            # 第1天：完整数据（96个点，15分钟间隔）
            for i in range(96):
                times.append(day_start + timedelta(minutes=15 * i))
        elif day == 1:
            # 第2天：缺少20个数据点
            for i in range(76):  # 只有76个点而不是96个
                times.append(day_start + timedelta(minutes=15 * i))
        else:
            # 第3天：缺少30个数据点
            for i in range(66):  # 只有66个点而不是96个
                times.append(day_start + timedelta(minutes=15 * i))

    # 创建示例DataFrame
    df = pd.DataFrame({
        'OBSERVETIME': times,
        'ACTIVEPOWER': np.random.rand(len(times)) * 1000
    })

    print("示例DataFrame:")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['OBSERVETIME'].min()} 到 {df['OBSERVETIME'].max()}")

    # 检查15分钟间隔的缺失数据
    missing_df = check_missing_dates(df, freq=15, time_column='OBSERVETIME')

    # 打印结果
    print("\n" + "=" * 50)
    print_missing_dates_summary(missing_df)

    # 也可以直接查看DataFrame
    if not missing_df.empty:
        print("\n缺失数据详情:")
        print(missing_df)