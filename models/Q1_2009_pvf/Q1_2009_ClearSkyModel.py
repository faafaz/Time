import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.preprocessing import StandardScaler
import warnings

from database.mysql_dataloader import query_solar_data

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ClearSkyModel:
    """
    实现Bacher et al. (2009)论文中的Clear Sky Model

    核心思想：
    p = p_cs * τ
    其中 p 是实际功率，p_cs 是晴空功率，τ 是归一化功率
    """

    def __init__(self, quantile_level=0.85, bandwidth_day=50, bandwidth_time=45):
        """
        初始化Clear Sky Model

        Parameters:
        - quantile_level: 分位数水平，论文中使用0.85
        - bandwidth_day: day of year维度的带宽（天）
        - bandwidth_time: time of day维度的带宽（分钟）
        """
        self.quantile_level = quantile_level
        self.bandwidth_day = bandwidth_day
        self.bandwidth_time = bandwidth_time
        self.clear_sky_power = None
        self.normalized_power = None

    def extract_time_features(self, df):
        """
        提取时间特征：day of year 和 time of day

        这是论文中的核心步骤，将时间分解为两个维度进行建模
        """
        df = df.copy()

        # 确保OBSERVETIME是datetime类型
        df['OBSERVETIME'] = pd.to_datetime(df['OBSERVETIME'])

        # 提取day of year (1-365/366)
        df['day_of_year'] = df['OBSERVETIME'].dt.dayofyear

        # 提取time of day (分钟，0-1439)
        df['time_of_day'] = (df['OBSERVETIME'].dt.hour * 60 +
                             df['OBSERVETIME'].dt.minute)

        return df

    def gaussian_kernel(self, x, xi, bandwidth):
        """
        高斯核函数

        论文公式 (39): w(xt, xi, hx) = f_std(|xt - xi| / hx)
        """
        return np.exp(-0.5 * ((x - xi) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

    def weighted_quantile(self, values, weights, quantile):
        """
        加权分位数计算

        这是Clear Sky Model的核心：使用加权分位数回归估计晴空功率
        """
        # 排序
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # 累积权重
        cumsum_weights = np.cumsum(sorted_weights)
        total_weight = cumsum_weights[-1]

        # 找到分位数对应的值
        target_weight = quantile * total_weight
        idx = np.searchsorted(cumsum_weights, target_weight)

        if idx >= len(sorted_values):
            return sorted_values[-1]
        elif idx == 0:
            return sorted_values[0]
        else:
            # 线性插值
            w1, w2 = cumsum_weights[idx - 1], cumsum_weights[idx]
            v1, v2 = sorted_values[idx - 1], sorted_values[idx]
            alpha = (target_weight - w1) / (w2 - w1) if w2 != w1 else 0
            return v1 + alpha * (v2 - v1)

    def estimate_clear_sky_power(self, df, target_point=None):
        """
        估计晴空功率

        使用二维高斯核加权分位数回归
        论文公式 (7): p_cs = f_max(x, y)
        """
        if target_point is None:
            # 为所有数据点估计晴空功率
            clear_sky_estimates = []

            print("开始估计晴空功率...")
            total_points = len(df)

            for i, row in df.iterrows():
                if i % 1000 == 0:
                    print(f"进度: {i}/{total_points} ({i / total_points * 100:.1f}%)")

                target_day = row['day_of_year']
                target_time = row['time_of_day']

                # 计算权重
                day_weights = self.gaussian_kernel(df['day_of_year'], target_day, self.bandwidth_day)
                time_weights = self.gaussian_kernel(df['time_of_day'], target_time, self.bandwidth_time)

                # 处理time_of_day的周期性（0和1440分钟是相邻的）
                time_diff = np.abs(df['time_of_day'] - target_time)
                time_diff_circular = np.minimum(time_diff, 1440 - time_diff)
                time_weights = self.gaussian_kernel(time_diff_circular, 0, self.bandwidth_time)

                # 组合权重
                combined_weights = day_weights * time_weights

                # 归一化权重
                if combined_weights.sum() > 0:
                    combined_weights = combined_weights / combined_weights.sum()

                    # 计算加权分位数
                    clear_sky_power = self.weighted_quantile(
                        df['ACTIVEPOWER'].values,
                        combined_weights,
                        self.quantile_level
                    )
                else:
                    clear_sky_power = df['ACTIVEPOWER'].iloc[i]

                clear_sky_estimates.append(clear_sky_power)

            self.clear_sky_power = np.array(clear_sky_estimates)

        else:
            # 为单个点估计晴空功率
            target_day, target_time = target_point

            day_weights = self.gaussian_kernel(df['day_of_year'], target_day, self.bandwidth_day)
            time_weights = self.gaussian_kernel(df['time_of_day'], target_time, self.bandwidth_time)
            combined_weights = day_weights * time_weights
            combined_weights = combined_weights / combined_weights.sum()

            return self.weighted_quantile(df['ACTIVEPOWER'].values, combined_weights, self.quantile_level)

    def compute_normalized_power(self, df):
        """
        计算归一化功率

        论文公式 (9): τ_t = p_t / p_cs_t
        """
        if self.clear_sky_power is None:
            raise ValueError("请先估计晴空功率")

        # 计算归一化功率
        normalized_power = df['ACTIVEPOWER'].values / self.clear_sky_power

        # 论文中的处理：移除晴空功率过小的点
        # 公式 (11): 移除 p_cs_t / max(p_cs_t) < 0.2 的点
        max_clear_sky = np.max(self.clear_sky_power)
        valid_mask = (self.clear_sky_power / max_clear_sky) >= 0.2

        print(f"移除低辐射时段: {(~valid_mask).sum()} 个点 ({(~valid_mask).sum() / len(df) * 100:.1f}%)")

        # 只保留有效点
        self.normalized_power = normalized_power[valid_mask]
        self.valid_mask = valid_mask

        return self.normalized_power, valid_mask

    def fit(self, df):
        """
        拟合Clear Sky Model

        主要步骤：
        1. 提取时间特征
        2. 估计晴空功率
        3. 计算归一化功率
        """
        print("=== Clear Sky Model 拟合开始 ===")

        # 步骤1: 提取时间特征
        print("步骤1: 提取时间特征...")
        df_processed = self.extract_time_features(df)

        # 步骤2: 估计晴空功率
        print("步骤2: 估计晴空功率...")
        self.estimate_clear_sky_power(df_processed)

        # 步骤3: 计算归一化功率
        print("步骤3: 计算归一化功率...")
        normalized_power, valid_mask = self.compute_normalized_power(df_processed)

        # 保存处理后的数据
        self.df_processed = df_processed
        self.df_valid = df_processed[valid_mask].copy()
        self.df_valid['clear_sky_power'] = self.clear_sky_power[valid_mask]
        self.df_valid['normalized_power'] = normalized_power

        print(f"=== Clear Sky Model 拟合完成 ===")
        print(f"有效数据点: {len(self.df_valid)} / {len(df)} ({len(self.df_valid) / len(df) * 100:.1f}%)")

        return self.df_valid

    def plot_results(self, sample_days=None):
        """
        绘制Clear Sky Model结果

        复现论文中的Figure 8和Figure 9
        """
        if self.df_valid is None:
            print("请先拟合模型")
            return

        # 如果没有指定天数，选择几个代表性的日子
        if sample_days is None:
            # 选择夏天的几个晴天
            summer_days = self.df_valid[
                (self.df_valid['day_of_year'] >= 150) &
                (self.df_valid['day_of_year'] <= 250)
                ]
            sample_dates = summer_days['OBSERVETIME'].dt.date.unique()[:7]
        else:
            sample_dates = sample_days

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        for date in sample_dates:
            day_data = self.df_valid[self.df_valid['OBSERVETIME'].dt.date == date]

            if len(day_data) > 0:
                time_hours = day_data['OBSERVETIME'].dt.hour + day_data['OBSERVETIME'].dt.minute / 60

                # 上图：实际功率 vs 晴空功率
                ax1.plot(time_hours, day_data['ACTIVEPOWER'], 'b-', alpha=0.7, linewidth=1,
                         label='实际功率' if date == sample_dates[0] else "")
                ax1.plot(time_hours, day_data['clear_sky_power'], 'r--', alpha=0.7, linewidth=1,
                         label='晴空功率' if date == sample_dates[0] else "")

                # 下图：归一化功率
                ax2.plot(time_hours, day_data['normalized_power'], 'g-', alpha=0.7, linewidth=1,
                         label='归一化功率' if date == sample_dates[0] else "")

        ax1.set_ylabel('功率 (W)')
        ax1.set_title('实际功率 vs 晴空功率 (Clear Sky Model)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(6, 18)  # 只显示白天时段

        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('归一化功率')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_title('归一化功率 (τ = p / p_cs)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(6, 18)
        ax2.set_ylim(0, 1.5)

        plt.tight_layout()
        plt.show()

        # 统计信息
        print("\n=== 归一化结果统计 ===")
        print(f"归一化功率均值: {self.df_valid['normalized_power'].mean():.3f}")
        print(f"归一化功率标准差: {self.df_valid['normalized_power'].std():.3f}")
        print(f"归一化功率最大值: {self.df_valid['normalized_power'].max():.3f}")
        print(f"τ > 1 的比例: {(self.df_valid['normalized_power'] > 1).sum() / len(self.df_valid) * 100:.1f}%")


# 使用示例
if __name__ == "__main__":
    print("Clear Sky Model 实现完成！")
    print("\n使用方法：")
    print("1. clear_sky = ClearSkyModel()")
    print("2. df_processed = clear_sky.fit(your_dataframe)")
    print("3. clear_sky.plot_results()")
    print("\n这将实现论文中的核心归一化步骤！")

    # 查询参数
    farmid = "1"  # 替换为实际的电站ID
    start_time = "2023-01-01 00:00:00"
    end_time = "2023-12-31 23:45:00"

    # 执行查询
    df = query_solar_data(farmid, start_time, end_time)

    clear_sky = ClearSkyModel()
    df_processed = clear_sky.fit(df)
    clear_sky.plot_results()
