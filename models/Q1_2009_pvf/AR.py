import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings

from database.mysql_dataloader import query_solar_data
from models.Q1_2009_pvf.model import run_clear_sky_model

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ARModel:
    """
    实现Bacher et al. (2009)论文中的AR模型

    AR模型公式(21): τ_{t+k} = m + a1*τ_t + a2*τ_{t-s(k)} + e_{t+k}
    其中 s(k) = 24 + k mod 24 (确保使用最新的日周期观测)

    这个模型使用：
    - 最近的归一化功率观测值 (AR(1) 项)
    - 24小时前的归一化功率观测值 (日周期 AR 项)
    """

    def __init__(self, prediction_horizons=None):
        """
        初始化AR模型

        Parameters:
        - prediction_horizons: 预测时长列表（小时），例如[1,2,3,6,12,24]
        """
        if prediction_horizons is None:
            self.prediction_horizons = [1, 2, 3, 6, 12, 24]  # 默认预测时长
        else:
            self.prediction_horizons = prediction_horizons

        self.models = {}  # 存储不同时长的模型参数
        self.is_fitted = False

    def prepare_data_for_ar(self, df_processed, time_resolution_minutes=15):
        """
        为AR模型准备数据

        支持15分钟、30分钟、60分钟等不同分辨率
        论文原文使用15分钟数据！

        Parameters:
        - df_processed: 包含归一化功率的DataFrame
        - time_resolution_minutes: 时间分辨率（分钟），15=15分钟，60=1小时

        Returns:
        - df_resampled: 重采样后的数据
        """

        print("=== 第五步：准备AR模型数据 ===")
        print(f"时间分辨率: {time_resolution_minutes} 分钟")

        df = df_processed.copy()
        df['OBSERVETIME'] = pd.to_datetime(df['OBSERVETIME'])
        df = df.sort_values('OBSERVETIME')

        # 如果是15分钟且数据已经是15分钟，则不需要重采样
        if time_resolution_minutes == 15:
            # 检查数据间隔
            time_diff = df['OBSERVETIME'].diff().median()
            if time_diff <= pd.Timedelta(minutes=20):  # 容忍一些误差
                print("数据已经是15分钟间隔，无需重采样")
                df_resampled = df.copy()
            else:
                print("重采样到15分钟间隔")
                df.set_index('OBSERVETIME', inplace=True)
                df_resampled = df.resample('15T').agg({
                    'normalized_power': 'mean',
                    'ACTIVEPOWER': 'mean',
                    'clear_sky_power': 'mean'
                }).dropna()
                df_resampled.reset_index(inplace=True)
        else:
            # 重采样到指定分辨率
            df.set_index('OBSERVETIME', inplace=True)
            resample_rule = f'{time_resolution_minutes}T'
            df_resampled = df.resample(resample_rule).agg({
                'normalized_power': 'mean',
                'ACTIVEPOWER': 'mean',
                'clear_sky_power': 'mean'
            }).dropna()
            df_resampled.reset_index(inplace=True)

        print(f"原始数据点: {len(df_processed)}")
        print(f"处理后数据点: {len(df_resampled)}")
        print(f"时间范围: {df_resampled['OBSERVETIME'].min()} 到 {df_resampled['OBSERVETIME'].max()}")
        print(f"时间跨度: {(df_resampled['OBSERVETIME'].max() - df_resampled['OBSERVETIME'].min()).days} 天")

        return df_resampled

    def create_ar_features(self, df_resampled, time_resolution_minutes=15, max_horizon_hours=24):
        """
        创建AR模型的特征

        论文公式(21): τ_{t+k} = m + a1*τ_t + a2*τ_{t-s(k)} + e_{t+k}
        其中 s(k) 确保使用最新的日周期观测

        Parameters:
        - df_resampled: 重采样后的数据
        - time_resolution_minutes: 时间分辨率（分钟）
        - max_horizon_hours: 最大预测时长（小时）

        Returns:
        - df_features: 包含特征的DataFrame
        """

        print(f"\n=== 创建AR特征 ===")

        df = df_resampled.copy()

        # 计算每小时包含多少个时间步
        steps_per_hour = 60 // time_resolution_minutes
        print(f"每小时时间步数: {steps_per_hour}")

        # 创建滞后特征
        # lag_1: 最近的观测值 (AR(1) 项)
        df['lag_1'] = df['normalized_power'].shift(1)

        # lag_daily: 24小时前的观测值 (日周期 AR 项)
        daily_lag = 24 * steps_per_hour
        df['lag_daily'] = df['normalized_power'].shift(daily_lag)

        print(f"滞后特征:")
        print(f"  lag_1: 1个时间步前 ({time_resolution_minutes}分钟前)")
        print(f"  lag_daily: {daily_lag}个时间步前 (24小时前)")

        # 创建不同预测时长的目标变量
        for k in self.prediction_horizons:
            if k <= max_horizon_hours:
                # 计算k小时对应的时间步数
                k_steps = k * steps_per_hour
                df[f'target_{k}h'] = df['normalized_power'].shift(-k_steps)
                print(f"  target_{k}h: {k_steps}个时间步后")

        # 移除缺失值
        df_features = df.dropna()

        print(f"\n特征创建完成:")
        print(f"  原始数据点: {len(df)}")
        print(f"  可用数据点: {len(df_features)}")
        print(f"  特征: lag_1, lag_daily")
        print(f"  目标变量: {[f'target_{k}h' for k in self.prediction_horizons if k <= max_horizon_hours]}")

        # 验证特征质量
        self._validate_features(df_features)

        return df_features

    def _validate_features(self, df_features):
        """验证特征质量"""

        print(f"\n特征质量验证:")

        # 检查自相关性
        corr_lag1 = df_features['normalized_power'].corr(df_features['lag_1'])
        corr_lag_daily = df_features['normalized_power'].corr(df_features['lag_daily'])

        print(f"  自相关性 lag_1: {corr_lag1:.3f}")
        print(f"  自相关性 lag_daily: {corr_lag_daily:.3f}")

        if corr_lag1 < 0.3:
            print("  ⚠️  警告: lag_1自相关性较低，可能影响预测效果")
        if corr_lag_daily < 0.2:
            print("  ⚠️  警告: lag_daily自相关性较低，日周期性不明显")

        # 检查数据分布
        print(f"  归一化功率统计: 均值={df_features['normalized_power'].mean():.3f}, "
              f"标准差={df_features['normalized_power'].std():.3f}")

    def simple_ar_fit(self, df_features):
        """
        简单的AR模型拟合（使用最小二乘法）

        在实现RLS之前，先用简单方法验证模型结构
        """

        print(f"\n=== 拟合简单AR模型 ===")

        for k in self.prediction_horizons:
            print(f"\n拟合 {k}小时预测模型...")

            # 准备训练数据
            target_col = f'target_{k}h'
            if target_col not in df_features.columns:
                print(f"  跳过: 目标变量 {target_col} 不存在")
                continue

            # 特征和目标
            X = df_features[['lag_1', 'lag_daily']].values
            y = df_features[target_col].values

            # 添加常数项
            X_with_const = np.column_stack([np.ones(len(X)), X])

            # 最小二乘法拟合
            try:
                coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

                # 存储模型参数
                self.models[k] = {
                    'intercept': coefficients[0],
                    'coef_lag1': coefficients[1],
                    'coef_lag_daily': coefficients[2],
                    'horizon': k
                }

                # 计算拟合优度
                y_pred = X_with_const @ coefficients
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

                print(
                    f"  模型参数: 截距={coefficients[0]:.4f}, lag1={coefficients[1]:.4f}, lag_daily={coefficients[2]:.4f}")
                print(f"  拟合效果: RMSE={rmse:.4f}, R²={r2:.4f}")

                self.models[k].update({
                    'rmse': rmse,
                    'r2': r2,
                    'n_samples': len(y)
                })

            except np.linalg.LinAlgError:
                print(f"  拟合失败: 矩阵奇异")
                continue

        self.is_fitted = True
        print(f"\nAR模型拟合完成，共拟合 {len(self.models)} 个时长")

    def predict(self, df_features, start_idx=None, n_predictions=100):
        """
        使用AR模型进行预测

        Parameters:
        - df_features: 特征数据
        - start_idx: 开始预测的位置，如果None则从中间开始
        - n_predictions: 预测的样本数量
        """

        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用 simple_ar_fit()")

        if start_idx is None:
            start_idx = len(df_features) // 2

        end_idx = min(start_idx + n_predictions, len(df_features))

        predictions = {}

        print(f"\n=== AR模型预测 ===")
        print(f"预测范围: 索引 {start_idx} 到 {end_idx}")

        for k in self.prediction_horizons:
            if k not in self.models:
                continue

            model = self.models[k]

            # 获取特征
            X = df_features.iloc[start_idx:end_idx][['lag_1', 'lag_daily']].values

            # 预测
            y_pred = (model['intercept'] +
                      model['coef_lag1'] * X[:, 0] +
                      model['coef_lag_daily'] * X[:, 1])

            # 获取真实值
            target_col = f'target_{k}h'
            if target_col in df_features.columns:
                y_true = df_features.iloc[start_idx:end_idx][target_col].values

                # 计算预测误差
                valid_mask = ~np.isnan(y_true)
                if np.sum(valid_mask) > 0:
                    rmse = np.sqrt(mean_squared_error(y_true[valid_mask], y_pred[valid_mask]))
                    print(f"  {k}小时预测 RMSE: {rmse:.4f}")
            else:
                y_true = None

            predictions[k] = {
                'y_pred': y_pred,
                'y_true': y_true,
                'time_index': df_features.iloc[start_idx:end_idx]['OBSERVETIME'].values,
                'horizon': k
            }

        return predictions

    def plot_ar_results(self, df_features, predictions=None):
        """
        可视化AR模型结果
        """

        if predictions is None:
            predictions = self.predict(df_features)

        # 选择几个代表性的预测时长进行可视化
        plot_horizons = [1, 6, 24] if 24 in predictions else list(predictions.keys())[:3]

        fig, axes = plt.subplots(len(plot_horizons), 1, figsize=(15, 4 * len(plot_horizons)))
        if len(plot_horizons) == 1:
            axes = [axes]

        for i, k in enumerate(plot_horizons):
            if k not in predictions:
                continue

            pred_data = predictions[k]
            time_index = pd.to_datetime(pred_data['time_index'])

            axes[i].plot(time_index, pred_data['y_pred'], 'r-', alpha=0.8, linewidth=2, label=f'预测值')

            if pred_data['y_true'] is not None:
                axes[i].plot(time_index, pred_data['y_true'], 'b-', alpha=0.7, linewidth=1, label=f'真实值')

            axes[i].set_ylabel('归一化功率 τ')
            axes[i].set_title(f'{k}小时预测结果 (AR模型)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # 只在最后一个子图显示x轴标签
            if i == len(plot_horizons) - 1:
                axes[i].set_xlabel('时间')

        plt.tight_layout()
        plt.show()

    def summary(self):
        """显示模型总结"""
        if not self.is_fitted:
            print("模型未拟合")
            return

        print(f"\n=== AR模型总结 ===")
        print(f"预测时长: {list(self.models.keys())} 小时")

        print(f"\n模型性能:")
        for k, model in self.models.items():
            print(f"  {k}小时: RMSE={model['rmse']:.4f}, R²={model['r2']:.4f}, 样本数={model['n_samples']}")

        print(f"\n模型结构: τ_{{t+k}} = m + a1*τ_t + a2*τ_{{t-24h}} + e_{{t+k}}")


# 主函数：运行AR模型
def run_ar_model(df_processed, prediction_horizons=None, time_resolution_minutes=15):
    """
    运行完整的AR模型流程

    Parameters:
    - df_processed: Clear Sky Model输出的处理后数据
    - prediction_horizons: 预测时长列表（小时）
    - time_resolution_minutes: 时间分辨率（分钟），15=论文原始分辨率
    """

    print("🤖 开始运行AR模型")
    print(f"时间分辨率: {time_resolution_minutes} 分钟 (论文原始分辨率)")

    # 初始化模型
    ar_model = ARModel(prediction_horizons)

    # 步骤1: 准备数据
    df_resampled = ar_model.prepare_data_for_ar(df_processed, time_resolution_minutes)

    # 检查数据量是否足够（15分钟数据需要更多样本）
    steps_per_hour = 60 // time_resolution_minutes
    min_required_points = 24 * steps_per_hour + max(ar_model.prediction_horizons) * steps_per_hour

    if len(df_resampled) < min_required_points:
        print(f"⚠️  数据量不足: 只有 {len(df_resampled)} 个点，建议至少 {min_required_points} 个点")
        print(f"   (需要24小时历史数据 + 最大预测时长，共约 {min_required_points // steps_per_hour} 小时)")
        return None, None

    # 步骤2: 创建特征
    df_features = ar_model.create_ar_features(df_resampled, time_resolution_minutes)

    if len(df_features) < 100:  # 至少需要100个有效样本进行可靠的回归
        print(f"⚠️  有效样本不足: 只有 {len(df_features)} 个样本，建议至少100个")
        return None, None

    # 步骤3: 拟合模型
    ar_model.simple_ar_fit(df_features)

    # 步骤4: 预测和可视化
    predictions = ar_model.predict(df_features)
    ar_model.plot_ar_results(df_features, predictions)

    # 步骤5: 显示总结
    ar_model.summary()

    print(f"\n✅ AR模型完成！")
    print(f"数据分辨率: {time_resolution_minutes} 分钟")
    print(f"有效样本数: {len(df_features)}")

    return ar_model, df_features


# 使用示例
if __name__ == "__main__":
    print("第五步：AR模型实现完成！")
    print("\n📊 完美支持15分钟数据分辨率 - 与论文原始设置完全一致！")
    print("\n使用方法:")
    print("# 使用15分钟分辨率（论文原始设置，推荐）:")
    print("ar_model, df_features = run_ar_model(df_processed, time_resolution_minutes=15)")
    print("\n# 或者使用小时分辨率:")
    print("ar_model, df_features = run_ar_model(df_processed, time_resolution_minutes=60)")
    print("\n🎯 15分钟数据的优势:")
    print("- 更精细的时间分辨率")
    print("- 更好的短期预测能力")
    print("- 与论文完全一致的设置")
    print("\n这将实现论文中的自回归模型，为下一步的ARX模型和RLS算法做准备！")

    # 查询参数
    farmid = "1"  # 替换为实际的电站ID
    start_time = "2023-03-01 00:00:00"
    end_time = "2023-13-31 23:45:00"

    # 执行查询
    df = query_solar_data(farmid, start_time, end_time)

    df_processed, info = run_clear_sky_model(df)

    ar_model, df_features = run_ar_model(
        df_processed,
        time_resolution_minutes=15,  # 与论文完全一致
        prediction_horizons=[1, 2, 3, 6, 12, 24]
    )
