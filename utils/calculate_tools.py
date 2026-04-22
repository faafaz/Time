import pandas as pd
import numpy as np


def _detect_column(df, candidates):
    """在多个候选列名中自动选择存在的一列，找不到则返回None。"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def calculate_station_metrics(
    csv_file_path,
    logger=None,
    station_col='区站号(数字)',
    time_col='Time',
    true_col=None,
    pred_col='pred',
):
    """
    按区站号（数字）进行分组，并在组内按时间排序，计算每个站点在整个时间段内的7项核心指标：
    MSE, MAE, RMSE, Pearson, MRE, MBE, R²。

    列名自适配：
      - time_col: 默认“时间”
      - station_col: 默认“区站号(数字)”
      - true_col: 若不指定，将在以下候选中自动匹配（按顺序）：
          ['观测温度','观测风速','tmp_grid','网格温度','网格风速','y','target']
      - pred_col: 默认“pred”

    返回：
        station_metrics_df: 按站点聚合后的DataFrame，列为 [区站号(数字), MSE, MAE, RMSE, Pearson, MRE, MBE, R2]
        overall_metrics: 按站点指标取均值后的总体字典（可用于日志汇总）
    """
    if logger is None:
        import logging
        logger = logging.getLogger('solar')

    # 读取数据
    try:
        df = pd.read_csv(csv_file_path)
    except Exception:
        df = pd.read_csv(csv_file_path, encoding='gbk')

    # 自动检测真实值列
    if true_col is None:
        true_col = _detect_column(df, ['Power'])

    # 基础校验
    for col, name in [(time_col, 'Time'), (pred_col, 'pred')]:
        if col not in df.columns:
            raise KeyError(f"缺少必要列: '{col}'（{name}）")
    if true_col is None or true_col not in df.columns:
        raise KeyError("无法在CSV中找到真实值列，请通过 true_col 参数显式指定，或确保列名为[观测温度/观测风速/tmp_grid/网格温度/网格风速/y/target]之一")

    # 转换时间与站点类型
    df[time_col] = pd.to_datetime(df[time_col])
    # 区站号尽可能转为数值，无法转换则保留原值
    try:
        df[station_col] = pd.to_numeric(df[station_col], errors='ignore')
    except Exception:
        pass

    station_results = []
    for sid, group in df.groupby(station_col):
        # 组内按时间排序
        g = group.sort_values(time_col).copy()
        # 仅使用有效对齐的真实与预测数据
        g = g.dropna(subset=[true_col, pred_col])
        if len(g) == 0:
            mse = mae = rmse = pearson_corr = mre = mbe = r2 = float('nan')
        else:
            true_values = g[true_col].values.astype(float)
            pred_values = g[pred_col].values.astype(float)
            # 基础误差
            diff = pred_values - true_values
            abs_true = np.abs(true_values) + 1e-6
            mse = np.mean(diff ** 2)
            mae = np.mean(np.abs(diff))
            rmse = np.sqrt(mse)
            # 皮尔逊相关系数
            pearson_corr = (
                np.corrcoef(true_values, pred_values)[0, 1]
                if len(true_values) > 1 else float('nan')
            )
            # 相对与偏差指标
            mre = np.mean(np.abs(diff) / abs_true)
            mbe = np.mean(diff)
            # 决定系数 R²
            mean_true = np.mean(true_values)
            ss_total = np.sum((true_values - mean_true) ** 2)
            ss_residual = np.sum((true_values - pred_values) ** 2)
            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else float('nan')

        station_results.append({
            station_col: sid,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Pearson': pearson_corr,
            'MRE': mre,
            'MBE': mbe,
            'R2': r2,
        })

    station_metrics_df = pd.DataFrame(station_results)

    # 按站点求平均(忽略NaN)
    overall_metrics = {
        'MSE': station_metrics_df['MSE'].mean(),
        'MAE': station_metrics_df['MAE'].mean(),
        'RMSE': station_metrics_df['RMSE'].mean(),
        'Pearson': station_metrics_df['Pearson'].mean(),
        'MRE': station_metrics_df['MRE'].mean(),
        'MBE': station_metrics_df['MBE'].mean(),
        'R2': station_metrics_df['R2'].mean(),
    }

    return station_metrics_df, overall_metrics


def print_station_metrics(station_metrics_df, overall_metrics, logger=None, station_col='区站号(数字)'):
    """
    打印每个区站号在整个时间段内的7项核心指标（不包含按天统计）。
    """
    if logger is None:
        import logging
        logger = logging.getLogger('solar')

    table_content = []
    table_content.append("")
    table_content.append("=" * 100)
    table_content.append("按区站号统计的指标（时间段均值）:")
    table_content.append("=" * 100)
    header = f"{station_col:<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Pearson':<12} {'MRE':<12} {'MBE':<12} {'R2':<12}"
    table_content.append(header)
    table_content.append("-" * 100)

    for _, row in station_metrics_df.sort_values(station_col).iterrows():
        sid_str = str(row[station_col])
        if pd.isna(row['MSE']):
            table_content.append(f"{sid_str:<12} {'--':<12} {'--':<12} {'--':<12} {'--':<12} {'--':<12} {'--':<12} {'--':<12}")
        else:
            table_content.append(
                f"{sid_str:<12} {row['MSE']:<12.4f} {row['MAE']:<12.4f} {row['RMSE']:<12.4f} {row['Pearson']:<12.4f} {row['MRE']:<12.4f} {row['MBE']:<12.4f} {row['R2']:<12.4f}")

    table_content.append("-" * 100)
    table_content.append(
        f"{'总体(站点均值)':<12} {overall_metrics['MSE']:<12.4f} {overall_metrics['MAE']:<12.4f} {overall_metrics['RMSE']:<12.4f} {overall_metrics['Pearson']:<12.4f} {overall_metrics['MRE']:<12.4f} {overall_metrics['MBE']:<12.4f} {overall_metrics['R2']:<12.4f}")
    table_content.append("=" * 100)

    logger.info("\n".join(table_content))




def calculate_global_metrics_from_arrays(true_list, pred_list, logger=None, acc_mode='relative', acc_tolerance=0.1):
    """
    不分站点，基于(真实, 预测)序列计算整体指标（使用传入的数据，无论是否归一化）：
    - MSE, MAE, RMSE, R², MAPE

    说明：测试阶段请传入反归一化（实际功率）数据，以满足当前评估需求。
    """
    import logging
    import numpy as np  # 确保导入了numpy
    
    if logger is None:
        logger = logging.getLogger('metrics')

    true_values = np.asarray(true_list, dtype=float)
    pred_values = np.asarray(pred_list, dtype=float)

    # 1. 对齐有效样本 (去除 NaN)
    mask = ~np.isnan(true_values) & ~np.isnan(pred_values)
    true_values = true_values[mask]
    pred_values = pred_values[mask]
    
    if true_values.size == 0:
        raise ValueError("没有有效的(真实, 预测)配对样本用于计算指标")

    # --- 基础误差计算 ---
    diff = pred_values - true_values
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    # --- 决定系数 R² ---
    mean_true = float(np.mean(true_values))
    ss_total = float(np.sum((true_values - mean_true) ** 2))
    ss_residual = float(np.sum(diff ** 2))
    r2 = float(1 - (ss_residual / ss_total)) if ss_total != 0 else float('nan')

    # --- MAPE 计算 (处理分母为 0 的情况) ---
    # 找到真实值不为 0 的索引
    non_zero_mask = true_values > 2
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(diff[non_zero_mask] / true_values[non_zero_mask]))) * 100
    else:
        mape = float('nan')
        logger.warning("所有真实值均为0，无法计算MAPE")

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape  # 单位为 %
    }
    
    return metrics




def print_global_metrics(metrics: dict, logger=None):
    """
    以易读格式打印整体指标（MSE, MAE, RMSE, R²；若存在则附加国家电网准确率）。
    """
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("整体指标（不分站点）:")
    lines.append(f"MSE: {metrics['MSE']:.4f}")
    lines.append(f"MAE: {metrics['MAE']:.4f}")
    lines.append(f"RMSE: {metrics['RMSE']:.4f}")
    if 'StateGrid_Accuracy' in metrics:
        lines.append(f"国家电网准确率（容量法）: {metrics['StateGrid_Accuracy']:.4f}%")
    lines.append(f"R²: {metrics['R2']:.4f}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    if logger is None:
        print(text)
    else:
        logger.info(text)





def calculate_state_grid_accuracy(true_list, pred_list, capacity=None, logger=None):
    """
    国家电网准确率（容量法）:
      accu = (1 - RMSE / capacity) * 100, 四舍五入到 2 位小数

    说明:
      - 输入 true_list, pred_list 为反归一化后的实际功率序列（同单位）
      - capacity 为机组/电站额定容量（同单位）。若未提供，将回退为 max(|true|) 作为估计

    返回:
      {
        'StateGrid_Accuracy': float,
        'RMSE': float,
        'Capacity': float,
      }
    """
    import logging
    if logger is None:
        logger = logging.getLogger('grid_metrics')

    true_values = np.asarray(true_list, dtype=float)
    pred_values = np.asarray(pred_list, dtype=float)

    # 对齐有效样本
    mask = ~np.isnan(true_values) & ~np.isnan(pred_values)
    true_values = true_values[mask]
    pred_values = pred_values[mask]

    if true_values.size == 0:
        raise ValueError("没有有效的(真实, 预测)配对样本用于计算国家电网准确率")

    # 容量估计：若未显式提供，则使用真实值绝对最大值作为近似容量
    cap = capacity
    if cap is None or not np.isfinite(cap) or cap <= 0:
        cap = float(np.max(np.abs(true_values)) + 1e-6)

    # RMSE
    rmse = float(np.sqrt(np.mean((pred_values - true_values) ** 2)))

    # 国家电网准确率（容量法）
    accu = (1.0 - (rmse / cap)) * 100.0
    accu = round(accu, 2)
    # 裁剪到 [0, 100]
    accu = float(np.clip(accu, 0.0, 100.0))

    return {
        'StateGrid_Accuracy': accu,
        'RMSE': rmse,
        'Capacity': float(cap),
    }



def print_metrics_summary(daily_metrics_df, overall_metrics, logger=None):
    """
    打印7项核心指标的汇总信息（去除样本/NaN/月最大等统计）
    """
    if logger is None:
        import logging
        logger = logging.getLogger('temperature')

    summary_content = []
    summary_content.append("")
    summary_content.append("=" * 60)
    summary_content.append("总体指标:")
    summary_content.append(f"MSE: {overall_metrics['MSE']:.4f}")
    summary_content.append(f"MAE: {overall_metrics['MAE']:.4f}")
    summary_content.append(f"RMSE: {overall_metrics['RMSE']:.4f}")
    summary_content.append(f"Pearson相关系数: {overall_metrics['Pearson']:.4f}")
    summary_content.append(f"MRE: {overall_metrics['MRE']:.4f}")
    summary_content.append(f"MBE: {overall_metrics['MBE']:.4f}")
    summary_content.append(f"R²: {overall_metrics['R2']:.4f}")

    # 每日指标统计（忽略NaN）
    if len(daily_metrics_df) > 0:
        summary_content.append("=" * 60)
        summary_content.append("每日指标统计:")
        for col, name in [
            ('MSE', 'MSE'), ('MAE', 'MAE'), ('RMSE', 'RMSE'),
            ('Pearson', 'Pearson'), ('MRE', 'MRE'), ('MBE', 'MBE'), ('R2', 'R²')
        ]:
            col_series = daily_metrics_df[col]
            summary_content.append(
                f"{name} - 平均: {col_series.mean():.4f}, 最大: {col_series.max():.4f}, 最小: {col_series.min():.4f}")

    logger.info("\n".join(summary_content))



# 使用示例
if __name__ == "__main__":
    import logging

    # 设置日志
    logger = logging.getLogger('solar')

    # 使用函数
    csv_file_path = "pred_file_path.csv"  # 替换为你的文件路径

    try:
        station_metrics_df, overall_metrics = calculate_station_metrics(csv_file_path)

        # 使用日志记录每个站点的详细指标
        print_station_metrics(station_metrics_df, overall_metrics, logger)

        # 保存按站点的指标到文件
        station_metrics_df.to_csv("station_metrics.csv", index=False)
        logger.info("按站点指标已保存到 station_metrics.csv")

    except Exception as e:
        logger.error(f"处理文件时出错: {e}")
