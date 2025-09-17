import pandas as pd
import numpy as np
from datetime import datetime


def calculate_daily_metrics(csv_file_path, logger=None):
    """
    按日计算MSE, MAE, RMSE, NSE, Accuracy指标

    Args:
        csv_file_path: CSV文件路径，包含时间, 温度/气温, pred列

    Returns:
        daily_metrics_df: 包含每日指标的DataFrame
        overall_metrics: 总体指标字典
    """
    if logger is None:
        import logging
        logger = logging.getLogger('solar')

    # 读取数据
    df = pd.read_csv(csv_file_path)

    # 转换时间列为datetime格式
    df['时间'] = pd.to_datetime(df['时间'])

    # 提取日期（年-月-日）和月份
    df['date'] = df['时间'].dt.strftime('%Y-%m-%d')  # 转换为字符串格式
    df['month'] = df['时间'].dt.to_period('M')  # 按月分组

    # 计算每月的最大功率值
    # monthly_max_power = df.groupby('month')['温度/气温'].max().to_dict()
    # 按日期分组计算指标
    daily_results = []
    for date, group in df.groupby('date'):
        # 统计NaN样本数
        nan_count = group['pred'].isna().sum()
        valid_count = len(group) - nan_count

        # 获取当月最大功率值
        month_key = group['month'].iloc[0]
        # cap = monthly_max_power.get(month_key, 0)
        cap = 150
        # 只对有效数据计算指标
        if valid_count > 0:
        # if valid_count == 24:
            # 过滤掉NaN值
            valid_data = group.dropna(subset=['pred', '温度/气温'])

            true_values = valid_data['温度/气温'].values
            pred_values = valid_data['pred'].values
            if len(true_values) > 0:
                # 计算各种指标
                mse = np.mean((pred_values - true_values) ** 2)
                mae = np.mean(np.abs(pred_values - true_values))
                rmse = np.sqrt(mse)

                # NSE (纳什效率系数)
                mean_true = np.mean(true_values)
                numerator = np.sum((pred_values - true_values) ** 2)
                # 防止分母为0
                # true_values = true_values + 1e-10
                denominator = np.sum((true_values - mean_true) ** 2)
                if denominator != 0:
                    nse = 1 - (numerator / denominator)
                else:
                    print(date)
                    print(true_values)
                    print(f"mean_true:{mean_true}")
                    nse = float('-inf')

                # Accuracy (如果提供了计算函数)
                pred_list = pred_values.tolist()

                # accuracy = calculate_acc_4h(true_values.tolist(), pred_list, cap)
                accuracy = calculate_acc(true_values.tolist(), pred_list)
                if accuracy is None:
                    logger.info(f"date:{date},样本:{len(group)},有效样本：{len(valid_data)},accuracy:{accuracy}")
            else:
                mse = mae = rmse = nse = accuracy = float('nan')
        else:
            mse = mae = rmse = nse = accuracy = float('nan')
            true_values = pred_values = []

        daily_results.append({
            'date': date,
            'total_count': len(group),
            'nan_count': nan_count,
            'valid_count': valid_count,
            'monthly_cap': cap,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'NSE': nse,
            'Accuracy': accuracy,
            'mean_true': np.mean(group['温度/气温']) if len(group) > 0 else float('nan'),
            'mean_pred': np.mean(group['pred'].dropna()) if group['pred'].dropna().size > 0 else float('nan')
        })

    # 转换为DataFrame
    daily_metrics_df = pd.DataFrame(daily_results)

    # 计算总体指标（使用每天指标的平均值，只使用有效数据）
    valid_daily_df = daily_metrics_df[daily_metrics_df['nan_count'] == 0]

    if len(valid_daily_df) > 0:
        overall_mse = valid_daily_df['MSE'].mean()
        overall_mae = valid_daily_df['MAE'].mean()
        overall_rmse = valid_daily_df['RMSE'].mean()
        overall_nse = valid_daily_df['NSE'].mean()
        overall_accuracy = valid_daily_df['Accuracy'].mean()  # 假设你的daily_results中有accuracy列
    else:
        print("没有有效的日数据用于计算总体指标")

    overall_metrics = {
        'total_samples': len(df),
        'total_nan_samples': df['pred'].isna().sum(),
        'total_valid_samples': f"{len(valid_daily_df)}天",
        'total_days': len(daily_metrics_df),
        'MSE': overall_mse,
        'MAE': overall_mae,
        'RMSE': overall_rmse,
        'NSE': overall_nse,
        'Accuracy': overall_accuracy
    }

    return daily_metrics_df, overall_metrics


def print_daily_metrics(daily_metrics_df, overall_metrics, logger=None):
    """
    打印每天的详细指标
    """
    if logger is None:
        import logging
        logger = logging.getLogger('solar')

    # 构建完整的表格内容
    table_content = []
    table_content.append("")  # 先换行
    table_content.append("=" * 120)
    table_content.append("每日指标详情:")
    table_content.append("=" * 120)
    table_content.append(
        f"{'日期':<12} {'总样本':<8} {'NaN数':<8} {'有效数':<8} {'月最大':<10} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'NSE':<10} {'Accuracy':<10}")
    table_content.append("-" * 120)

    for _, row in daily_metrics_df.iterrows():
        # 确保日期格式正确显示
        date_str = str(row['date'])
        if pd.isna(row['MSE']):
            table_content.append(
                f"{date_str:<12} {row['total_count']:<8} {row['nan_count']:<8} {row['valid_count']:<8} {row['monthly_cap']:<10.2f} {'--':<10} {'--':<10} {'--':<10} {'--':<10} {'--':<10}")
        else:
            acc_str = f"{row['Accuracy']:.4f}" if not pd.isna(row['Accuracy']) else "--"
            table_content.append(
                f"{date_str:<12} {row['total_count']:<8} {row['nan_count']:<8} {row['valid_count']:<8} {row['monthly_cap']:<10.2f} {row['MSE']:<10.4f} {row['MAE']:<10.4f} {row['RMSE']:<10.4f} {row['NSE']:<10.4f} {acc_str:<10}")

    table_content.append("-" * 120)
    # print(overall_metrics['Accuracy'])
    overall_acc_str = f"{overall_metrics['Accuracy']:.4f}" if not pd.isna(overall_metrics['Accuracy']) else "--"
    table_content.append(
        f"{'总体指标':<12} {overall_metrics['total_samples']:<8} {overall_metrics['total_nan_samples']:<8} {overall_metrics['total_valid_samples']:<8} {'--':<10} {overall_metrics['MSE']:<10.4f} {overall_metrics['MAE']:<10.4f} {overall_metrics['RMSE']:<10.4f} {overall_metrics['NSE']:<10.4f} {overall_acc_str:<10}")
    table_content.append("=" * 120)

    # 一次性输出完整表格
    logger.info("\n".join(table_content))


def print_metrics_summary(daily_metrics_df, overall_metrics, logger=None):
    """
    打印指标汇总信息
    """
    if logger is None:
        import logging
        logger = logging.getLogger('temperature')

    # 构建完整的汇总内容
    summary_content = []
    summary_content.append("")  # 先换行
    summary_content.append("=" * 60)
    summary_content.append("总体指标:")
    summary_content.append(f"总样本数: {overall_metrics['total_samples']}")
    summary_content.append(f"NaN样本数: {overall_metrics['total_nan_samples']}")
    summary_content.append(f"有效样本数: {overall_metrics['total_valid_samples']}")
    summary_content.append(f"总天数: {overall_metrics['total_days']}")
    summary_content.append(f"MSE: {overall_metrics['MSE']:.4f}")
    summary_content.append(f"MAE: {overall_metrics['MAE']:.4f}")
    summary_content.append(f"RMSE: {overall_metrics['RMSE']:.4f}")
    summary_content.append(f"NSE: {overall_metrics['NSE']:.4f}")
    if not pd.isna(overall_metrics['Accuracy']):
        summary_content.append(f"Accuracy: {overall_metrics['Accuracy']:.4f}")
    else:
        summary_content.append("Accuracy: --")

    # 只对有有效数据的天数进行统计
    valid_days = daily_metrics_df[daily_metrics_df['valid_count'] > 0]

    if len(valid_days) > 0:
        summary_content.append("=" * 60)
        summary_content.append("每日指标统计 (仅包含有效预测的天数):")
        summary_content.append(f"有效天数: {len(valid_days)}")
        summary_content.append(
            f"MSE - 平均: {valid_days['MSE'].mean():.4f}, 最大: {valid_days['MSE'].max():.4f}, 最小: {valid_days['MSE'].min():.4f}")
        summary_content.append(
            f"MAE - 平均: {valid_days['MAE'].mean():.4f}, 最大: {valid_days['MAE'].max():.4f}, 最小: {valid_days['MAE'].min():.4f}")
        summary_content.append(
            f"RMSE - 平均: {valid_days['RMSE'].mean():.4f}, 最大: {valid_days['RMSE'].max():.4f}, 最小: {valid_days['RMSE'].min():.4f}")
        summary_content.append(
            f"NSE - 平均: {valid_days['NSE'].mean():.4f}, 最大: {valid_days['NSE'].max():.4f}, 最小: {valid_days['NSE'].min():.4f}")

        # Accuracy统计
        valid_acc_days = valid_days.dropna(subset=['Accuracy'])
        if len(valid_acc_days) > 0:
            summary_content.append(
                f"Accuracy - 平均: {valid_acc_days['Accuracy'].mean():.4f}, 最大: {valid_acc_days['Accuracy'].max():.4f}, 最小: {valid_acc_days['Accuracy'].min():.4f}")
        else:
            summary_content.append("Accuracy - 无有效数据")

    # 统计NaN情况
    summary_content.append("=" * 60)
    summary_content.append("NaN数据统计:")
    summary_content.append(f"完全没有预测的天数: {len(daily_metrics_df[daily_metrics_df['valid_count'] == 0])}")
    summary_content.append(
        f"部分有预测的天数: {len(daily_metrics_df[(daily_metrics_df['valid_count'] > 0) & (daily_metrics_df['nan_count'] > 0)])}")
    summary_content.append(f"完全有预测的天数: {len(daily_metrics_df[daily_metrics_df['nan_count'] == 0])}")

    # 一次性输出完整汇总
    logger.info("\n".join(summary_content))


# 定义你的accuracy计算函数
def calculate_acc_4h(true_list, pred_list, Cap):
    """
    计算风电场超短期功率预测的4小时前预测准确率（按日统计，按月考核）
    参数:
        true_list (list[float]): 长度为96的列表，表示每个时刻的功率值。
                              - 限电时刻: 可用功率
                              - 不限电时刻: 实际功率
                              注：目前不区分限电与不限电，统一用实际功率
        pred_list (list[float]): 长度为96的列表，表示4小时前预测的功率值
        Cap (float): 风电场当月装机容量（单位与功率值一致） 即 当月的最大值
    返回:
        float: 预测准确率百分比（取值范围可能超过100%，需根据实际情况处理）
    异常:
        ValueError: 输入列表长度不为96时触发
    """
    if len(true_list) != 96 or len(pred_list) != 96:
        return None

    sum_squares = 0.0

    # 遍历所有时刻（15分钟间隔，共96个点）
    for i in range(96):
        ti = true_list[i]
        pi = pred_list[i]

        # 根据当前时刻实际功率选择计算公式
        if ti >= 0.2 * Cap:
            # 使用实际功率归一化
            denominator = ti
        else:
            # 使用装机容量的20%归一化
            denominator = 0.2 * Cap

        # 计算归一化误差平方项
        error_term = (ti - pi) / denominator
        sum_squares += error_term ** 2

    # 计算均方根误差（RMSE）
    # 注：原公式中sqrt(96/96*sum(...))等效于sqrt(sum(...))，此处采用更合理的归一化方式
    rmse = (sum_squares / 96) ** 0.5  # 平均误差计算

    # 计算准确率百分比
    accuracy = (1 - rmse) * 100

    return accuracy

def calculate_acc(true_list, pred_list):
    """
    计算预测准确率（简化版）
    
    参数:
        true_list (list[float]): 实际值列表
        pred_list (list[float]): 预测值列表
        
    返回:
        float: 预测准确率百分比
    """
    if len(true_list) != len(pred_list):
        return None
    
    if len(true_list) == 0:
        return None
    
    # 过滤掉NaN值
    valid_pairs = [(t, p) for t, p in zip(true_list, pred_list) 
                   if not (pd.isna(t) or pd.isna(p))]
    
    if len(valid_pairs) == 0:
        return None
    
    true_vals, pred_vals = zip(*valid_pairs)
    
    # 使用1-MAPE方法计算准确率，避免除零错误
    errors = []
    for t, p in zip(true_vals, pred_vals):
        # 如果真实值为0，使用一个小的阈值来避免除零
        if abs(t) < 1e-8:  # 当温度接近0时
            if abs(p) < 1e-8:  # 预测值也接近0
                error = 0.0  # 完美预测
            else:
                error = abs(p) / 1.0  # 使用1作为基准值计算相对误差
        else:
            error = abs((t - p) / t)
        errors.append(error)
    
    if len(errors) == 0:
        return None
    
    mape = np.mean(errors)
    
    # 准确率 = (1 - MAPE) * 100
    accuracy = (1 - mape) * 100
    
    return accuracy


# 使用示例
if __name__ == "__main__":
    import logging

    # 设置日志
    logger = logging.getLogger('solar')

    # 使用函数
    csv_file_path = "pred_file_path.csv"  # 替换为你的文件路径

    try:
        daily_metrics_df, overall_metrics = calculate_daily_metrics(csv_file_path)

        # 使用日志记录每天的详细指标
        print_daily_metrics(daily_metrics_df, overall_metrics, logger)

        # 如果需要，也可以记录汇总信息
        # print_metrics_summary(daily_metrics_df, overall_metrics, logger)

        # 保存每日指标到文件
        daily_metrics_df.to_csv("daily_metrics.csv", index=False)
        logger.info("每日指标已保存到 daily_metrics.csv")

    except Exception as e:
        logger.error(f"处理文件时出错: {e}")
