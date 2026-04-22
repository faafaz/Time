import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_power_comparison(file_path, figsize=(15, 8)):
    """
    从CSV读取后绘制Power与pred对比图（全局，不分站点）。

    【改进】
    - 删除按站点分组的逻辑，直接全局绘制
    - 只绘制预测值非空的数据点
    """
    # 中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 读取CSV
    try:
        df = pd.read_csv(file_path)
    except Exception:
        df = pd.read_csv(file_path, encoding='gbk')

    # 统一样式
    plt.style.use('default')
    sns.set_palette("husl")

    # 列存在性判断
    time_col = 'Time'
    actual_col = 'Power'
    pred_col = 'pred'

    if time_col not in df.columns or actual_col not in df.columns or pred_col not in df.columns:
        raise KeyError("缺少必要列：Time/Power/pred")

    # 全局绘制（不分站点）
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # 【关键改进】只保留pred列有值的行
    df_valid = df[df[pred_col].notna()].copy()

    if df_valid.empty:
        raise ValueError("错误：没有有效的预测数据可以绘图")

    fig, ax = plt.subplots(figsize=figsize)

    # 使用过滤后的数据绘图
    sns.lineplot(data=df_valid, x=time_col, y=actual_col,
                 label='Actual', linewidth=1.5, color='#2E86AB', ax=ax)
    sns.lineplot(data=df_valid, x=time_col, y=pred_col,
                 label='Predicted', linewidth=1.5, color='#A23B72', linestyle='--', ax=ax)

    ax.set_title(f'Actual vs Predicted Power - 有效数据点: {len(df_valid)}',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = str(Path(file_path)).replace('.csv', '.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig, ax


def plot_loss(train_loss, valid_loss, test_loss, model_save_path):
    """
    使用seaborn绘制优美的训练损失折线图

    Args:
        train_loss: 训练损失列表
        valid_loss: 验证损失列表
        test_loss: 测试损失列表
        model_save_path: 模型保存路径
    """
    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # 创建图形和子图
    plt.figure(figsize=(12, 8))

    # 准备数据
    epochs = range(1, len(train_loss) + 1)

    # 创建DataFrame便于seaborn处理
    data_dict = {
        'Epoch': list(epochs) * 3,
        'Loss': train_loss + valid_loss + test_loss,
        'Type': ['Training'] * len(train_loss) +
                ['Validation'] * len(valid_loss) +
                ['Test'] * len(test_loss)
    }
    df = pd.DataFrame(data_dict)

    # 绘制折线图
    ax = sns.lineplot(data=df, x='Epoch', y='Loss', hue='Type',
                      linewidth=2.5, marker='o', markersize=6)

    # 美化图形
    plt.title('Model Loss Curves', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Loss', fontsize=16, fontweight='bold')

    # 设置图例
    plt.legend(title='Loss Type', title_fontsize=14, fontsize=12,
               loc='upper right', frameon=True, shadow=True)

    # 设置刻度标签大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')

    # 调整布局
    plt.tight_layout()

    # 保存图形
    save_dir = Path(model_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存为高质量图片
    loss_plot_path = save_dir / 'loss_curves.png'
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')


def save_predict(test_file_path, result_file_path, pred_list, seq_len, pred_len, plot=True):
    """
    读取CSV文件并添加预测列后保存
    
    【改进】绘图时只显示有效的预测值

    参数:
        test_file_path: 原始CSV文件路径
        result_file_path: 结果保存的路径
        pred_list: 要添加的预测值列表
        seq_len: 序列长度(需要验证的行数差)
        pred_len: 预测长度
    """
    # 读取原始数据
    df = pd.read_csv(test_file_path)

    # 应当预测的次数 也是预测出来的数据个数
    num_pred_times = len(df) - seq_len - pred_len + 1

    # 验证数据长度
    if num_pred_times != len(pred_list):
        raise ValueError(
            f"数据长度不匹配: 应该预测出行数：{num_pred_times},pred_list行数({len(pred_list)})")

    # 创建带None填充的完整预测列
    full_pred = [None] * seq_len + pred_list + (pred_len - 1) * [None]

    if plot:
        plt.figure(figsize=(20, 8), dpi=100)
        plt.title("True vs Pred (仅显示有效预测值)", fontsize=16, fontweight='bold')
        
        # 绘制真实值（完整）
        plt.plot(range(len(df)), df['ACTIVEPOWER'], color='r', label='True', linewidth=2)
        
        # 【关键改进】只绘制非None的预测值
        # 找出pred不为None的索引和值
        valid_indices = [i for i, val in enumerate(full_pred) if val is not None]
        valid_preds = [val for val in full_pred if val is not None]
        
        # 绘制有效预测值
        plt.plot(valid_indices, valid_preds, 'o-', color='blue', 
                label=f'Pred (有效点数: {len(valid_preds)})', linewidth=1.5, markersize=3)
        
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('ACTIVEPOWER', fontsize=12)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(result_file_path, "pred.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # 添加预测列
    df['pred'] = full_pred

    # 生成新文件名
    new_path = os.path.join(result_file_path, "pred.csv")

    # 保存文件
    df.to_csv(new_path, index=False)
    
    print(f"预测结果已保存到: {new_path}")
    print(f"总行数: {len(df)}, 有效预测值: {len(pred_list)}")
    
    return new_path


def calculate_metrics(actual, predicted):
    """
    计算预测指标，仅使用预测值非空的数据
    
    参数:
        actual: 实际值数组或列表
        predicted: 预测值数组或列表
    
    返回:
        dict: 包含各种指标的字典
    """
    # 转换为numpy数组
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 【关键改进】创建mask，只选择预测值非空的位置
    mask = ~np.isnan(predicted) & (predicted is not None)
    if isinstance(predicted[0], (int, float)):
        # 如果是数值类型，进一步过滤
        mask = mask & np.isfinite(predicted)
    
    # 过滤数据
    actual_valid = actual[mask]
    predicted_valid = predicted[mask]
    
    if len(actual_valid) == 0:
        raise ValueError("没有有效的预测数据用于计算指标")
    
    # 计算各种指标
    mse = np.mean((actual_valid - predicted_valid) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_valid - predicted_valid))
    mape = np.mean(np.abs((actual_valid - predicted_valid) / actual_valid)) * 100
    
    # R² 分数
    ss_res = np.sum((actual_valid - predicted_valid) ** 2)
    ss_tot = np.sum((actual_valid - np.mean(actual_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'valid_points': len(actual_valid),
        'total_points': len(actual)
    }
    
    return metrics


if __name__ == "__main__":
    # 模拟一些损失数据
    np.random.seed(42)
    epochs = 50

    # 生成模拟损失数据（递减趋势 + 噪声）
    train_loss = [2.5 * np.exp(-0.1 * i) + 0.1 * np.random.random() for i in range(epochs)]
    valid_loss = [2.6 * np.exp(-0.08 * i) + 0.15 * np.random.random() for i in range(epochs)]
    test_loss = [2.55 * np.exp(-0.09 * i) + 0.12 * np.random.random() for i in range(epochs)]

    model_save_path = "./models/best_model.pth"

    print("绘制损失曲线...")
    plot_loss(train_loss, valid_loss, test_loss, model_save_path)
    
    # 测试指标计算（仅使用有效预测值）
    print("\n测试指标计算...")
    actual = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    predicted = np.array([None, None, 2.9, 3.8, 5.2, 6.1, None, None])
    predicted = np.array([np.nan if x is None else x for x in predicted])
    
    metrics = calculate_metrics(actual, predicted)
    print(f"有效数据点: {metrics['valid_points']}/{metrics['total_points']}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"R²: {metrics['R2']:.4f}")