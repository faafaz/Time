import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path




def plot_power_comparison(file_path, figsize=(15, 8)):
    """
    从CSV文件读取数据并绘制温度/气温和pred的对比折线图

    参数:
    file_path (str): CSV文件路径
    figsize (tuple): 图形大小，默认(15, 8)
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_csv(file_path, encoding='gbk')

    # 确保时间列是datetime格式并排序
    df['时间'] = pd.to_datetime(df['时间'])
    df = df.sort_values('时间').reset_index(drop=True)

    # 设置图形样式
    plt.style.use('default')
    sns.set_palette("husl")

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制折线图 - 学术风格
    sns.lineplot(data=df, x='时间', y='温度/气温',
                 label='Actual', linewidth=1.5, color='#2E86AB', ax=ax)
    sns.lineplot(data=df, x='时间', y='pred',
                 label='Predicted', linewidth=1.5, color='#A23B72', linestyle='--', ax=ax)

    # 设置标题和标签
    ax.set_title('Actual vs Predicted Temperature', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature', fontsize=12)

    # 设置图例和网格
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # 调整x轴标签角度
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    # 保存为高质量图片
    output_path = str(Path(file_path)).replace('.csv', '.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
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
    # save_dir = Path(model_save_path).parent
    save_dir = Path(model_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存为高质量图片
    loss_plot_path = save_dir / 'loss_curves.png'
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    # plt.show()


def save_predict(test_file_path, result_file_path, pred_list, seq_len, pred_len, plot=True):
    """
    读取CSV文件并添加预测列后保存

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
        plt.title("true vs pred")
        plt.plot(df['ACTIVEPOWER'], color='r', label='true')
        plt.plot(full_pred, label='pred')
        plt.legend(loc='best')
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(result_file_path, "pred.png"))

    # 添加预测列
    df['pred'] = full_pred

    # 生成新文件名
    new_path = os.path.join(result_file_path, "pred.csv")

    # 保存文件
    df.to_csv(new_path, index=False)
    return new_path


if __name__ == "__main__":
    # 模拟一些损失数据
    np.random.seed(42)
    epochs = 50

    # 生成模拟损失数据（递减趋势 + 噪声）
    train_loss = [2.5 * np.exp(-0.1 * i) + 0.1 * np.random.random() for i in range(epochs)]
    valid_loss = [2.6 * np.exp(-0.08 * i) + 0.15 * np.random.random() for i in range(epochs)]
    test_loss = [2.55 * np.exp(-0.09 * i) + 0.12 * np.random.random() for i in range(epochs)]

    model_save_path = "./models/best_model.pth"

    # 基础版本
    print("绘制基础版本损失曲线...")
    plot_loss(train_loss, valid_loss, test_loss, model_save_path)
