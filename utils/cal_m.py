import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(file_path):
    # 1. 读取数据
    try:
        # 尝试常用编码读取
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='gbk')

    # 2. 清洗数据：只保留 Power 和 pred 都不为空的行
    # 注意：根据你的 CSV 示例，存在一些行 pred 为空，这些行无法参与指标计算
    clean_df = df.dropna(subset=['Power', 'pred']).copy()
    
    if clean_df.empty:
        print("错误：没有找到有效的真实值(Power)与预测值(pred)配对数据。")
        return

    # 提取数组
    y_true = clean_df['Power'].values
    y_pred = clean_df['pred'].values

    # 3. 计算各项指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # 4. 打印结果
    print("-" * 30)
    print(f"数据统计结果 (样本量: {len(clean_df)})")
    print("-" * 30)
    print(f"MSE  (均方误差):     {mse:.4f}")
    print(f"MAE  (平均绝对误差):  {mae:.4f}")
    print(f"RMSE (均方根误差):    {rmse:.4f}")
    print(f"R²   (决定系数):      {r2:.4f}")
    print("-" * 30)

    # 简单分析
    if r2 < 0:
        print("警告：R² 为负数，说明模型预测效果甚至不如直接取真实值的平均值。")
    elif r2 > 0.9:
        print("提示：R² 较高，模型拟合效果优秀。")

if __name__ == "__main__":
    # 请确保该文件在当前目录下，或者填写绝对路径
    file_path = r"checkpoints\TimeXer\20260305_152942_TimeXer\20260305_162642_test\TimeXer_pred.csv" 
    calculate_metrics(file_path)