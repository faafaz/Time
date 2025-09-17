import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_ice_cream_data(start_date='2023-01-01', days=365):
    """生成虚构的雪糕销售数据"""

    data = []
    start = pd.to_datetime(start_date)

    for day in range(days):
        current_date = start + timedelta(days=day)
        month = current_date.month
        day_of_week = current_date.weekday()  # 0=周一, 6=周日
        day_of_month = current_date.day

        # 生成一天24小时的数据
        for hour in range(24):
            # 基础销量
            base_sales = 10

            # 季节性影响：夏天销量高，冬天销量低
            season_effect = np.sin((month - 1) * np.pi / 6) * 0.8 + 1.2

            # 一天中的时间模式：下午和傍晚销量高
            if 6 <= hour <= 11:  # 上午
                time_effect = 0.6
            elif 12 <= hour <= 18:  # 下午（高峰）
                time_effect = 1.5 + 0.3 * np.sin((hour - 12) * np.pi / 6)
            elif 19 <= hour <= 22:  # 晚上（次高峰）
                time_effect = 1.2
            else:  # 深夜和凌晨
                time_effect = 0.3

            # 周末效应：周末销量高
            if day_of_week >= 5:  # 周六周日
                weekend_bonus = 1.4
            else:
                weekend_bonus = 1.0

            # 月末发工资效应
            if day_of_month > 25:
                payday_bonus = 1.1
            else:
                payday_bonus = 1.0

            # 天气随机效应
            weather_effect = np.random.normal(1.0, 0.2)

            # 计算最终销量
            sales = base_sales * season_effect * time_effect * weekend_bonus * payday_bonus * weather_effect
            sales = max(1, sales)  # 最少1支

            data.append({
                'datetime': current_date + timedelta(hours=hour),
                'month': month,
                'day': day_of_month,
                'weekday': day_of_week,
                'hour': hour,
                'sales': round(sales, 1)
            })

    return pd.DataFrame(data)


class NumericalModel(nn.Module):
    """使用数值特征的模型"""

    def __init__(self, input_size=4, hidden_size=64, output_size=1):
        super(NumericalModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
            nn.ReLU()  # 确保销量为正
        )

    def forward(self, x):
        return self.network(x)


class EmbeddingModel(nn.Module):
    """使用时间嵌入的模型"""

    def __init__(self, hidden_size=64, output_size=1):
        super(EmbeddingModel, self).__init__()
        embed_dim = 128
        # 嵌入层
        self.month_embed = nn.Embedding(13, embed_dim)  # 月份 1-12 (索引0不用)
        self.day_embed = nn.Embedding(32, embed_dim)  # 日期 1-31 (索引0不用)
        self.weekday_embed = nn.Embedding(7, embed_dim)  # 星期 0-6
        self.hour_embed = nn.Embedding(24, embed_dim)  # 小时 0-23

        # 计算总的嵌入维度
        # embed_dim = 8 + 8 + 4 + 8  # 28
        # 神经网络
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
            nn.ReLU()
        )

    def forward(self, month, day, weekday, hour):
        # 获取嵌入向量
        month_emb = self.month_embed(month)
        day_emb = self.day_embed(day)
        weekday_emb = self.weekday_embed(weekday)
        hour_emb = self.hour_embed(hour)

        # 拼接所有嵌入向量
        # combined = torch.cat([month_emb, day_emb, weekday_emb, hour_emb], dim=1)
        combined = month_emb + day_emb + weekday_emb + hour_emb
        return self.network(combined)


def prepare_data(df):
    """准备训练数据"""

    # 数值特征（标准化）
    numerical_features = ['month', 'day', 'weekday', 'hour']
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(df[numerical_features])

    # 嵌入特征（原始整数）
    X_embedding = df[['month', 'day', 'weekday', 'hour']].values

    # 目标变量
    y = df['sales'].values

    return X_numerical, X_embedding, y, scaler


def train_models(X_numerical, X_embedding, y, epochs=200, lr=0.001):
    """训练两个模型"""

    # 分割数据
    X_num_train, X_num_test, X_emb_train, X_emb_test, y_train, y_test = train_test_split(
        X_numerical, X_embedding, y, test_size=0.2, random_state=42
    )

    # 转换为tensor
    X_num_train = torch.FloatTensor(X_num_train)
    X_num_test = torch.FloatTensor(X_num_test)
    X_emb_train = torch.LongTensor(X_emb_train)
    X_emb_test = torch.LongTensor(X_emb_test)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    # 创建模型
    numerical_model = NumericalModel()
    embedding_model = EmbeddingModel()

    # 优化器
    optimizer_num = optim.Adam(numerical_model.parameters(), lr=lr)
    optimizer_emb = optim.Adam(embedding_model.parameters(), lr=lr)

    # 损失函数
    criterion = nn.MSELoss()

    # 训练历史
    history = {
        'num_train_loss': [],
        'num_test_loss': [],
        'emb_train_loss': [],
        'emb_test_loss': []
    }

    print("开始训练模型...")

    for epoch in range(epochs):
        # 训练数值模型
        numerical_model.train()
        optimizer_num.zero_grad()
        num_pred_train = numerical_model(X_num_train)
        num_loss_train = criterion(num_pred_train, y_train)
        num_loss_train.backward()
        optimizer_num.step()

        # 训练嵌入模型
        embedding_model.train()
        optimizer_emb.zero_grad()
        emb_pred_train = embedding_model(
            X_emb_train[:, 0], X_emb_train[:, 1],
            X_emb_train[:, 2], X_emb_train[:, 3]
        )
        emb_loss_train = criterion(emb_pred_train, y_train)
        emb_loss_train.backward()
        optimizer_emb.step()

        # 验证
        if epoch % 10 == 0:
            with torch.no_grad():
                numerical_model.eval()
                embedding_model.eval()

                # 测试数值模型
                num_pred_test = numerical_model(X_num_test)
                num_loss_test = criterion(num_pred_test, y_test)

                # 测试嵌入模型
                emb_pred_test = embedding_model(
                    X_emb_test[:, 0], X_emb_test[:, 1],
                    X_emb_test[:, 2], X_emb_test[:, 3]
                )
                emb_loss_test = criterion(emb_pred_test, y_test)

                # 记录历史
                history['num_train_loss'].append(num_loss_train.item())
                history['num_test_loss'].append(num_loss_test.item())
                history['emb_train_loss'].append(emb_loss_train.item())
                history['emb_test_loss'].append(emb_loss_test.item())

                if epoch % 50 == 0:
                    print(f"Epoch {epoch:3d} | "
                          f"Num: {num_loss_test.item():.3f} | "
                          f"Emb: {emb_loss_test.item():.3f}")

    return numerical_model, embedding_model, history, (X_num_test, X_emb_test, y_test)


def visualize_results(df, numerical_model, embedding_model, history, test_data, scaler):
    """可视化结果"""

    X_num_test, X_emb_test, y_test = test_data

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🍦 雪糕销售时间嵌入学习效果对比', fontsize=16, fontweight='bold')

    # 1. 原始数据模式可视化
    ax = axes[0, 0]
    hourly_avg = df.groupby('hour')['sales'].mean()
    ax.plot(hourly_avg.index, hourly_avg.values, 'o-', linewidth=2, markersize=6)
    ax.set_title('每小时平均销量模式', fontweight='bold')
    ax.set_xlabel('小时')
    ax.set_ylabel('平均销量')
    ax.grid(True, alpha=0.3)

    # 2. 季节性模式
    ax = axes[0, 1]
    monthly_avg = df.groupby('month')['sales'].mean()
    ax.bar(monthly_avg.index, monthly_avg.values, color='skyblue', alpha=0.7)
    ax.set_title('每月平均销量模式', fontweight='bold')
    ax.set_xlabel('月份')
    ax.set_ylabel('平均销量')
    ax.grid(True, alpha=0.3)

    # 3. 周内模式
    ax = axes[0, 2]
    weekday_avg = df.groupby('weekday')['sales'].mean()
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    ax.bar(range(7), weekday_avg.values, color='lightgreen', alpha=0.7)
    ax.set_title('星期销量模式', fontweight='bold')
    ax.set_xlabel('星期')
    ax.set_ylabel('平均销量')
    ax.set_xticks(range(7))
    ax.set_xticklabels(weekday_names, rotation=45)
    ax.grid(True, alpha=0.3)

    # 4. 训练损失对比
    ax = axes[1, 0]
    epochs_plot = range(0, len(history['num_train_loss']) * 10, 10)
    ax.plot(epochs_plot, history['num_train_loss'], 'b-', label='数值模型-训练', linewidth=2)
    ax.plot(epochs_plot, history['num_test_loss'], 'b--', label='数值模型-测试', linewidth=2)
    ax.plot(epochs_plot, history['emb_train_loss'], 'r-', label='嵌入模型-训练', linewidth=2)
    ax.plot(epochs_plot, history['emb_test_loss'], 'r--', label='嵌入模型-测试', linewidth=2)
    ax.set_title('训练损失对比', fontweight='bold')
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('MSE损失')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 预测效果对比
    ax = axes[1, 1]
    with torch.no_grad():
        numerical_model.eval()
        embedding_model.eval()

        num_pred = numerical_model(X_num_test).numpy().flatten()
        emb_pred = embedding_model(
            X_emb_test[:, 0], X_emb_test[:, 1],
            X_emb_test[:, 2], X_emb_test[:, 3]
        ).numpy().flatten()

        y_true = y_test.numpy().flatten()

        # 选择前100个样本进行可视化
        n_samples = min(100, len(y_true))
        indices = np.random.choice(len(y_true), n_samples, replace=False)

        ax.scatter(y_true[indices], num_pred[indices], alpha=0.6, label='数值模型', color='blue')
        ax.scatter(y_true[indices], emb_pred[indices], alpha=0.6, label='嵌入模型', color='red')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.5)
        ax.set_title('预测 vs 真实值', fontweight='bold')
        ax.set_xlabel('真实销量')
        ax.set_ylabel('预测销量')
        # ax.legend()
        ax.grid(True, alpha=0.3)

        # 计算误差指标
        num_mae = mean_absolute_error(y_true, num_pred)
        emb_mae = mean_absolute_error(y_true, emb_pred)

        ax.text(0.05, 0.95, f'数值模型 MAE: {num_mae:.2f}\n嵌入模型 MAE: {emb_mae:.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 6. 学习到的小时嵌入可视化
    ax = axes[1, 2]
    with torch.no_grad():
        hour_embeddings = embedding_model.hour_embed.weight.numpy()

        # 使用前两个维度进行可视化
        x_coords = hour_embeddings[:, 0]
        y_coords = hour_embeddings[:, 1]

        # 根据销量给小时着色
        hourly_sales = df.groupby('hour')['sales'].mean().values
        scatter = ax.scatter(x_coords, y_coords, c=hourly_sales, cmap='viridis', s=100)

        # 添加小时标签
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.annotate(f'{i}h', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_title('学习到的小时嵌入向量 (2D投影)', fontweight='bold')
        ax.set_xlabel('嵌入维度 1')
        ax.set_ylabel('嵌入维度 2')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('平均销量')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_embeddings(embedding_model, df):
    """分析学习到的嵌入向量"""

    print("\n" + "=" * 50)
    print("🔍 嵌入向量分析")
    print("=" * 50)

    with torch.no_grad():
        # 分析小时嵌入
        hour_embeddings = embedding_model.hour_embed.weight.numpy()

        print("\n📊 小时嵌入向量相似性分析:")
        print("-" * 30)

        # 计算嵌入向量之间的余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        hour_sim = cosine_similarity(hour_embeddings)

        # 找出最相似的时间对
        similar_pairs = []
        for i in range(24):
            for j in range(i + 1, 24):
                similarity = hour_sim[i, j]
                similar_pairs.append((i, j, similarity))

        # 排序并显示最相似的前5对
        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        print("最相似的时间对 (余弦相似度):")
        for i, (h1, h2, sim) in enumerate(similar_pairs[:5]):
            avg_sales_h1 = df[df['hour'] == h1]['sales'].mean()
            avg_sales_h2 = df[df['hour'] == h2]['sales'].mean()
            print(f"{i + 1}. {h1:2d}点 & {h2:2d}点: {sim:.3f} "
                  f"(销量: {avg_sales_h1:.1f} & {avg_sales_h2:.1f})")

        print("\n最不相似的时间对:")
        for i, (h1, h2, sim) in enumerate(similar_pairs[-5:]):
            avg_sales_h1 = df[df['hour'] == h1]['sales'].mean()
            avg_sales_h2 = df[df['hour'] == h2]['sales'].mean()
            print(f"{i + 1}. {h1:2d}点 & {h2:2d}点: {sim:.3f} "
                  f"(销量: {avg_sales_h1:.1f} & {avg_sales_h2:.1f})")

        # 分析月份嵌入
        print("\n📅 月份嵌入向量分析:")
        print("-" * 30)

        month_embeddings = embedding_model.month_embed.weight.numpy()[1:]  # 跳过索引0
        month_sim = cosine_similarity(month_embeddings)

        month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                       '7月', '8月', '9月', '10月', '11月', '12月']

        # 找出夏季月份的相似性
        summer_months = [5, 6, 7]  # 6月、7月、8月 (索引从0开始)
        winter_months = [11, 0, 1]  # 12月、1月、2月

        print("夏季月份(6-8月)相似度:")
        for i in summer_months:
            for j in summer_months:
                if i < j:
                    print(f"{month_names[i]} & {month_names[j]}: {month_sim[i, j]:.3f}")

        print("\n冬季月份(12-2月)相似度:")
        for i in winter_months:
            for j in winter_months:
                if i < j:
                    print(f"{month_names[i]} & {month_names[j]}: {month_sim[i, j]:.3f}")


def main():
    """主函数"""

    print("🍦 雪糕销售时间嵌入学习演示")
    print("=" * 50)

    # 1. 生成数据
    print("📊 生成虚构销售数据...")
    df = generate_ice_cream_data()
    print(f"生成了 {len(df)} 条记录")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"销量范围: {df['sales'].min():.1f} 到 {df['sales'].max():.1f}")

    # 2. 数据预处理
    print("\n🔧 数据预处理...")
    X_numerical, X_embedding, y, scaler = prepare_data(df)

    # 3. 训练模型
    print("\n🚀 训练模型...")
    numerical_model, embedding_model, history, test_data = train_models(
        X_numerical, X_embedding, y, epochs=200
    )

    # 4. 评估模型
    print("\n📈 模型评估...")
    X_num_test, X_emb_test, y_test = test_data

    with torch.no_grad():
        numerical_model.eval()
        embedding_model.eval()

        num_pred = numerical_model(X_num_test).numpy().flatten()
        emb_pred = embedding_model(
            X_emb_test[:, 0], X_emb_test[:, 1],
            X_emb_test[:, 2], X_emb_test[:, 3]
        ).numpy().flatten()

        y_true = y_test.numpy().flatten()

        num_mae = mean_absolute_error(y_true, num_pred)
        emb_mae = mean_absolute_error(y_true, emb_pred)
        num_rmse = np.sqrt(mean_squared_error(y_true, num_pred))
        emb_rmse = np.sqrt(mean_squared_error(y_true, emb_pred))

        print(f"数值特征模型 - MAE: {num_mae:.3f}, RMSE: {num_rmse:.3f}")
        print(f"嵌入特征模型 - MAE: {emb_mae:.3f}, RMSE: {emb_rmse:.3f}")

        improvement = (num_mae - emb_mae) / num_mae * 100
        print(f"嵌入模型相对改进: {improvement:.1f}%")

    # 5. 可视化结果
    print("\n📊 生成可视化图表...")
    visualize_results(df, numerical_model, embedding_model, history, test_data, scaler)

    # 6. 分析嵌入向量
    analyze_embeddings(embedding_model, df)

    print("\n✅ 演示完成！")
    print("通过对比可以看出：")
    print("1. 嵌入模型能够学习到更复杂的时间模式")
    print("2. 相似的时间点在嵌入空间中更加接近")
    print("3. 在复杂的时间序列数据上，嵌入模型通常表现更好")


if __name__ == "__main__":
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    np.random.seed(42)

    main()