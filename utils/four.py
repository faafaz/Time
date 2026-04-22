# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1. Load Data
# # 假设你的文件名为 data.csv
# df = pd.read_csv(r'dataset\china_wind_power\data_processed\wind_farms\Wind farm site 2.csv')

# # 2. Preprocessing: Convert Time and filter 2019 data
# df['Time'] = pd.to_datetime(df['Time'])
# df_2019 = df[df['Time'].dt.year == 2019].copy()

# # 3. Define season division function
# def get_season(month):
#     if month in [3, 4, 5]:
#         return 'Spring'
#     elif month in [6, 7, 8]:
#         return 'Summer'
#     elif month in [9, 10, 11]:
#         return 'Autumn'
#     else:  # 12, 1, 2
#         return 'Winter'

# df_2019['Season'] = df_2019['Time'].dt.month.apply(get_season)

# # Set plotting style
# sns.set_theme(style="whitegrid")
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 换成英文后不再需要强制中文字体
# plt.rcParams['axes.unicode_minus'] = False 

# # 4. Create visualization canvas
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# # Plot A: Seasonal Power Distribution (Box Plot)
# sns.boxplot(x='Season', y='Power', data=df_2019, palette='coolwarm', ax=axes[0])
# axes[0].set_title('Seasonal Distribution of Wind Power in 2019 (Box Plot)', fontsize=14)
# axes[0].set_xlabel('Season', fontsize=12)
# axes[0].set_ylabel('Power (MW)', fontsize=12)

# # Plot B: Seasonal Power Density Distribution (KDE Plot)
# sns.kdeplot(data=df_2019, x='Power', hue='Season', fill=True, common_norm=False, palette='viridis', ax=axes[1])
# axes[1].set_title('Probability Density Distribution of Wind Power in 2019 (KDE Plot)', fontsize=14)
# axes[1].set_xlabel('Power (MW)', fontsize=12)
# axes[1].set_ylabel('Density', fontsize=12)

# plt.tight_layout()
# plt.savefig('seasonal_distribution_2019.png', dpi=300)
# plt.show()

# # 5. Output simple statistical analysis
# print("Summary of Seasonal Power Statistics in 2019:")
# print(df_2019.groupby('Season')['Power'].describe())



# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# # --- 生成模拟风电数据 (模拟 Farm 2) ---
# np.random.seed(42)
# periods = 96 * 4  # 生成 4 天的 15 分钟间隔数据 (共 384 点)
# time = pd.date_range("2020-01-01", periods=periods, freq="15min")
# hours = time.hour

# # 基础功率模型：日周期 + 随机噪声 (峰值在 200 MW 左右)
# actual = 100 + 70 * np.sin(np.linspace(0, 4*np.pi, periods)) + np.random.normal(0, 10, periods)
# actual = np.clip(actual, 0, 200) # 裁剪到 0-200 MW 区间

# # 模拟特征：风速 (与功率大致相关)
# # 假设功率正比于风速的立方，但加入非线性噪声和边界限制
# base_ws = (actual / 200)**(1/3) * 15 # 逆向立方缩放，得到大约风速
# wind_speed = base_ws + np.random.normal(0, 1.5, periods)
# wind_speed = np.clip(wind_speed, 0, 20) # 裁剪风速区间

# # 预测模型 (模拟“Ours”模型的表现)
# # 加入确定性滞后和异方差噪声 (功率越大，噪声相对越大)
# lag = 2 # 滞后 2 个时间步 (30 分钟)
# pred = np.roll(actual, lag) # 从滞后开始
# noise_scaling = 0.05 * pred + 3 # 噪声随功率幅值增加
# noise = np.random.normal(0, noise_scaling, periods)
# pred = pred + noise
# pred[0:lag] = actual[0:lag] # 修复前几个滞后的点
# pred = np.clip(pred, 0, 200) # 裁剪预测值

# # 计算残差 (Actual - Predicted)
# residuals = actual - pred
# abs_residuals = np.abs(residuals)

# # 定义季节 (用于示例 3)
# def get_season(month):
#     if month in [3, 4, 5]: return 'Spring'
#     if month in [6, 7, 8]: return 'Summer'
#     if month in [9, 10, 11]: return 'Autumn'
#     return 'Winter'

# time_index = pd.to_datetime(time)
# seasons = time_index.month.map(get_season)

# # 合并为 DataFrame
# df = pd.DataFrame({
#     'Time': time,
#     'Actual': actual,
#     'Predicted': pred,
#     'Residuals': residuals,
#     'Abs_Residuals': abs_residuals,
#     'WindSpeed': wind_speed,
#     'Season': seasons,
#     'Hour': hours
# })

# # 设置绘图全局风格
# sns.set_theme(style="whitegrid")
# plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# # --- 示例 1 代码 ---
# fig = plt.figure(figsize=(16, 10), dpi=100)
# # 使用 GridSpec 定义复杂布局：3行2列，不同行列的比例
# gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1.2], hspace=0.15, wspace=0.2)

# # 1. 主预测曲线图 (叠加误差阴影)
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.plot(df['Time'], df['Actual'], label='Actual Power', color='#2c3e50', linewidth=1.5)
# ax1.plot(df['Time'], df['Predicted'], label='AFMS-Wind (Ours)', color='#e74c3c', linestyle='--', linewidth=1.5)
# # 添加误差阴影带 (Actual ± |Residual|)
# ax1.fill_between(df['Time'], df['Predicted'] - df['Abs_Residuals'], df['Predicted'] + df['Abs_Residuals'], 
#                  color='#e74c3c', alpha=0.1, label='Uncertainty Range')
# ax1.set_ylabel('Power (MW)', fontsize=12)
# ax1.set_title('Comprehensive Forecasting Diagnosis (Farm 2)', fontsize=14)
# ax1.legend(loc='upper right', frameon=True)
# ax1.grid(True, linestyle='--', alpha=0.5)
# ax1.set_xticklabels([]) # 隐藏主图 X 轴标签

# # 1a. Inset Plot: 局部放大图 (例如捕获 ramp 事件)
# # 定义放大区域
# ramp_start = periods // 2
# ramp_end = ramp_start + 48 # 放大 12 小时的区域
# ax1_inset = ax1.inset_axes([0.02, 0.05, 0.45, 0.45]) # [x, y, width, height] 相对于父图
# ax1_inset.plot(df['Time'][ramp_start:ramp_end], df['Actual'][ramp_start:ramp_end], color='#2c3e50', lw=1.2)
# ax1_inset.plot(df['Time'][ramp_start:ramp_end], df['Predicted'][ramp_start:ramp_end], color='#e74c3c', ls='--', lw=1.2)
# ax1_inset.fill_between(df['Time'][ramp_start:ramp_end], df['Predicted'][ramp_start:ramp_end] - df['Abs_Residuals'][ramp_start:ramp_end], 
#                        df['Predicted'][ramp_start:ramp_end] + df['Abs_Residuals'][ramp_start:ramp_end], color='#e74c3c', alpha=0.1)
# ax1_inset.set_title('Zoom-in: Ramp Tracking', fontsize=10)
# ax1_inset.grid(True, ls='--', alpha=0.3)
# ax1_inset.set_xticklabels([]) # 隐藏 inset 的 X 轴
# # 在主图上标记放大区域
# from matplotlib.patches import ConnectionPatch
# rect_xy = (df['Time'][ramp_start], df['Actual'][ramp_start:ramp_end].min() - 5)
# rect_w = df['Time'][ramp_end] - df['Time'][ramp_start]
# rect_h = df['Actual'][ramp_start:ramp_end].max() - df['Actual'][ramp_start:ramp_end].min() + 10
# ax1.add_patch(plt.Rectangle(rect_xy, rect_w, rect_h, edgecolor='gray', facecolor='none', lw=1, ls='-'))

# # 2. 残差时序图 (下部子图)
# ax2 = fig.add_subplot(gs[1, 0], sharex=ax1) # 共享 X 轴
# ax2.bar(df['Time'], df['Residuals'], color='#3498db', alpha=0.6, label='Residuals ($y - \hat{y}$)')
# ax2.axhline(0, color='black', lw=1, ls='-')
# ax2.set_ylabel('Residual (MW)', fontsize=12)
# ax2.set_xlabel('Time Sequence', fontsize=12)
# ax2.legend(loc='upper left')
# ax2.grid(True, linestyle='--', alpha=0.5)

# # 3. 散点相关性分析 (右侧子图: Parity Plot)
# ax3 = fig.add_subplot(gs[0, 1])
# # 计算统计指标用于展示
# r2_val = np.corrcoef(df['Actual'], df['Predicted'])[0, 1]**2
# mse_val = np.mean(df['Residuals']**2)
# sns.regplot(x=df['Actual'], y=df['Predicted'], ax=ax3, 
#             scatter_kws={'alpha':0.4, 's':15, 'color':'#e74c3c'}, line_kws={'color':'#2c3e50', 'lw':1.5})
# # 添加对角线 (参考线)
# ax3.plot([0, 200], [0, 200], color='gray', linestyle=':', lw=1)
# ax3.set_xlim(0, 200); ax3.set_ylim(0, 200)
# ax3.set_xlabel('Actual Value', fontsize=10)
# ax3.set_ylabel('Predicted Value', fontsize=10)
# ax3.set_title('Correlation & Statistics', fontsize=12)
# # 在图上添加文本标签
# text_str = f'$R^2$: {r2_val:.4f}\nMSE: {mse_val:.2f}'
# ax3.text(10, 170, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# # 4. 误差频率分布 (右下子图)
# ax4 = fig.add_subplot(gs[1, 1])
# sns.histplot(df['Residuals'], kde=True, ax=ax4, color='#3498db', bins=30)
# ax4.set_xlabel('Error Distribution', fontsize=10)
# ax4.set_title('Residual Statistics', fontsize=12)

# # plt.tight_layout() # gs 布局下 tight_layout 容易错乱，使用 hspace/wspace 控制
# plt.savefig('combined_diagnosis.png', bbox_inches='tight', dpi=300)
# plt.show()

# # --- 示例 2 代码 ---
# # fig, (ax_ts, ax_cond) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# # # 1. 上部：时间序列预测
# # ax_ts.plot(df['Time'], df['Actual'], label='Actual', color='#2c3e50', lw=1.5)
# # ax_ts.plot(df['Time'], df['Predicted'], label='Predicted', color='#e74c3c', ls='--', lw=1.5)
# # ax_ts.set_ylabel('Power (MW)', fontsize=12)
# # ax_ts.set_title('Conditional Error Analysis: Power vs. Wind Speed', fontsize=14)
# # ax_ts.legend(loc='upper right')

# # # 2. 下部：基于风速特征的条件误差
# # # 将风速划分为不同的 Bin
# # df['WS_Bin'] = pd.cut(df['WindSpeed'], bins=np.arange(0, 22, 2.5))
# # # 计算每个 Bin 的均方误差 (MSE)
# # cond_mse = df.groupby('WS_Bin', observed=False)['Residuals'].apply(lambda x: np.mean(x**2)).reset_index(name='MSE')

# # # 绘制柱状图展示条件误差
# # sns.barplot(data=cond_mse, x='WS_Bin', y='MSE', color='#3498db', alpha=0.7, ax=ax_cond)
# # # 叠加一个平滑趋势线
# # x_ticks = np.arange(len(cond_mse))
# # ax_cond.plot(x_ticks, cond_mse['MSE'], color='#2c3e50', marker='o', lw=1, ms=5, ls='-')

# # ax_cond.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
# # ax_cond.set_xlabel('Wind Speed Bins (m/s)', fontsize=12)
# # ax_cond.set_title('Prediction Error conditional on Wind Speed', fontsize=12)
# # # 格式化下部 X 轴标签
# # ax_cond.set_xticklabels([str(label) for label in cond_mse['WS_Bin']], rotation=0)

# # plt.tight_layout()
# # plt.savefig('conditional_error.png', dpi=300)
# # plt.show()


# # ==========================================
# # 5. 模拟 6 个模型在 4 个指标上的数据 (用于热力图)
# # ==========================================
# models = ['iTransformer', 'Transformer', 'PatchTST', 'TimeXer', 'DLinear', 'Ours']
# metrics = ['MSE', 'MAE', 'RMSE', 'R2']

# # 模拟实验数据 (参考您之前提供的 Farm 2 数据)
# data = np.array([
#     [182.06, 9.06, 13.49, 0.9408], # iTransformer
#     [237.00, 11.20, 15.39, 0.9229], # Transformer
#     [209.65, 10.47, 14.48, 0.9318], # PatchTST
#     [178.70, 9.02, 13.37, 0.9419], # TimeXer
#     [194.06, 8.59, 13.93, 0.9369], # DLinear
#     [164.50, 8.38, 12.83, 0.9465]  # Ours
# ])

# metric_df = pd.DataFrame(data, index=models, columns=metrics)

# # ==========================================
# # 6. 绘制性能排名热力图 (Heatmap)
# # ==========================================
# plt.figure(figsize=(10, 6))
# # 对指标进行归一化处理（0-1），以便颜色深浅能反映排名
# # 注意：MSE/MAE/RMSE是越小越好，R2是越大越好
# norm_df = metric_df.copy()
# for col in ['MSE', 'MAE', 'RMSE']:
#     norm_df[col] = (norm_df[col].max() - norm_df[col]) / (norm_df[col].max() - norm_df[col].min())
# norm_df['R2'] = (norm_df['R2'] - norm_df['R2'].min()) / (norm_df['R2'].max() - norm_df['R2'].min())

# sns.heatmap(norm_df, annot=metric_df.values, fmt=".2f", cmap="YlGnBu", 
#             linewidths=.5, cbar_kws={'label': 'Normalized Performance'})
# plt.title('Model Performance Comparison Heatmap (Farm 2)', fontsize=14)
# plt.ylabel('Models')
# plt.xlabel('Evaluation Metrics')
# plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
# plt.show()

# # ==========================================
# # 7. 绘制泰勒图 (Taylor Diagram - 自定义实现)
# # ==========================================
# def plot_taylor_diagram(actual_std, model_stats, save_name='taylor_diagram.png'):
#     """
#     actual_std: 真实数据的标准差
#     model_stats: 字典，键为模型名，值为 (std, correlation)
#     """
#     fig = plt.figure(figsize=(9, 9))
#     ax = fig.add_subplot(111, projection='polar')
    
#     # 设置极坐标范围 (0 到 90 度对应 相关系数 1 到 0)
#     # 相关系数 r 对应 角度 theta = arccos(r)
#     sample_corr = np.linspace(0, 1, 100)
#     theta = np.arccos(sample_corr)
    
#     # 1. 绘制相关系数弧线
#     ax.set_thetagrids(np.degrees(np.arccos([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])), 
#                       ['0', '0.2', '0.4', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99', '1'])
#     ax.set_theta_offset(0)
#     ax.set_thetamin(0)
#     ax.set_thetamax(90)
    
#     # 2. 设置标准差刻度 (径向)
#     max_std = actual_std * 1.5
#     ax.set_ylim(0, max_std)
#     ax.set_xlabel('Standard Deviation (Normalized)', labelpad=30, fontsize=12)
    
#     # 3. 绘制参考点 (真实数据)
#     ax.plot(0, actual_std, 'ko', markersize=10, label='Reference (Actual)')
    
#     # 4. 绘制等均方根误差 (RMSE) 弧线
#     rs, ts = np.meshgrid(np.linspace(0, max_std, 100), np.linspace(0, np.pi/2, 100))
#     rms = np.sqrt(actual_std**2 + rs**2 - 2 * actual_std * rs * np.cos(ts))
#     contours = ax.contour(ts, rs, rms, levels=5, colors='gray', linestyles='--', alpha=0.5)
#     ax.clabel(contours, inline=1, fontsize=8, fmt='RMS: %.1f')

#     # 5. 绘制模型点
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e74c3c']
#     for (name, stats), color in zip(model_stats.items(), colors):
#         std_val, corr_val = stats
#         theta_val = np.arccos(corr_val)
#         ax.plot(theta_val, std_val, 'o', color=color, markersize=8, label=name)

#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
#     plt.title('Taylor Diagram for Model Comparison', pad=20, fontsize=14)
#     plt.savefig(save_name, dpi=300, bbox_inches='tight')
#     plt.show()

# # 为 6 个模型模拟 (标准差, 相关系数)
# # 真实数据的标准差
# ref_std = df['Actual'].std()
# # 模拟各模型的表现
# taylor_stats = {
#     'iTransformer': (ref_std * 1.12, 0.9408),
#     'Transformer':  (ref_std * 1.25, 0.9229),
#     'PatchTST':     (ref_std * 1.18, 0.9318),
#     'TimeXer':      (ref_std * 1.08, 0.9419),
#     'DLinear':      (ref_std * 0.90, 0.9369),
#     'Ours':         (df['Predicted'].std(), np.corrcoef(df['Actual'], df['Predicted'])[0,1])
# }

# plot_taylor_diagram(ref_std, taylor_stats)








# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 1. 整理实验数据
# data = {
#     'Season': ['Spring']*6 + ['Summer']*6 + ['Autumn']*6 + ['Winter']*6,
#     'Model': ['iTransformer', 'Transformer', 'PatchTST', 'TimeXer', 'DLinear', 'Ours'] * 4,
#     'MSE': [
#         213.3615, 410.5920, 213.3737, 298.7681, 274.6122, 193.7989, # Spring
#         211.4858, 297.3580, 199.0778, 245.9120, 218.5206, 196.8597, # Summer
#         199.0113, 278.5880, 203.7615, 254.7511, 205.1619, 190.4830, # Autumn
#         238.4417, 407.3321, 238.0503, 316.3474, 309.8527, 236.3088  # Winter
#     ],
#     'MAE': [
#         9.6726, 14.9381, 10.2804, 12.2108, 10.9859, 9.0955,
#         10.0644, 13.1892, 10.0815, 11.1325, 9.3256, 9.4109,
#         9.6485, 11.9413, 10.1147, 11.3888, 8.8994, 9.1540,
#         10.0253, 14.6022, 10.2471, 12.3105, 11.0084, 9.9833
#     ],
#     'RMSE': [
#         14.6068, 20.2630, 14.6073, 17.2849, 16.5714, 13.9211,
#         14.5426, 17.2441, 14.1095, 15.6816, 14.7824, 14.0307,
#         14.1071, 16.6910, 14.2745, 15.9609, 14.3235, 13.8016,
#         15.4416, 20.1825, 15.4289, 17.7862, 17.6026, 15.3723
#     ],
#     'R2': [
#         0.9277, 0.8609, 0.9277, 0.8988, 0.9070, 0.9343,
#         0.8982, 0.8568, 0.9041, 0.8816, 0.8948, 0.9052,
#         0.9316, 0.9042, 0.9299, 0.9124, 0.9295, 0.9345,
#         0.9358, 0.8904, 0.9359, 0.9149, 0.9166, 0.9364
#     ]
# }

# df = pd.DataFrame(data)
# df['Season-Model'] = df['Season'] + "-" + df['Model']
# df_heatmap = df.set_index('Season-Model')[['MSE', 'MAE', 'RMSE', 'R2']]

# # 2. 归一化处理 (以便颜色深浅代表性能优劣)
# norm_df = df_heatmap.copy()
# for col in ['MSE', 'MAE', 'RMSE']:
#     # 越小越好：(Max - Val) / (Max - Min)
#     norm_df[col] = (norm_df[col].max() - norm_df[col]) / (norm_df[col].max() - norm_df[col].min())
# # R2 越大越好：(Val - Min) / (Max - Min)
# norm_df['R2'] = (norm_df['R2'] - norm_df['R2'].min()) / (norm_df['R2'].max() - norm_df['R2'].min())

# # 3. 绘制热力图
# plt.figure(figsize=(10, 12))
# sns.heatmap(norm_df, annot=df_heatmap.values, fmt=".4f", cmap="YlGnBu", 
#             linewidths=.5, cbar_kws={'label': 'Normalized Performance Rank'})

# # 添加季节分割线
# for i in range(1, 4):
#     plt.axhline(i * 6, color='white', lw=3)

# plt.title('Seasonal Performance Matrix for Farm 2', fontsize=16, pad=20)
# plt.ylabel('Season and Model Pairs', fontsize=12)
# plt.xlabel('Evaluation Metrics', fontsize=12)
# plt.savefig('seasonal_heatmap_farm2.png', dpi=300, bbox_inches='tight')
# plt.show()






# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # 1. 录入 Farm 2 实验数据
# data_dict = {
#     'Spring': [[213.3615, 9.6726, 14.6068, 0.9277, 23.6221], [410.5920, 14.9381, 20.2630, 0.8609, 39.3722], [213.3737, 10.2804, 14.6073, 0.9277, 29.1359], [298.7681, 12.2108, 17.2849, 0.8988, 34.3769], [274.6122, 10.9859, 16.5714, 0.9070, 30.4632], [193.7989, 9.0955, 13.9211, 0.9343, 22.0204]],
#     'Summer': [[211.4858, 10.0644, 14.5426, 0.8982, 23.8025], [297.3580, 13.1892, 17.2441, 0.8568, 26.9638], [199.0778, 10.0815, 14.1095, 0.9041, 28.5729], [245.9120, 11.1325, 15.6816, 0.8816, 31.8910], [218.5206, 9.3256, 14.7824, 0.8948, 22.5676], [196.8597, 9.4109, 14.0307, 0.9052, 22.7229]],
#     'Autumn': [[199.0113, 9.6485, 14.1071, 0.9316, 25.0728], [278.5880, 11.9413, 16.6910, 0.9042, 27.5697], [203.7615, 10.1147, 14.2745, 0.9299, 29.3162], [254.7511, 11.3888, 15.9609, 0.9124, 34.5458], [205.1619, 8.8994, 14.3235, 0.9295, 23.7867], [190.4830, 9.1540, 13.8016, 0.9345, 24.3757]],
#     'Winter': [[238.4417, 10.0253, 15.4416, 0.9358, 38.0208], [407.3321, 14.6022, 20.1825, 0.8904, 54.5195], [238.0503, 10.2471, 15.4289, 0.9359, 39.2740], [316.3474, 12.3105, 17.7862, 0.9149, 52.0666], [309.8527, 11.0084, 17.6026, 0.9166, 43.4971], [236.3088, 9.9833, 15.3723, 0.9364, 36.9634]]
# }

# models = ['iTransformer', 'Transformer', 'PatchTST', 'TimeXer', 'DLinear', 'Ours']
# metrics = ['MSE', 'MAE', 'RMSE', 'R²', 'MAPE']

# # 高对比度颜色配色
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e31a1c'] # 红色突出 Ours

# # 绘图初始化
# N = len(metrics)
# angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# angles += angles[:1]

# plt.rcParams.update({'font.size': 11, 'font.weight': 'normal'}) # 统一非加粗字体
# fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw=dict(polar=True))

# for idx, (season, s_data) in enumerate(data_dict.items()):
#     ax = axes[idx // 2, idx % 2]
#     season_raw = np.array(s_data)
    
#     # 计算当前季节局部刻度 (增加留白)
#     s_min = season_raw.min(axis=0) * 0.95
#     s_max = season_raw.max(axis=0) * 1.05
    
#     for m_idx, model_name in enumerate(models):
#         raw_vals = season_raw[m_idx]
#         # 归一化：(val - min) / (max - min)。R2 越大越靠外
#         norm_vals = (raw_vals - s_min) / (s_max - s_min)
#         plot_vals = norm_vals.tolist() + [norm_vals[0]]
        
#         ax.plot(angles, plot_vals, color=colors[m_idx], lw=1.8, alpha=0.9, label=model_name if idx==0 else "")
#         ax.fill(angles, plot_vals, color=colors[m_idx], alpha=0.06)

#     # 绘制轴刻度数据
#     tick_levels = [0.2, 0.5, 0.8, 1.0]
#     for i, angle in enumerate(angles[:-1]):
#         for lvl in tick_levels:
#             val = s_min[i] + lvl * (s_max[i] - s_min[i])
#             label = f"{val:.2f}" if i == 3 or val < 10 else (f"{int(val)}" if val > 100 else f"{val:.1f}")
#             # ax.text(angle, lvl, label, color='gray', fontsize=8.5, ha='center', va='center',
#             #         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

#     # 坐标轴美化
#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)
#     ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
#     ax.set_yticklabels([]) # 隐藏极坐标背景刻度线
#     ax.set_ylim(0, 1.1)
    
#     # 将季节名称放在左侧 [重点修改]
#     ax.annotate(season, xy=(-0.25, 0.5), xycoords='axes fraction', 
#                 ha='center', va='center', fontsize=16, rotation=90, color='#333333')

# # 底部图例
# fig.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.04), frameon=False, fontsize=12)
# plt.subplots_adjust(left=0.18, right=0.88, wspace=0.0, hspace=0.25, bottom=0.15, top=0.95)

# # 底部图例（保持不变）
# fig.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.04), frameon=False, fontsize=12)

# # 保存时注意不要再调用 tight_layout，否则会覆盖上面的 adjust 设置
# plt.savefig('seasonal_radar_centered.png', dpi=300, bbox_inches='tight')
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # 1. 实验数据
# k_values = [6, 7, 8, 9, 10]

# # Farm 1
# f1_mse = [44.5514, 42.9268, 39.8036, 41.5534, 42.3916] 
# f1_r2  = [0.9215, 0.9244, 0.9298, 0.9268, 0.920]

# # Farm 2
# f2_mse = [174.0563, 155.0757, 164.4971, 171.3237, 172.3237] 
# f2_r2  = [0.9434, 0.9496, 0.9465, 0.9443, 0.9343]

# # Farm 3
# f3_mse = [32.5804, 30.1631, 30.5376, 33.5651, 34.3916] 
# f3_r2  = [0.9223, 0.9281, 0.9272, 0.9200, 0.9187]

# # Farm 4
# f4_mse = [12.6087, 11.8380, 12.1816, 13.3774, 13.8300]
# f4_r2  = [0.9734, 0.9750, 0.9742, 0.9717, 0.9709]

# # 2. 图形初始化 (改为 2行2列)
# plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
# fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# # 颜色配置
# color_mse = '#3498db'  # 蓝色代表误差
# color_r2 = '#e74c3c'   # 红色代表精度

# def plot_sensitivity(ax, k, mse, r2, title, mse_range, r2_range):
#     # 绘制左侧 Y 轴 (MSE)
#     ax.bar(k, mse, color=color_mse, alpha=0.3, width=0.5, label='MSE (Left)')
#     ax.plot(k, mse, color=color_mse, marker='o', linewidth=2, markersize=6)
#     ax.set_xlabel('Number of Modalities ($K$)', fontsize=10)
#     ax.set_ylabel('MSE (MW)', color=color_mse, fontsize=10)
#     ax.tick_params(axis='y', labelcolor=color_mse)
#     ax.set_ylim(mse_range)
#     ax.set_xticks(k)
    
#     # 创建右侧 Y 轴 (R2)
#     ax_twin = ax.twinx()
#     ax_twin.plot(k, r2, color=color_r2, marker='s', linestyle='--', linewidth=2, markersize=6, label='$R^2$ (Right)')
#     ax_twin.set_ylabel('$R^2$ Score', color=color_r2, fontsize=10)
#     ax_twin.tick_params(axis='y', labelcolor=color_r2)
#     ax_twin.set_ylim(r2_range)
    
#     ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
#     ax.grid(axis='y', linestyle='--', alpha=0.3)
    
#     return ax, ax_twin

# # 3. 绘制四个站点
# # 根据每个站点的数据分布手动调整 y 轴范围以突出变化趋势
# plot_sensitivity(axs[0, 0], k_values, f1_mse, f1_r2, 'Farm 1', [38, 46], [0.915, 0.935])
# plot_sensitivity(axs[0, 1], k_values, f2_mse, f2_r2, 'Farm 2', [150, 180], [0.930, 0.955])
# plot_sensitivity(axs[1, 0], k_values, f3_mse, f3_r2, 'Farm 3', [29, 36], [0.915, 0.935])
# plot_sensitivity(axs[1, 1], k_values, f4_mse, f4_r2, 'Farm 4', [11, 14.5], [0.965, 0.980])

# # 4. 添加统一图例
# # 获取任意一个子图的句柄
# h1, l1 = axs[0, 0].get_legend_handles_labels()
# h2, l2 = axs[0, 0].get_legend_handles_labels() # 这里需要获取右轴的，逻辑微调如下
# ax_l = axs[0, 0]
# ax_r = ax_l.get_shared_x_axes().get_siblings(ax_l)[1] # 获取现有的右轴，而不是新建

# # 或者更简单的做法：直接手动定义图例元素（最稳妥，不会产生多余轴）
# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], color=color_mse, marker='o', lw=2, label='MSE (Left Axis)'),
#     Line2D([0], [0], color=color_r2, marker='s', linestyle='--', lw=2, label='$R^2$ Score (Right Axis)')
# ]

# # 绘制图例
# fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
#            bbox_to_anchor=(0.5, 0.02), frameon=True, fontsize=11)

# # 调整布局，确保底部留出空间
# plt.tight_layout(rect=[0, 0.05, 1, 1])
# plt.savefig('sensitivity_analysis_K_4_farms.png', dpi=300, bbox_inches='tight')














import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# ==========================================
# 1. 核心参数配置区 (Configurable Parameters)
# ==========================================
file_paths = [
    r'checkpoints\iTransformer\20260307_165018_iTransformer\20260307_181748_test\iTransformer_pred.csv', 
    r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260404_001318_iTransformer_xLSTM_VMD_Preprocessed\20260407_102112_test\iTransformer_xLSTM_VMD_Preprocessed_pred.csv',
    r'checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260311_145935_iTransformer_xLSTM_VMD_Preprocessed\20260311_151243_test\iTransformer_xLSTM_VMD_Preprocessed_pred.csv', 
    r'checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260309_184904_iTransformer_xLSTM_VMD_Preprocessed\20260309_190513_test\iTransformer_xLSTM_VMD_Preprocessed_pred.csv', 
    r'checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260307_165913_iTransformer_xLSTM_VMD_Preprocessed\20260307_181811_test\iTransformer_xLSTM_VMD_Preprocessed_pred.csv'
]

titles = ['iTransformer(it)', 'it + VMD', 'it + AFDM', 'it + SRGRU', 'it + AFDM + SRGRU']
START_TIME = '2020-10-30 00:00:00'
END_TIME = '2020-10-31 23:45:00'

# ==========================================
# 2. 数据处理与绘图逻辑
# ==========================================
def plot_multi_farm_predictions(files, titles, start_time, end_time):
    # 🌟 修改 1：大幅增加画布的绝对高度 (从 20 增加到 24)
    fig = plt.figure(figsize=(16, 24))
    
    # 🌟 修改 2：调整 height_ratios，把序列图的比例从 3 加大到 4
    # [4, 1] 代表主图高度是残差图的 4 倍
    # 1.5 是排与排之间的空白距离
    gs = gridspec.GridSpec(8, 4, height_ratios=[5, 1, 1.5, 5, 1, 1.5, 5, 1], hspace=0.08, wspace=0.2)
    
    for i, file_path in enumerate(files):
        # 1. 加载并对齐数据
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"⚠️ 找不到文件: {file_path}，请检查路径。")
            continue
            
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
        if start_time and end_time:
            df = df.loc[start_time:end_time]
        df = df.dropna(subset=['Power', 'pred'])
        df['Residual'] = df['Power'] - df['pred']
        
        # 2. 动态计算网格位置
        if i == 4:
            # 最后一个子图
            block_row = 2
            col_start = 1
            col_end = 3
        else:
            # 前 4 个子图
            block_row = i // 2
            col_start = (i % 2) * 2
            col_end = col_start + 2
            
        # 计算真实行索引时，要跳过空白行
        row_main = block_row * 3
        row_res = row_main + 1
        
        # 3. 绘制上方的主预测曲线图
        ax_main = fig.add_subplot(gs[row_main, col_start:col_end])
        ax_main.plot(df.index, df['Power'], label='Actual Power', color='#2c3e50', linewidth=1.5)
        ax_main.plot(df.index, df['pred'], label='Predicted Power', color='#e74c3c', linestyle='--', linewidth=1.5)
        
        ax_main.set_title(titles[i], fontsize=14)
        ax_main.set_ylabel('Power(MW)', fontsize=10)
        ax_main.legend(loc='upper right', framealpha=0.8)
        ax_main.grid(True, linestyle='--', alpha=0.5)
        
        # 隐藏主图的 X 轴时间标签
        plt.setp(ax_main.get_xticklabels(), visible=False)
        
        # 4. 绘制下方的残差图
        ax_res = fig.add_subplot(gs[row_res, col_start:col_end], sharex=ax_main)
        ax_res.bar(df.index, df['Residual'], color='#3498db', alpha=0.7, width=0.01)
        ax_res.axhline(0, color='black', linewidth=1, linestyle='-')
        
        ax_res.set_ylabel('Residual', fontsize=10)
        ax_res.grid(True, linestyle='--', alpha=0.5)
        
        # 格式化 X 轴只显示小时和分钟
        ax_res.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

plot_multi_farm_predictions(file_paths, titles, START_TIME, END_TIME)