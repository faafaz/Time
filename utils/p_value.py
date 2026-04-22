# import scipy.stats as stats

# # 步骤 1：填入你 5 次独立实验记录下的误差指标（例如 MSE）
# # 注意：数据的顺序要一一对应，比如第一个位置都是 seed=42 跑出来的结果
# baseline_mse = [48.1147, 49.2555, 48.7669, 49.5554, 48.3611]  # 替换为最强基线的 5 次 MSE
# ours_mse = [39.8036, 40.3592, 40.3306, 40.9258, 43.0034]      # 替换为你 AMR-Wind 的 5 次 MSE

# # 步骤 2：进行 Wilcoxon 配对符号秩检验
# # alternative='greater' 意思是：我们检验假设 "Baseline 的误差显著大于 Ours 的误差"
# stat, p_value = stats.wilcoxon(baseline_mse, ours_mse, alternative='greater')

# # 步骤 3：输出结果
# print(f"Wilcoxon 检验 p-value: {p_value:.5f}")

# if p_value < 0.05:
#     print("结论：p < 0.05，提升在统计学上是【显著的】，可以直接写进论文！")
# else:
#     print("结论：p >= 0.05，提升不够显著（可能需要再多跑几个种子，或者用配对 T 检验试试）。")


import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 您的文件路径字典 (保持不变)
# ==========================================
target_metric = 'MSE'
file_paths_dict = {
    'iTransformer': [
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer\20260307_165018_iTransformer\20260307_181748_test\iTransformer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer\20260405_152558_iTransformer\20260405_154202_test\iTransformer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer\20260405_161100_iTransformer\20260405_165741_test\iTransformer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer\20260405_174522_iTransformer\20260405_185120_test\iTransformer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer\20260405_192313_iTransformer\20260405_204205_test\iTransformer_global_metrics.json'
    ],
    'Transformer': [
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\Transformer\20260307_163728_Transformer\20260307_181736_test\Transformer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\Transformer\20260405_151200_Transformer\20260405_154148_test\Transformer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\Transformer\20260405_155750_Transformer\20260405_165728_test\Transformer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\Transformer\20260405_173207_Transformer\20260405_185106_test\Transformer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\Transformer\20260405_191000_Transformer\20260405_204151_test\Transformer_global_metrics.json'
    ],
    'PatchTST': [
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\PatchTST\20260307_165557_PatchTST\20260307_181759_test\PatchTST_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\PatchTST\20260405_153130_PatchTST\20260405_154215_test\PatchTST_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\PatchTST\20260405_161824_PatchTST\20260405_165754_test\PatchTST_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\PatchTST\20260405_175001_PatchTST\20260405_185133_test\PatchTST_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\PatchTST\20260405_193017_PatchTST\20260405_204218_test\PatchTST_global_metrics.json'
    ],
    'TimeXer': [
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\TimeXer\20260307_162733_TimeXer\20260307_181724_test\TimeXer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\TimeXer\20260405_150120_TimeXer\20260405_154134_test\TimeXer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\TimeXer\20260405_154745_TimeXer\20260405_165714_test\TimeXer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\TimeXer\20260405_172153_TimeXer\20260405_185053_test\TimeXer_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\TimeXer\20260405_185948_TimeXer\20260405_204137_test\TimeXer_global_metrics.json'
    ],
    'DLinear': [
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\DLinear\20260405_222647_DLinear\20260405_223906_test\DLinear_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\DLinear\20260405_223937_DLinear\20260405_224343_test\DLinear_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\DLinear\20260405_224411_DLinear\20260405_224915_test\DLinear_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\DLinear\20260405_224938_DLinear\20260405_225312_test\DLinear_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\DLinear\20260405_225336_DLinear\20260405_225803_test\DLinear_global_metrics.json'
    ],
    'AMR-Wind (Ours)': [
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260307_165913_iTransformer_xLSTM_VMD_Preprocessed\20260307_181811_test\iTransformer_xLSTM_VMD_Preprocessed_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260404_113447_iTransformer_xLSTM_VMD_Preprocessed\20260404_120931_test\iTransformer_xLSTM_VMD_Preprocessed_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260405_125236_iTransformer_xLSTM_VMD_Preprocessed\20260405_131404_test\iTransformer_xLSTM_VMD_Preprocessed_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260405_131502_iTransformer_xLSTM_VMD_Preprocessed\20260405_140727_test\iTransformer_xLSTM_VMD_Preprocessed_global_metrics.json',
        r'C:\Users\Administrator\Desktop\TimeSeries\Time2\checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260405_142934_iTransformer_xLSTM_VMD_Preprocessed\20260405_145550_test\iTransformer_xLSTM_VMD_Preprocessed_global_metrics.json'
    ]
}

# ==========================================
# 2. 纯粹的 95% 置信区间计算
# ==========================================
display_names = list(file_paths_dict.keys())
means = []
ci_margins = [] # 存储置信区间的单侧半径

print(f"{'='*60}")
print(f"📉 {target_metric} 95% 置信区间计算结果 (可直接填入论文表格)")
print(f"{'='*60}")

for model_name, paths in file_paths_dict.items():
    model_metrics = []
    
    for file_path in paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_metrics.append(data[target_metric])
        except Exception:
            pass # 略过读取报错信息，专注于最终结果
            
    if len(model_metrics) > 1:
        metrics_arr = np.array(model_metrics)
        n = len(metrics_arr)
        
        # 1. 计算均值
        mean_val = np.mean(metrics_arr)
        
        # 2. 计算标准差
        std_val = np.std(metrics_arr, ddof=1)
        
        # 3. 计算 95% 置信区间半径
        confidence = 0.95
        se = std_val / np.sqrt(n) # 标准误
        margin = se * stats.t.ppf((1 + confidence) / 2., n-1)
        
        means.append(mean_val)
        ci_margins.append(margin)
        
        # 计算上下界
        lower_bound = mean_val - margin
        upper_bound = mean_val + margin
        
        print(f"🟢 {model_name:<18} | 均值: {mean_val:.4f} | 95% CI: [{lower_bound:.4f}, {upper_bound:.4f}]  (即 {mean_val:.4f} ± {margin:.4f})")
    else:
        print(f"🔴 {model_name:<18} | 数据不足以计算区间")
        means.append(0)
        ci_margins.append(0)

print(f"{'='*60}\n")

# ==========================================
# 3. 绘制“置信区间图” (纯线段森林图，无柱状图)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# 采用横向布局，这样能极其清晰地对比各个模型的区间范围
fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(display_names))

# 将 AMR-Wind 设为红色，其他为深灰色
colors = ['#546E7A'] * (len(display_names) - 1) + ['#D32F2F']

# 使用 errorbar 直接画点和置信区间线 (不画柱子)
for i in range(len(display_names)):
    ax.errorbar(means[i], y_pos[i], xerr=ci_margins[i], fmt='o', 
                color=colors[i], ecolor=colors[i], elinewidth=2, capsize=6, markersize=8)
    
    # 在点的右侧写上具体的区间数值
    ax.text(means[i] + ci_margins[i] + (0.01 * max(means)), y_pos[i], 
            f'{means[i]:.2f} ± {ci_margins[i]:.2f}', 
            va='center', color=colors[i], fontsize=10, fontweight='bold')

# 设置轴标签
trend_text = "(Lower is better)" if target_metric in ['MSE', 'MAE', 'RMSE', 'MAPE'] else "(Higher is better)"
# ax.set_xlabel(f'{target_metric} 95% Confidence Interval {trend_text}', fontsize=12, fontweight='bold')
ax.set_yticks(y_pos)
ax.set_yticklabels(display_names, fontsize=11)

# 反转 Y 轴，让 Ours 显示在最上面或最下面，这里默认把排在最后的显示在最下面
ax.invert_yaxis()

# 极简背景设置
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.5)

# plt.title('95% Confidence Intervals across 5 Runs', fontsize=14, pad=15)
plt.tight_layout()

# 保存纯净版区间图
save_path = f'interval_plot_{target_metric}.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"✅ 置信区间图已成功保存为图片: {os.path.abspath(save_path)}")

# plt.show()