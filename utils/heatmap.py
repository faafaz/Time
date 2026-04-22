import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. 数据准备
# ==========================================
# np.random.seed(42)
# time_steps = 200
# t = np.linspace(0, 8*np.pi, time_steps)
# original_data = np.sin(t) + np.sin(3*t) + np.random.normal(0, 0.3, time_steps)
# imfs = []
# for i in range(1, 9):
#     freq = 0.5 * i
#     noise = np.random.normal(0, 0.1, time_steps) * (0.1 * i)
#     imf = np.sin(freq * t) / i + noise
#     imfs.append(imf)

# data_dict = {'Original': original_data}
# for i, imf in enumerate(imfs):
#     data_dict[f'IMF{i+1}'] = imf
# df = pd.DataFrame(data_dict)
df = pd.read_csv(r'vmd_data_results\sample_3_decomposition_data.csv') 

# bandwidths = [8.0509, 6.2528, 5.4123, 5.5400, 5.9996, 7.8810, 8.5163, 4.9128]
bandwidths = [5.4106, 5.9277, 6.4238, 6.2905, 6.3364, 5.7651, 7.3417, 6.8444]

# ==========================================
# 2. 绘图代码
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(15, 12))

# 保持 GridSpec 手动布局，确保所有黑框（Axes）本身是对齐的
# left=0.12: 稍微加大一点左边距，给强制对齐的标签留空间
gs = gridspec.GridSpec(9, 2, width_ratios=[25, 1.2], wspace=0.03, hspace=0.35,
                       left=0.12, right=0.92, top=0.92, bottom=0.05)

norm = plt.Normalize(min(bandwidths), max(bandwidths))
cmap = plt.cm.RdYlBu_r 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# --- A. 绘制原始数据 ---
ax_orig = fig.add_subplot(gs[0, 0])
ax_orig.plot(df.iloc[:, 0], color='black', linewidth=1.5)
# ax_orig.set_title('原始序列分解与自适应频带宽度可视化', fontsize=16, pad=20)
ax_orig.set_xticks([])
ax_orig.grid(True, linestyle='--', alpha=0.3)
ax_orig.tick_params(axis='both', which='both', length=0) 

# 【核心修改1】移除 set_ylabel，使用 ax.text 强制对齐
# x=-0.08: 位于坐标轴左侧 8% 的位置 (相对坐标)
# ha='right': 文字右对齐，视觉上更整齐
ax_orig.text(-0.06, 0.5, 'Original', transform=ax_orig.transAxes, 
             fontsize=11, fontweight='bold', ha='right', va='center')

ax_blank = fig.add_subplot(gs[0, 1])
ax_blank.axis('off')

axes_bw = []

# --- B. 循环绘制 ---
for i in range(8):
    row_idx = i + 1 
    imf_col_name = f'IMF{i+1}'
    bw_value = bandwidths[i]
    color = cmap(norm(bw_value))
    
    # 1. 左侧: 时序波形图
    ax_imf = fig.add_subplot(gs[row_idx, 0])
    if imf_col_name in df.columns:
        series_data = df[imf_col_name]
    else:
        series_data = df.iloc[:, i+1]
        
    ax_imf.plot(series_data, color=color, linewidth=1.2)
    ax_imf.grid(True, linestyle='--', alpha=0.3)
    ax_imf.tick_params(axis='both', which='both', length=0)
    
    # 【核心修改2】对所有 IMF 使用相同的 text 坐标参数
    # 这样无论左边的刻度数字是长是短，标签 "IMFx" 的位置绝对固定
    ax_imf.text(-0.06, 0.5, f'IMF{i+1}', transform=ax_imf.transAxes, 
                fontsize=10, ha='right', va='center')
    
    if i < 7:
        ax_imf.set_xticks([])
    else:
        ax_imf.set_xlabel('Time Steps', fontsize=12)

    # 2. 右侧: 带宽数值色块
    ax_bw = fig.add_subplot(gs[row_idx, 1])
    rect = plt.Rectangle((0, 0), 1, 1, color=color)
    ax_bw.add_patch(rect)
    
    text_color = 'white' if abs(norm(bw_value) - 0.5) > 0.2 else 'black'
    ax_bw.text(0.5, 0.5, f'{bw_value:.4f}', 
               ha='center', va='center', 
               fontsize=10, fontweight='bold', color=text_color)
    ax_bw.axis('off')
    axes_bw.append(ax_bw)

# --- C. Colorbar ---
pos_top = axes_bw[0].get_position()
pos_bottom = axes_bw[-1].get_position()

left = pos_top.x1 + 0.01 
bottom = pos_bottom.y0
width = 0.015
height = pos_top.y1 - pos_bottom.y0

cbar_ax = fig.add_axes([left, bottom, width, height])
cb = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cb.set_label('Bandwidth', fontsize=12, labelpad=10)
cb.ax.tick_params(axis='y', length=0) 

plt.savefig('IMFs_Perfectly_Aligned.png', dpi=300)
plt.show()