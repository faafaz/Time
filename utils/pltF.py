import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置路径 =================
# 请修改为你实际保存 npy 文件的路径
DATA_DIR = r'checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260307_210815_iTransformer_xLSTM_VMD_Preprocessed\20260307_221302_test' 
SPECTRUM_FILE = os.path.join(DATA_DIR, 'test_set_all_modes_spectra.npy')
FREQ_AXIS_FILE = os.path.join(DATA_DIR, 'test_set_freq_axis.npy')

def visualize_spectrum():
    # 1. 加载数据
    if not os.path.exists(SPECTRUM_FILE) or not os.path.exists(FREQ_AXIS_FILE):
        print(f"错误: 找不到文件。请检查路径:\n{SPECTRUM_FILE}\n{FREQ_AXIS_FILE}")
        return

    print(">>> 正在加载 .npy 文件...")
    spectra = np.load(SPECTRUM_FILE) 
    freqs = np.load(FREQ_AXIS_FILE)
    N_samples, N_freq, K = spectra.shape
    print(f"数据加载成功! 样本数: {N_samples}, 频率点: {N_freq}, 模态数(K): {K}")

    # ================= 图表 1: 平均频谱 (保持不变) =================
    plt.figure(figsize=(12, 6))
    mean_spectrum = np.mean(spectra, axis=0)
    for k in range(K):
        plt.plot(freqs, mean_spectrum[:, k], label=f'IMF {k+1}')
    # plt.title('Average Frequency Spectrum of All Modes (Test Set)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # ================= 图表 2: 单个样本分解 (保持不变) =================
    sample_idx = 1
    if sample_idx >= N_samples: sample_idx = 0
    fig, axes = plt.subplots(K, 1, figsize=(10, 1.5*K), sharex=True)
    sample_data = spectra[sample_idx]
    if K == 1: axes = [axes]
    for k in range(K):
        ax = axes[k]
        ax.plot(freqs, sample_data[:, k], color='tab:blue')
        ax.set_ylabel(f'IMF {k+1}')
        ax.grid(True, alpha=0.3)
        ax.fill_between(freqs, sample_data[:, k], alpha=0.3, color='tab:blue')
    axes[-1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

    # ================= 图表 3 (修改重点): 共享色条的热力图 =================
    print(">>> 正在生成所有模态的时频热力图...")
    
    # 截取数据
    time_steps = 3000 
    if time_steps > N_samples: time_steps = N_samples
    
    # [关键步骤 1] 计算全局最大值和最小值，确保所有子图颜色代表的数值一致
    # 这样共享的 Colorbar 才有意义
    heatmap_data_all = spectra[:time_steps, :, :].transpose(2, 1, 0) # [K, Freq, Time]
    global_min = heatmap_data_all.min()
    global_max = heatmap_data_all.max()

    # 创建画布，适当调整宽度以容纳右侧标签
    fig, axes = plt.subplots(K, 1, figsize=(15, 1.5*K), sharex=True, sharey=True)
    if K == 1: axes = [axes]

    # [关键步骤 2] 调整子图间距，为共享 Colorbar 留出空间
    # wspace/hspace=0 让图紧挨着，看起来像一张大图
    plt.subplots_adjust(hspace=0.05, right=0.85) 

    im = None # 用于保存最后一张图对象给 colorbar 使用

    for k in range(K):
        ax = axes[k]
        # 数据转置: [频率, 时间]
        heatmap_data = spectra[:time_steps, :, k].T 
        
        # 绘制热力图，使用全局 vmin/vmax
        im = ax.imshow(heatmap_data, aspect='auto', origin='lower', cmap='turbo', 
                       extent=[0, time_steps, freqs[0], freqs[-1]],
                       vmin=global_min, vmax=global_max)
        
        # [关键步骤 3] 在右侧显示 IMF 标签
        # 移除左侧默认 ylabel，改在右侧显示
        ax.set_ylabel(f"IMF {k+1}", rotation=0, labelpad=20, fontsize=12)
        ax.yaxis.set_label_position("right")
        
        # 只保留最下面子图的 X 轴刻度，其他的隐藏
        if k < K - 1:
            ax.tick_params(labelbottom=False)

    # [关键步骤 4] 添加全局元素
    # 设置最底部的 X 轴标签
    axes[-1].set_xlabel('Time Step', fontsize=12)
    
    # 设置全局 Y 轴标签 (创建一个不可见的坐标轴来居中显示标签)
    fig.text(0.08, 0.5, 'Frequency (Hz)', va='center', rotation='vertical', fontsize=14)

    # [关键步骤 5] 添加共享 Colorbar
    # cbar_ax 定义 colorbar 的位置 [left, bottom, width, height]
    # left=0.88 放在图表右侧
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) 
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Amplitude', fontsize=12)

    # 移除 plt.tight_layout()，因为我们手动调整了布局 (subplots_adjust)
    # plt.tight_layout() # 不要用这个，会打乱 add_axes 的布局
    
    plt.show()

if __name__ == '__main__':
    visualize_spectrum()