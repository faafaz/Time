import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置路径 =================
# 请修改为你实际保存 npy 文件的文件夹路径
DATA_DIR = r'checkpoints\iTransformer_xLSTM_VMD_Preprocessed\20260307_210815_iTransformer_xLSTM_VMD_Preprocessed\20260307_221302_test' 

# 文件名
WEIGHTS_FILE = os.path.join(DATA_DIR, 'mslstm_gate_weights.npy')
POWER_FILE = os.path.join(DATA_DIR, 'mslstm_input_power.npy') 

def plot_weights_line():
    # 1. 加载数据
    if not os.path.exists(WEIGHTS_FILE):
        print(f"错误: 找不到文件 {WEIGHTS_FILE}")
        return

    print(f">>> 正在加载 {WEIGHTS_FILE} ...")
    weights = np.load(WEIGHTS_FILE) 
    
    powers = None
    if os.path.exists(POWER_FILE):
        powers = np.load(POWER_FILE)
    else:
        print("提示: 未找到功率数据文件。")

    # 2. 选样逻辑 (这里保持和你原代码一致，选方差最小的样本，或者你可以改回 argmax 选波动大的)
    weight_variance = np.var(weights, axis=1).sum(axis=1) 
    # sample_idx = np.argmin(weight_variance) # 选最平稳的
    sample_idx = np.argmin(weight_variance) # [建议] 改为 argmax 选波动最剧烈的，曲线变化才明显
    
    # 如果想指定样本，取消下面注释
    # sample_idx = 641 
    for i in range(997):
        sample_idx = i
        print(f">>> 正在绘制样本 {sample_idx} 的曲线图...")
        
        sample_weights = weights[sample_idx] # [Seq_Len, 3]
        seq_len = sample_weights.shape[0]
        time_steps = np.arange(seq_len)
        
        # 准备绘图数据
        w_small = sample_weights[:, 0]  # k=7
        w_medium = sample_weights[:, 1] # k=15
        w_large = sample_weights[:, 2]  # k=25
        
        # ================= 开始绘图 =================
        if powers is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
            sample_power = powers[sample_idx]
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(12, 5))
            ax1 = None

        # --- 子图 1: 原始功率序列 ---
        if ax1 is not None:
            ax1.plot(time_steps, sample_power, color='black', linewidth=1.5, label='Wind Power')
            # ax1.set_title(f'Input Wind Power Sequence (Sample {sample_idx})', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Normalized Power')
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend(loc='upper right')

        # --- 子图 2: 多尺度权重曲线图 (核心修改) ---
        # 使用 plot 而不是 stackplot
        # 设置 linewidth 加粗线条，alpha 设置透明度防止完全遮挡
        
        # 小尺度 (k=7) - 红色实线
        ax2.plot(time_steps, w_small, color='#D62728', linewidth=2.5, linestyle='-', 
                label='Small Scale (k=7)', alpha=0.9)
        
        # 中尺度 (k=15) - 绿色虚线
        ax2.plot(time_steps, w_medium, color='#2CA02C', linewidth=2.0, linestyle='--', 
                label='Medium Scale (k=15)', alpha=0.9)
        
        # 大尺度 (k=25) - 蓝色点划线
        ax2.plot(time_steps, w_large, color='#1F77B4', linewidth=2.0, linestyle='-.', 
                label='Large Scale (k=25)', alpha=0.9)
        
        # ax2.set_title('Dynamic Evolution of Multi-Scale Gating Weights (Line Chart)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Attention Weight Probability')
        ax2.set_xlabel('Time Step')
        
        # 设置 Y 轴范围 [0, 1.05] 留一点顶部空间
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(0, seq_len-1)
        
        # 网格线
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # 图例放在底部
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=11)

        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(DATA_DIR, 'mslstm_case_study_line.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f">>> 图片已保存至: {save_path}")
        plt.show()

if __name__ == '__main__':
    plot_weights_line()