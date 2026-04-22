"""
统一的VMD (Variational Mode Decomposition) 分解模块。

支持两种模式:
1. StandardVMD - 标准固定K模态分解
2. SparseAdaptiveVMD - 稀疏自适应VMD，可学习门控自动选择有效模态

所有VMD相关代码集中于此，消除多文件重复定义。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 尝试导入 vmdpy，如果不存在则忽略 (训练通常只用GPU分支)
try:
    import vmdpy
except ImportError:
    vmdpy = None


class StandardVMD(nn.Module):
    """
    标准VMD分解，固定K个模态，每个模态有独立可学习带宽参数。

    支持两种实现:
    - CPU: 使用vmdpy库的精确VMD分解 (推理/debug用)
    - GPU: 可微FFT滤波器组近似实现 (训练用)
    """

    def __init__(
        self,
        K: int = 8,
        impl: str = 'fftbank',
        alpha: float = 2000.0,
        tau: float = 0.0,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-7,
        max_iter: int = 500,
        trace_spectrum: bool = True
    ):
        """
        Args:
            K: 分解模态数量
            impl: 实现方式 'vmdpy' | 'fftbank' | 'auto'
            alpha: 初始带宽控制参数 (alpha越大带宽越窄)
            tau: 梯度步长 (仅vmdpy使用)
            DC: 是否保留直流分量 (仅vmdpy使用)
            init: 初始化方法 (仅vmdpy使用)
            tol: 收敛阈值 (仅vmdpy使用)
            max_iter: 最大迭代次数 (仅vmdpy使用)
            trace_spectrum: 是否缓存频谱数据用于可视化
        """
        super().__init__()
        self.K = K
        self.impl = impl
        self.trace_spectrum = trace_spectrum
        self.last_spectral_data = None
        self.last_time_data = None

        # 每个模态独立的带宽控制参数
        # alpha越大 -> sigma越小 -> 带宽越窄
        initial_val = torch.full((K,), float(alpha))
        noise = torch.randn(K) * 0.05
        self.vmd_alpha = nn.Parameter(initial_val + noise, requires_grad=True)

        # 整体频带缩放参数 (控制中心频率分布)
        self.vmd_k = nn.Parameter(torch.tensor(float(K)), requires_grad=True)

        # VMD原始参数
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter

        # 检测vmdpy是否可用
        self._use_vmdpy = False
        if impl in ('auto', 'vmdpy') and vmdpy is not None:
            self._use_vmdpy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, 1] 输入单变量序列

        Returns:
            modes: [B, L, K] 分解后的K个模态
        """
        # x: [B, L, 1]
        B, L, C = x.shape
        assert C == 1, f"VMDDecomposer expects [B, L, 1], got {x.shape}"

        if self._use_vmdpy:
            # --- CPU vmdpy 实现 (通常仅用于非训练阶段或debug) ---
            # 注意: 标准vmdpy不支持每个模态不同alpha，这里取平均值
            modes_list = []
            x_np = x.detach().cpu().numpy()
            alpha_scalar = float(self.vmd_alpha.mean().item())

            for b in range(B):
                signal = x_np[b, :, 0].astype(np.float64)
                u, _, _ = vmdpy.VMD(
                    signal,
                    alpha=alpha_scalar,
                    tau=self.tau,
                    K=self.K,
                    DC=self.DC,
                    init=self.init,
                    tol=self.tol
                )
                # u: [K, L] -> 转置为 [L, K]
                modes_list.append(torch.from_numpy(u.T.astype(np.float32)))

            modes = torch.stack(modes_list, dim=0).to(x.device)  # [B, L, K]
            return modes
        else:
            # --- GPU 可微FFT-Bank实现 (核心训练路径) ---

            # 1. 实数FFT [B, L, 1] -> [B, Nf]
            x_freq = torch.fft.rfft(x.squeeze(-1), dim=1, norm='ortho')
            B, n_freq = x_freq.shape
            K_fixed = self.K

            # 2. 生成频率轴索引 [Nf, 1]
            freq_idx = torch.arange(n_freq, device=x.device).float().unsqueeze(-1)

            # 3. 计算频带划分比例k_scale (标量)
            # 将参数限制在 (0.5, 1.5) 防止频谱中心跑偏太远
            k_scale = torch.sigmoid(self.vmd_k)

            # 4. 处理alpha (向量化)，softplus确保为正数
            alpha_eff = F.softplus(self.vmd_alpha) + 1e-6  # [K]

            # 5. 计算每个模态的带宽sigma (向量化)
            # sigma = n_freq / (K * alpha_eff)
            sigma = n_freq / (K_fixed * alpha_eff)  # [K]

            # 6. 计算中心频率 centers (向量化)
            centers = (torch.arange(K_fixed, device=x.device).float() + 0.5) \
                      * (n_freq / K_fixed) * k_scale  # [K]

            # 7. 计算高斯权重 (广播机制)
            # freq_idx: [Nf, 1], centers: [K], sigma: [K]
            diffs = (freq_idx - centers) / sigma  # [Nf, K]
            w = torch.exp(-0.5 * (diffs ** 2))  # [Nf, K]

            # 归一化: 每个频率点上，所有模态权重和为1
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # [Nf, K]

            # 8. 频域滤波 [B, Nf, K]
            x_freq_expanded = x_freq.unsqueeze(-1) * w.unsqueeze(0)

            # 9. 逆FFT变换回时域
            x_freq_flat = x_freq_expanded.permute(0, 2, 1).reshape(B * K_fixed, n_freq)
            x_time_flat = torch.fft.irfft(x_freq_flat, n=L, dim=1, norm='ortho')

            # 10. 重组输出 [B, L, K]
            modes = x_time_flat.reshape(B, K_fixed, L).permute(0, 2, 1)

            if self.trace_spectrum:
                self.last_spectral_data = {
                    'mode_spectra': torch.abs(x_freq_expanded).detach().cpu(),
                    'filters': w.detach().cpu(),
                    'freq_idx': freq_idx.squeeze().detach().cpu()
                }
                self.last_time_data = {
                    'input': x.detach().cpu(),
                    'modes': modes.detach().cpu()
                }

            return modes


class SparseAdaptiveVMD(nn.Module):
    """
    稀疏自适应VMD (S-ABDM)。

    改进特点:
    1. 引入可学习门控机制，自动抑制无效模态
    2. 配合稀疏损失，可以自动学习出最佳有效模态数量
    3. K代表K_max (最大模态池容量)，建议设置较大如12或16
    """

    def __init__(
        self,
        K: int = 12,
        impl: str = 'fftbank',
        alpha: float = 2000.0,
        tau: float = 0.0,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-7,
        max_iter: int = 500,
        trace_spectrum: bool = False
    ):
        """
        Args:
            K: 最大模态池容量 (K_max)，建议设为12或16
            impl: 实现方式 'vmdpy' | 'fftbank' | 'auto'
            alpha: 初始带宽控制参数
            tau: 梯度步长 (仅vmdpy使用)
            DC: 是否保留直流分量 (仅vmdpy使用)
            init: 初始化方法 (仅vmdpy使用)
            tol: 收敛阈值 (仅vmdpy使用)
            max_iter: 最大迭代次数 (仅vmdpy使用)
            trace_spectrum: 是否缓存频谱数据用于可视化
        """
        super().__init__()
        self.K = K
        self.impl = impl
        self.trace_spectrum = trace_spectrum
        self.last_spectral_data = None
        self.last_time_data = None

        # 1. 带宽控制参数 (每个模态独立)
        initial_alpha = torch.full((K,), float(alpha))
        noise = torch.randn(K) * 0.1
        self.vmd_alpha = nn.Parameter(initial_alpha + noise, requires_grad=True)

        # 2. 整体频带缩放参数
        self.vmd_k_scale = nn.Parameter(torch.tensor(float(K)), requires_grad=True)

        # 3. [核心] 模态门控参数 (可学习，自动选择有效模态)
        # 初始化为0，Sigmoid(0)=0.5，所有模态初始活性50%
        self.mode_gate_logits = nn.Parameter(torch.zeros(K), requires_grad=True)

        # VMD原始参数
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter

        # 检测vmdpy是否可用
        self._use_vmdpy = (impl in ('auto', 'vmdpy') and vmdpy is not None)

    def get_sparsity_loss(self) -> torch.Tensor:
        """
        计算L1稀疏损失，鼓励减少激活模态数量。

        Returns:
            loss: 标量稀疏损失
        """
        gates = torch.sigmoid(self.mode_gate_logits)
        return torch.mean(gates)

    def get_effective_k(self, threshold: float = 0.1) -> int:
        """
        获取当前实际上激活的模态数量。

        Args:
            threshold: 门控值阈值，大于此值视为激活

        Returns:
            激活模态数量
        """
        with torch.no_grad():
            gates = torch.sigmoid(self.mode_gate_logits)
            return (gates > threshold).sum().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, 1] 输入单变量序列

        Returns:
            modes: [B, L, K_max] 分解后的模态，已应用门控
        """
        # x: [B, L, 1]
        B, L, C = x.shape
        assert C == 1, f"SparseAdaptiveVMD expects [B, L, 1], got {x.shape}"

        if self._use_vmdpy and not self.training:
            # --- CPU vmdpy实现 (仅在非训练且强制要求时作为fallback) ---
            modes_list = []
            x_np = x.detach().cpu().numpy()
            alpha_scalar = float(self.vmd_alpha.mean().item())

            for b in range(B):
                signal = x_np[b, :, 0].astype(np.float64)
                u, _, _ = vmdpy.VMD(
                    signal,
                    alpha=alpha_scalar,
                    tau=self.tau,
                    K=self.K,
                    DC=self.DC,
                    init=self.init,
                    tol=self.tol
                )
                modes_list.append(torch.from_numpy(u.T.astype(np.float32)))

            modes = torch.stack(modes_list, dim=0).to(x.device)

            # 后处理应用门控
            gates = torch.sigmoid(self.mode_gate_logits).view(1, 1, self.K)
            modes = modes * gates
            return modes
        else:
            # --- GPU 可微FFT-Bank实现 (训练核心路径) ---

            # 1. 实数FFT
            x_freq = torch.fft.rfft(x.squeeze(-1), dim=1, norm='ortho')  # [B, Nf]
            B, n_freq = x_freq.shape
            K_fixed = self.K

            # 2. 频率轴
            freq_idx = torch.arange(n_freq, device=x.device).float().unsqueeze(-1)  # [Nf, 1]

            # 3. 计算频带参数
            k_scale_val = torch.sigmoid(self.vmd_k_scale)
            alpha_eff = F.softplus(self.vmd_alpha) + 1e-6  # [K]
            sigma = torch.clamp(n_freq / (K_fixed * alpha_eff), min=1.0)  # [K]
            centers = (torch.arange(K_fixed, device=x.device).float() + 0.5) \
                      * (n_freq / K_fixed) * k_scale_val  # [K]

            # 4. 生成高斯滤波器组
            diffs = (freq_idx - centers) / sigma  # [Nf, K]
            w = torch.exp(-0.5 * (diffs ** 2))  # [Nf, K]
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # [Nf, K]

            # 5. [核心] 应用可学习门控
            gates = torch.sigmoid(self.mode_gate_logits).view(1, 1, K_fixed)  # [1, 1, K]

            # 6. 频域滤波 + 门控
            x_freq_expanded = x_freq.unsqueeze(-1) * w.unsqueeze(0)  # [B, Nf, K]
            x_freq_gated = x_freq_expanded * gates  # [B, Nf, K]

            # 7. iFFT回时域
            x_freq_flat = x_freq_gated.permute(0, 2, 1).reshape(B * K_fixed, n_freq)
            x_time_flat = torch.fft.irfft(x_freq_flat, n=L, dim=1, norm='ortho')

            # [B*K, L] -> [B, K, L] -> [B, L, K]
            modes = x_time_flat.reshape(B, K_fixed, L).permute(0, 2, 1)

            if self.trace_spectrum:
                self.last_spectral_data = {
                    'mode_spectra': torch.abs(x_freq_gated).detach().cpu(),
                    'filters': w.detach().cpu(),
                    'gates': gates.squeeze().detach().cpu()
                }
                self.last_time_data = {
                    'input': x.detach().cpu(),
                    'modes': modes.detach().cpu()
                }

            return modes


class VMDDecomposer(nn.Module):
    """
    统一VMD分解器，根据配置选择StandardVMD或SparseAdaptiveVMD。

    Config参数:
        vmd_mode: 'standard' (默认) 或 'sparse_adaptive'
        vmd_k: 分解模态数量 (standard) 或 K_max (sparse_adaptive)
        vmd_impl: 'fftbank' (默认) 或 'vmdpy'
        vmd_alpha: 初始alpha值，默认2000.0
        ... 其他VMD参数
    """

    def __init__(self, configs):
        super().__init__()
        vmd_mode = getattr(configs, 'vmd_mode', 'standard')
        vmd_k = getattr(configs, 'vmd_k', 8)
        vmd_impl = getattr(configs, 'vmd_impl', 'fftbank')
        vmd_alpha = getattr(configs, 'vmd_alpha', 2000.0)
        vmd_tau = getattr(configs, 'vmd_tau', 0.0)
        vmd_dc = getattr(configs, 'vmd_dc', 0)
        vmd_init = getattr(configs, 'vmd_init', 1)
        vmd_tol = getattr(configs, 'vmd_tol', 1e-7)
        vmd_max_iter = getattr(configs, 'vmd_max_iter', 500)
        trace_spectrum = getattr(configs, 'trace_spectrum', False)

        if vmd_mode == 'standard':
            self.decomposer = StandardVMD(
                K=vmd_k,
                impl=vmd_impl,
                alpha=vmd_alpha,
                tau=vmd_tau,
                DC=vmd_dc,
                init=vmd_init,
                tol=vmd_tol,
                max_iter=vmd_max_iter,
                trace_spectrum=trace_spectrum
            )
        elif vmd_mode == 'sparse_adaptive':
            self.decomposer = SparseAdaptiveVMD(
                K=vmd_k,
                impl=vmd_impl,
                alpha=vmd_alpha,
                tau=vmd_tau,
                DC=vmd_dc,
                init=vmd_init,
                tol=vmd_tol,
                max_iter=vmd_max_iter,
                trace_spectrum=trace_spectrum
            )
        else:
            raise ValueError(f"Unknown vmd_mode: {vmd_mode}, expected 'standard' or 'sparse_adaptive'")

        self.vmd_mode = vmd_mode
        self.vmd_k = vmd_k

    def get_sparsity_loss(self):
        """获取稀疏损失 (仅sparse_adaptive模式有效)"""
        if self.vmd_mode == 'sparse_adaptive':
            return self.decomposer.get_sparsity_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_effective_k(self, threshold=0.1):
        """获取有效模态数量 (仅sparse_adaptive模式有效)"""
        if self.vmd_mode == 'sparse_adaptive':
            return self.decomposer.get_effective_k(threshold)
        return self.vmd_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decomposer(x)
