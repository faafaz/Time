# 导入必要的模块
import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch


class DWT1DForward(nn.Module):
    """ 执行图像的一维离散小波变换（DWT）前向分解

    参数:
        J (int): 分解的层数
        wave (str 或 pywt.Wavelet 或 tuple(ndarray)): 使用哪种小波.
            可以是:
            1) 传递给 pywt.Wavelet 构造函数的字符串
            2) pywt.Wavelet 类
            3) numpy数组元组 (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' 或 'periodization'。填充方案
        """
    def __init__(self, J=1, wave='db1', mode='zero', use_amp = False):
        super().__init__()
        self.use_amp = use_amp  # 是否使用自动混合精度
        # 根据输入类型解析小波
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, h1 = wave.dec_lo, wave.dec_hi  # 获取分解滤波器
        else:
            assert len(wave) == 2
            h0, h1 = wave[0], wave[1]

        # 准备滤波器 - 将它们转换为列滤波器
        filts = lowlevel.prep_filt_afb1d(h0, h1)
        self.register_buffer('h0', filts[0])  # 低通滤波器
        self.register_buffer('h1', filts[1])  # 高通滤波器
        self.J = J      # 分解层数
        self.mode = mode  # 填充模式

    def forward(self, x):
        """ DWT的前向传递

        参数:
            x (tensor): 输入形状为 :math:`(N, C_{in}, L_{in})`

        返回:
            (yl, yh)
                低通系数(yl)和带通系数(yh)的元组.
                yh是一个长度为J的列表，第一个条目是最高频系数.
        """
        assert x.ndim == 3, "只能处理3维输入 (N, C, L)"  # 确保输入是3维
        highs = []  # 存储高频系数
        x0 = x     # 初始化低频系数
        mode = lowlevel.mode_to_int(self.mode)  # 将模式转换为整数

        # 执行多层变换
        for j in range(self.J):
            # 应用一级滤波器组
            x0, x1 = lowlevel.AFB1D.apply(x0, self.h0, self.h1, mode, self.use_amp)
            highs.append(x1)  # 存储高频系数

        return x0, highs  # 返回低频系数和高频系数列表


class DWT1DInverse(nn.Module):
    """ 执行图像的一维离散小波变换（DWT）逆变换重构

    参数:
        wave (str 或 pywt.Wavelet 或 tuple(ndarray)): 使用哪种小波.
            可以是:
            1) 传递给 pywt.Wavelet 构造函数的字符串
            2) pywt.Wavelet 类
            3) numpy数组元组 (g0, g1)
        mode (str): 'zero', 'symmetric', 'reflect' 或 'periodization'。填充方案
    """
    def __init__(self, wave='db1', mode='zero', use_amp = False):
        super().__init__()
        self.use_amp = use_amp  # 是否使用自动混合精度
        # 根据输入类型解析小波
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi  # 获取重构滤波器
        else:
            assert len(wave) == 2
            g0, g1 = wave[0], wave[1]

        # 准备滤波器
        filts = lowlevel.prep_filt_sfb1d(g0, g1)
        self.register_buffer('g0', filts[0])  # 重构低通滤波器
        self.register_buffer('g1', filts[1])  # 重构高通滤波器
        self.mode = mode  # 填充模式

    def forward(self, coeffs):
        """
        参数:
            coeffs (yl, yh): 低通和带通系数的元组，应与DWT1DForward返回的格式匹配.

        返回:
            重构后的输入，形状为 :math:`(N, C_{in}, L_{in})`

        注意:
            任何高频尺度可以为None，将被视为零值（尽管不是高效的方式）.
        """
        x0, highs = coeffs  # 获取低频系数和高频系数列表
        assert x0.ndim == 3, "只能处理3维输入 (N, C, L)"  # 确保输入是3维
        mode = lowlevel.mode_to_int(self.mode)  # 将模式转换为整数
        # 执行多层逆变换
        for x1 in highs[::-1]:  # 从最高频到最低频逆序处理
            if x1 is None:
                x1 = torch.zeros_like(x0)  # 如果高频系数为None，则设为零

            # '取消填充' 添加的信号
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]  # 调整尺寸以匹配
            # 应用合成滤波器组
            x0 = lowlevel.SFB1D.apply(x0, x1, self.g0, self.g1, mode, self.use_amp)
        return x0  # 返回重构后的信号