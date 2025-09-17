# 定义公共接口，指定哪些名称会被导入到 from package import * 语句中
__all__ = [
    '__version__',        # 版本号
    'DTCWTForward',       # 双树复小波变换前向变换
    'DTCWTInverse',       # 双树复小波变换逆变换
    'DWTForward',         # 离散小波变换前向变换（2D）
    'DWTInverse',         # 离散小波变换逆变换（2D）
    'DWT1DForward',       # 离散小波变换前向变换（1D）
    'DWT1DInverse',       # 离散小波变换逆变换（1D）
    'DTCWT',              # DTCWTForward 的别名
    'IDTCWT',             # DTCWTInverse 的别名
    'DWT',                # DWTForward 的别名
    'IDWT',               # DWTInverse 的别名
    'DWT1D',              # DWT1DForward 的别名
    'DWT2D',              # DWT 的别名（2D小波变换）
    'IDWT1D',             # DWT1DInverse 的别名
    'IDWT2D',             # IDWT 的别名（2D小波逆变换）
    'ScatLayer',          # 散射层
    'ScatLayerj2'         # 第二层散射层
]

# 导入版本信息
from pytorch_wavelets._version import __version__

# 从各个子模块导入主要类和函数
from pytorch_wavelets.dtcwt.transform2d import DTCWTForward, DTCWTInverse  # 导入双树复小波变换
from pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse        # 导入2D离散小波变换
from pytorch_wavelets.dwt.transform1d import DWT1DForward, DWT1DInverse    # 导入1D离散小波变换
from pytorch_wavelets.scatternet import ScatLayer, ScatLayerj2            # 导入散射网络层

# 定义一些别名，方便用户使用
DTCWT = DTCWTForward      # 双树复小波正向变换的简短别名
IDTCWT = DTCWTInverse     # 双树复小波逆变换的简短别名
DWT = DWTForward          # 2D小波正向变换的简短别名
IDWT = DWTInverse         # 2D小波逆变换的简短别名
DWT2D = DWT               # 2D小波变换的别名
IDWT2D = IDWT             # 2D小波逆变换的别名

DWT1D = DWT1DForward      # 1D小波正向变换的简短别名
IDWT1D = DWT1DInverse     # 1D小波逆变换的简短别名