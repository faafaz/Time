import torch
import torch.nn as nn

"""
对每个时间点的特征向量做线性变换
nn.Conv2d(in_channels, out_channels, kernel_size=1)
seq_len * in_channels 要输出成 seq_len * out_channels
也就是说参数数量是 in_channels * out_channels
# 在时刻1：
    输入特征 = [1.0, 5.0, 9.0]  # 3个embedding特征
    权重1   = [0.1, 0.2, 0.3]  # 第1个输出特征的权重
    权重2   = [0.4, 0.5, 0.6]  # 第2个输出特征的权重
    
    新特征1 = 1.0×0.1 + 5.0×0.2 + 9.0×0.3 = 3.8
    新特征2 = 1.0×0.4 + 5.0×0.5 + 9.0×0.6 = 7.9
    
    输出 = [3.8, 7.9]  # 2个新特征
    关键点：
    
    ⏰ 时间维度不变 - 每个时间点独立处理
    🔄 权重共享 - 所有时间点用相同权重
    🎯 特征混合 - 学习最优的特征组合
    📊 维度变换 - 3个特征 → 2个特征

全连接层与Linear层的区别
    Linear层的参数只依赖输入和输出维度， 参数数量都是固定的 当前维度*目标维度
    全连接层处理序列数据时，通常需要将整个序列**拉平(flatten)**后输入：
        序列: [seq_len, feature_dim] → 拉平后: [seq_len × feature_dim]
        参数数量 = (seq_len × feature_dim) × output_dim + output_dim
        可以看到，参数数量直接与seq_len成正比。
"""


class Inception_Block_V1(nn.Module):
    """
        参数in_channels=d_model,out_channels=d_ff
    """

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            """
                输出尺寸 = (输入尺寸 + 2×padding - kernel_size) / stride + 1,此处stride=1
                输出尺寸 = 输入尺寸 + 2×padding - kernel_size + 1
                为什么Conv2d的padding=i?
                    卷积核编号   kernel_size     padding=0的输出    padding=i的输出
                    i=0         1×1            32×32             32×32 
                    i=1         3×3            30×30             32×32 
                    i=2         5×5            28×28             32×32 
                    i=3         7×7            26×26             32×32
                也就是说在Conv2d中padding会乘以2么，Conv1d中padding乘以1?
                答：错。
                    Conv1d: padding × 2  # 不是 × 1！
                    Conv2d: padding × 2  # 每个维度都是两边加padding
                    Conv3d: padding × 2  # 每个维度都是两边加padding
                    1D: 序列的左边和右边
                    2D: 上下两边 + 左右两边
                    3D: 前后 + 上下 + 左右，每个维度都是两边
                padding的具体数值是多少？
                    nn.Conv2d(3, 16, kernel_size=3, padding=1, padding_mode='zeros')     # 用0填充（默认）
                    zeros/reflect/replicate/circular  用0填充/反射填充/复制边缘值/循环填充
            """
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:  # 卷积层的偏置参数（bias）
                    nn.init.constant_(m.bias, 0)  # 将偏置初始化为0

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        """
            以下代码实现了多尺度特征融合
            # 不同卷积核捕获不同尺度的特征
            1x1 卷积核 → 捕获点特征
            3x3 卷积核 → 捕获局部特征  
            5x5 卷积核 → 捕获中等范围特征
            7x7 卷积核 → 捕获更大范围特征
            
            # 通过平均融合所有尺度的信息
            最终输出 = (1x1特征 + 3x3特征 + 5x5特征 + 7x7特征) / 4
        """
        # stacked.shape: [B, C, H, W, res_list.len]
        res = torch.stack(res_list, dim=-1)
        # 对最后一个维度（即第-1维）求平均值
        res = res.mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            """
                V1: 使用正方形卷积核
                    1×1, 3×3, 5×5, 7×7...
                V2: 使用非对称卷积核(长方形卷积核)  
                    1×3与3×1, 1×5和5×1, 1×7和7×1
                
                V1与V2卷积核参数大小
                (V1) 5×5卷积 → 25个参数
                (V2) 5×5卷积 → 1×5卷积 + 5×1卷积 → 5 + 5 = 10个参数
                参数减少 60%！
                [注]卷积核的参数是固定的,不会因为seq_len的变化而变化,序列中会共享卷积核的权重。
                    比如3*3的卷积核，权重就是9。而不是每9个像素就有一个3*3的卷积核权重参数。
            """
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        """
            1×3, 3×1 卷积 → 捕获方向性、空间模式
            1×1      卷积 → 捕获通道间关系、点特征
            这里1×1卷积相当于一个全连接层，然而用全连接层的话参数会很多，且随着序列长度增加而增加(具体看此文件最上面的注释)，故使用1×1卷积
        """
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        # 由于num_kernels=6 故总共：6个非对称卷积 + 1个1×1卷积 = 7个卷积核
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        #  + 1是因为另外还有一个1*1的卷积核
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))

        # stacked.shape: [B, C, H, W, res_list.len]
        res = torch.stack(res_list, dim=-1)
        # 对最后一个维度（即第-1维）求平均值
        res = res.mean(-1)
        return res
