import torch
import torch.nn as nn
from utils.RevIN import RevIN
from utils.DFS import DifferentiableFeatureSelector

"""
    DLinear（Decomposition-Linear）是一个用于时间序列预测的模型，其核心思想是将时间序列分解为季节性（seasonal）和趋势性（trend）两个组件，然后分别用线性层进行预测，最后将结果合并。
    1. moving_avg 类 - 移动平均模块
    2. series_decomp 类 - 序列分解模块
    
    平均池化
        平均池化在DLinear中的核心作用：
            信号分解：将复杂时间序列分解为趋势+季节性
            噪声抑制：通过平均操作减少随机波动
            趋势保留：维持数据的主要时间模式
            计算高效：相比复杂滤波器，计算成本低
            
        不同stride的影响
            stride = 1（重叠滑动） DLinear选择
                特点：窗口逐步移动，输出序列长度最大
                用途：保持时间分辨率，适合趋势提取
                DLinear使用：stride=1保持序列长度
            
            stride = kernel_size（非重叠）
                特点：窗口不重叠，输出序列长度大幅减少
                用途：下采样，减少数据量
                场景：数据压缩、计算加速
        
        为什么选择平均池化而不是其他方法？
            vs 最大池化
                最大池化：max([1, 5, 2]) = 5
                平均池化：avg([1, 5, 2]) = 2.67
                最大池化：保留峰值，丢失整体信息
                平均池化：保留整体趋势，更适合趋势分析
    
            vs 中位数滤波
                中位数：median([1, 5, 2]) = 2  
                平均值：mean([1, 5, 2]) = 2.67
                中位数：对异常值更鲁棒，但计算复杂
                平均值：计算简单，适合神经网络
    
            vs 高斯滤波
                高斯滤波：加权平均，中心权重大
                平均池化：等权重平均，实现简单
        
        kernel_size的选择
            在DLinear中，kernel_size=25：
                对于日数据，25天 ≈ 1个月的趋势
                对于小时数据，25小时 ≈ 1天多的趋势
            选择原则：
                太小：无法充分平滑噪声
                太大：可能过度平滑，丢失重要变化
                合适：平衡平滑效果和信息保留
            实际效果
                kernel_size=3: 短期平滑，保留更多细节
                kernel_size=25: 长期趋势，更强的平滑效果
                kernel_size=100: 极长期趋势，可能过度平滑
        
        什么是先验知识调参？
            先验知识调参指的是在使用某个模型或算法之前，需要根据对数据特性的先验了解来手动设置参数，而不是让模型自动学习这些参数。
            
            需要先验知识的方法
                傅里叶变换 - 需要知道主要频率成分
                frequencies = [1/365, 1/7]  # 需要知道年周期和周周期
            
            不需要先验知识的方法
                移动平均的kernel_size=25是通用设置，不需要领域知识
                这个25是如何确定的？
                    不是基于特定数据的周期性
                    不是基于特定领域的经验
                    而是一个在多种数据上都表现良好的通用值
                    类似于"经验法则"，但不需要用户了解
"""


class moving_avg(nn.Module):
    """
        实现移动平均操作，用于提取时间序列的趋势成分。

        为什么要填充？
            1. 保持序列长度不变
                例子设置
                    输入序列：[1, 2, 3, 4, 5]（长度=5）
                            kernel_size = 3
                            stride = 1
                    不填充的情况
                    AvgPool1d(kernel_size=3, stride=1, padding=0)
                    位置0: avg([1, 2, 3]) = 2.0
                    位置1: avg([2, 3, 4]) = 3.0
                    位置2: avg([3, 4, 5]) = 4.0
                    输出: [2.0, 3.0, 4.0]  # 长度从5变成3！ 从而导致输入时间步不同，就无法进行时间序列分解。 原序列 = 趋势成分 + 季节性成分 + 噪声成分

        不同填充方式的对比
            1. 零填充（Zero Padding）
                F.avg_pool1d(x, kernel_size=3, padding=1)
                优点：简单，计算标准
                缺点：在边界引入不真实的0值，影响趋势估计
            2. 反射填充（Reflection Padding） 镜像边界值
                原序列: [1, 2, 3, 4, 5]
                填充后: [2, 1, 1, 2, 3, 4, 5, 5, 4]
                优点：保持梯度连续性
                缺点：可能引入不真实的波动
            2. 重复填充（Replication Padding） DLinear使用的方式
                原序列: [1, 2, 3, 4, 5]
                填充后: [1, 1, 1, 2, 3, 4, 5, 5, 5]
                优点：保持梯度连续性
                缺点：可能引入不真实的波动

        DLinear为什么选择重复边界值？
            1. 连续性保持
                好的填充方式：重复边界值
                    [1, 1, 2, 3, 4, 5, 5]  # 连续过渡
                不好的填充方式：零填充
                    [0, 0, 1, 2, 3, 4, 5, 0, 0]  # 突然跳变
            2. 避免人工引入偏差
                零填充会在边界处引入偏差
                    原序列边界值: 1, 5
                    零填充平均: avg([0, 0, 1]) = 0.33  # 明显低估
                    重复填充平均: avg([1, 1, 1]) = 1.00  # 保持原值
            3. 符合时间序列特性
                时间序列的边界处，最合理的假设是趋势延续
                例如股价：如果最后价格是100，下一个价格更可能接近100而不是0
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # 平均池化（Average Pooling）是一种下采样操作，它通过计算滑动窗口内所有值的平均值来减少数据量，同时保留重要特征。
        # padding=0/1/2 在序列两端各添加0/1/2个零
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 计算需要的填充长度 并在前端填充和后端填充
        padding_length = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding_length, 1)
        end = x[:, -1:, :].repeat(1, padding_length, 1)
        x = torch.cat([front, x, end], dim=1)
        # (batch_size,seq_len,n_vars) -> (batch_size,n_vars,seq_len)
        x = x.permute(0, 2, 1)
        # 假如(64,2,96) 2为辐射量，功率。2和96合起来理解就是，要对辐射量、功率两个特征都进行池化。
        x = self.avg(x)
        # (batch_size,n_vars,seq_len) -> (batch_size,seq_len,n_vars)
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
        序列分解模块，用于将时间序列分解为趋势和季节性成分。

        为什么季节性成分是原序列减去趋势的残差?
            加法分解模型
                在时间序列分析中，最常用的分解模型是加法模型：
                    原序列 = 趋势成分 + 季节性成分 + 噪声成分
                    Y(t) = Trend(t) + Seasonal(t) + Noise(t)
            简化假设
                在DLinear中，为了简化模型，假设噪声相对较小，可以忽略或合并到季节性成分中：
                    Y(t) ≈ Trend(t) + Seasonal(t)

            例子：月度销售数据
                假设我们有一年的月度销售数据：
                    月份:     1   2   3   4   5   6   7   8   9  10  11  12
                    原序列:  100 110 120 140 150 160 170 180 190 210 220 230
                步骤1：提取趋势成分
                    通过移动平均（窗口=3）得到趋势：
                    趋势:    110 123 137 150 160 170 180 200 207 220
                步骤2：计算季节性成分
                    季节性 = 原序列 - 趋势
                    月份:     2   3   4   5   6   7   8   9  10  11
                    季节性:  -13  -3   3   0   0   0   0 -10  13   0
                步骤3：解释结果
                    2月: -13，表示2月销售比趋势低13个单位（春节影响）
                    4月: +3，表示4月销售比趋势高3个单位（春季促销）
                    10月: +13，表示10月销售比趋势高13个单位（节日促销）
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # 趋势Trend(t)
        res = x - moving_mean  # 季节性成分Seasonal(t)
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = configs.individual
        self.channels = configs.enc_in
        
        # 修改线性层，使其输出单变量预测
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # RevIn
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x /  stdev
        
        # 分解为季节性和趋势成分
        seasonal_init, trend_init = self.decompsition(x)
        
        # 调整维度 [batch, seq_len, n_vars] -> [batch, n_vars, seq_len]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)


        if self.individual:
            # 创建预测结果矩阵
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            # 对多变量进行预测
            for i in range(self.channels):
                # 对第i个变量进行预测 self.Linear_Seasonal[i]表示第i个变量的季节线性层权重
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # (batch_size,n_vars,seq_len) -> (batch_size,n_vars,pred_len)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)

        # InReVin
        x = x * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        x = x + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return x[:, :, self.channels - 1:self.channels]  # 输出形状: [Batch, Output length, 1]