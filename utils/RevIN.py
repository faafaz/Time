# 代码来自 https://github.com/ts-kim/RevIN，经过轻微修改

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        RevIN (Reversible Instance Normalization) 可逆实例归一化
        
        :param num_features: 特征数或通道数
        :param eps: 为数值稳定性添加的值
        :param affine: 如果为True，RevIN具有可学习的仿射参数
        :param subtract_last: 是否减去最后一个值而不是减去均值
        """
        super(RevIN, self).__init__()
        self.num_features = num_features  # 特征数量
        self.eps = eps                    # 数值稳定性常数
        self.affine = affine              # 是否使用仿射变换
        self.subtract_last = subtract_last  # 是否使用最后一个值进行归一化
        if self.affine:
            self._init_params()           # 初始化仿射参数

    def forward(self, x, mode:str):
        """
        前向传播函数
        
        :param x: 输入张量
        :param mode: 操作模式，'n'表示归一化，'d'表示反归一化
        """
        if mode == 'n':
            self._get_statistics(x)       # 获取统计信息
            x = self._normalize(x)        # 归一化
        elif mode == 'd':
            x = self._denormalize(x)      # 反归一化
        else: raise NotImplementedError   # 不支持的模式
        return x

    def _init_params(self):
        # 初始化RevIN参数: (C,)
        # 仿射权重参数，初始化为1
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        # 仿射偏置参数，初始化为0
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        计算输入数据的统计信息（均值和标准差）
        
        :param x: 输入张量
        """
        # 确定需要reduce的维度（除了batch维度和最后一个特征维度）
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            # 如果subtract_last为True，保存最后一个时间步的数据
            self.last = x[:,-1,:].unsqueeze(1) # batch中最后一个样本
        else:
            # 计算均值并detach（不参与梯度计算）
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach() 
        # 计算标准差并detach（不参与梯度计算）
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """
        对输入数据进行归一化
        
        :param x: 输入张量
        """
        if self.subtract_last:
            # 如果使用subtract_last，减去最后一个时间步的值
            x = x - self.last
        else:
            # 否则减去均值
            x = x - self.mean
        # 除以标准差进行标准化
        x = x / self.stdev
        if self.affine:
            # 如果使用仿射变换，应用可学习的权重和偏置
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        对归一化后的数据进行反归一化
        
        :param x: 归一化后的张量
        """
        if self.affine:
            # 如果使用仿射变换，先进行反变换
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)  # 防止除零
        # 乘以标准差
        x = x * self.stdev
        if self.subtract_last:
            # 如果使用subtract_last，加上最后一个时间步的值
            x = x + self.last
        else:
            # 否则加上均值
            x = x + self.mean
        return x