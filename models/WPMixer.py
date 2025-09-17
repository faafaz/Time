# 导入必要的模块和类
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import Permute, Reshape
from utils.RevIN import RevIN

import matplotlib.pyplot as plt
import numpy as np
from layers.wavelet_patch_mixer import WPMixerCore

# 短期预测的WPMixer包装类
class Model(nn.Module):
    def __init__(self, configs):        # 是否使用自动混合精度
        super(Model, self).__init__()
        # 初始化核心WPMixer模型
        self.model = WPMixer(c_in = 1, 
                             c_out = 1, 
                             seq_len = 96, 
                             out_len = 8, 
                             d_model = 256,
                            dropout = 0.05, 
                            embedding_dropout = 0.05, 
                            device = "cuda" if torch.cuda.is_available() else "cpu", 
                            batch_size = 128,
                            tfactor = 5, 
                            dfactor = 5, 
                            wavelet = "db2", 
                            level = 3, 
                            patch_len = 16,
                            stride = 8, 
                            no_decomposition = False,
                            use_amp = False)
        
    def forward(self, x, _unknown1, _unknown2, _unknown3):
        x = x[:, :, 1:2]
        # 前向传播函数，忽略额外的未知参数
        out = self.model(x)
        return out
    

# WPMixer主模型类
class WPMixer(nn.Module):
    def __init__(self,
                 c_in = [],           # 输入通道数
                 c_out = [],          # 输出通道数
                 seq_len = [],        # 输入序列长度
                 out_len = [],        # 预测序列长度
                d_model = [],         # 模型维度
                dropout = [],         # dropout比率
                embedding_dropout = [], # 嵌入层dropout比率
                device = [],          # 计算设备
                batch_size = [],      # 批次大小
                tfactor = [],         # 时间因子
                dfactor = [],         # 维度因子
                wavelet = [],         # 小波名称
                level = [],           # 小波分解层数
                patch_len = [],       # patch长度
                stride = [],          # patch步长
                no_decomposition = [], # 是否不进行分解
                use_amp = []):        # 是否使用自动混合精度
        
        super(WPMixer, self).__init__()
        # 设置模型基本参数
        self.pred_len = out_len              # 预测长度
        self.channel_in = c_in               # 输入通道数
        self.channel_out = c_out             # 输出通道数
        self.patch_len = patch_len           # patch长度
        self.stride = stride                 # patch步长
        self.seq_len = seq_len               # 序列长度
        self.d_model = d_model               # 模型维度
        self.dropout = dropout               # dropout比率
        self.embedding_dropout = embedding_dropout  # 嵌入层dropout比率
        self.batch_size = batch_size         # 批次大小
        self.tfactor = tfactor               # 时间因子
        self.dfactor = dfactor               # 维度因子
        self.wavelet = wavelet               # 小波名称
        self.level = level                   # 小波分解层数
        # patch预测器相关参数
        self.actual_seq_len = seq_len        # 实际序列长度
        self.no_decomposition = no_decomposition  # 是否不进行分解
        self.use_amp = use_amp               # 是否使用自动混合精度
        self.device = device                 # 计算设备
        
        # 初始化WPMixer核心组件
        self.wpmixerCore = WPMixerCore(input_length = self.actual_seq_len,
                                                      pred_length = self.pred_len,
                                                      wavelet_name = self.wavelet,
                                                      level = self.level,
                                                      batch_size = self.batch_size,
                                                      channel = self.channel_in, 
                                                      d_model = self.d_model, 
                                                      dropout = self.dropout, 
                                                      embedding_dropout = self.embedding_dropout,
                                                      tfactor = self.tfactor, 
                                                      dfactor = self.dfactor, 
                                                      device = self.device,
                                                      patch_len = self.patch_len, 
                                                      patch_stride = self.stride,
                                                      no_decomposition = self.no_decomposition,
                                                      use_amp = self.use_amp)
        
        
    def forward(self, x):
        # 前向传播函数
        pred = self.wpmixerCore(x)           # 通过核心组件进行预测
        pred = pred[:, :, -self.channel_out:] # 只取最后channel_out个通道的输出
        return pred 