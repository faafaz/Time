# 导入必要的模块和类
import torch.nn as nn
import torch
import numpy as np
from utils.RevIN import RevIN
from layers.decomposition import Decomposition

class WPMixerCore(nn.Module):
    def __init__(self, 
                 input_length = [],        # 输入序列长度
                 pred_length = [],         # 预测序列长度
                 wavelet_name = [],        # 小波名称
                 level = [],               # 小波分解层数
                 batch_size = [],          # 批次大小
                 channel = [],             # 通道数
                 d_model = [],             # 模型维度
                 dropout = [],             # dropout比率
                 embedding_dropout = [],   # 嵌入层dropout比率
                 tfactor = [],             # 时间因子
                 dfactor = [],             # 维度因子
                 device = [],              # 计算设备
                 patch_len = [],           # patch长度
                 patch_stride = [],        # patch步长
                 no_decomposition = [],    # 是否不进行分解
                 use_amp = []):            # 是否使用自动混合精度
        
        super(WPMixerCore, self).__init__()
        # 初始化各种参数
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition 
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp
        
        # 初始化分解模型
        self.Decomposition_model = Decomposition(input_length = self.input_length, 
                                        pred_length = self.pred_length,
                                        wavelet_name = self.wavelet_name,
                                        level = self.level,
                                        batch_size = self.batch_size,
                                        channel = self.channel,
                                        d_model = self.d_model,
                                        tfactor = self.tfactor,
                                        dfactor = self.dfactor,
                                        device = self.device,
                                        no_decomposition = self.no_decomposition,
                                        use_amp = self.use_amp)
        
        # 获取分解后输入和预测系数序列的维度
        self.input_w_dim = self.Decomposition_model.input_w_dim # 输入系数序列长度列表
        self.pred_w_dim = self.Decomposition_model.pred_w_dim   # 预测系数序列长度列表

        self.patch_len = patch_len
        self.patch_stride = patch_stride
        
        # 创建(m+1)个分辨率分支，对应不同分辨率的系数
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq = self.input_w_dim[i],
                                                           pred_seq = self.pred_w_dim[i],
                                                           batch_size = self.batch_size,
                                                           channel = self.channel,
                                                           d_model = self.d_model,
                                                           dropout = self.dropout,
                                                           embedding_dropout = self.embedding_dropout,
                                                           tfactor = self.tfactor,
                                                           dfactor = self.dfactor,
                                                           patch_len = self.patch_len,
                                                           patch_stride = self.patch_stride) for i in range(len(self.input_w_dim))])
        
        # 初始化RevIN归一化层
        self.revin = RevIN(self.channel, eps=1e-5, affine = True, subtract_last = False)
        
    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : 回望窗口数据: [Batch, look_back_length, channel]

        Returns
        -------
        xT : 预测时间序列: [Batch, prediction_length, output_channel]
        '''
        
        # 对输入数据进行RevIN归一化
        x = self.revin(xL, 'n')
        x = x.transpose(1, 2) # [batch, channel, look_back_length]
        
        # xA: 近似系数序列, 
        # xD: 细节系数序列
        # yA: 预测的近似系数序列
        # yD: 预测的细节系数序列
        
        # 对输入数据进行小波分解
        xA, xD = self.Decomposition_model.transform(x) 
        
        # 通过不同分辨率分支处理系数序列
        yA = self.resolutionBranch[0](xA)  # 处理近似系数
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])  # 处理细节系数
            yD.append(yD_i)
        
        # 对预测的系数进行小波重构
        y = self.Decomposition_model.inv_transform(yA, yD) 
        y = y.transpose(1, 2)
        y = y[:, -self.pred_length:, :] # 分解输出总是偶数，但预测长度可能是奇数
        # 对输出数据进行RevIN反归一化
        xT = self.revin(y, 'd')
        
        return xT


class ResolutionBranch(nn.Module):
    def __init__(self, 
                 input_seq = [],           # 输入序列长度
                 pred_seq = [],            # 预测序列长度
                 batch_size = [],          # 批次大小
                 channel = [],             # 通道数
                 d_model = [],             # 模型维度
                 dropout = [],             # dropout比率
                 embedding_dropout = [],   # 嵌入层dropout比率
                 tfactor = [],             # 时间因子
                 dfactor = [],             # 维度因子
                 patch_len = [],           # patch长度
                 patch_stride = []):       # patch步长
        super(ResolutionBranch, self).__init__()
        # 初始化各种参数
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len 
        self.patch_stride = patch_stride 
        # 计算patch数量
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)
        
        # patch归一化层
        self.patch_norm = nn.BatchNorm2d(self.channel)
        # patch嵌入层
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model) # 在所有通道间共享
        # 两个Mixer层
        self.mixer1 = Mixer(input_seq = self.patch_num, 
                            out_seq = self.patch_num,
                            batch_size = self.batch_size,
                            channel = self.channel,
                            d_model = self.d_model,
                            dropout = self.dropout,
                            tfactor = self.tfactor, 
                            dfactor = self.dfactor)
        self.mixer2 = Mixer(input_seq = self.patch_num, 
                            out_seq = self.patch_num, 
                            batch_size = self.batch_size, 
                            channel = self.channel,
                            d_model = self.d_model,
                            dropout = self.dropout,
                            tfactor = self.tfactor, 
                            dfactor = self.dfactor)
        # 归一化层
        self.norm = nn.BatchNorm2d(self.channel)
        # dropout层
        self.dropoutLayer = nn.Dropout(self.embedding_dropout) 
        # 输出头
        self.head = nn.Sequential(nn.Flatten(start_dim = -2 , end_dim = -1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_seq))
        # RevIN归一化层
        self.revin = RevIN(self.channel)
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : 输入系数序列: [Batch, channel, length_of_coefficient_series]
        
        Returns
        -------
        out : 预测系数序列: [Batch, channel, length_of_pred_coeff_series]
        '''
        
        # 对输入系数进行RevIN归一化
        x = x.transpose(1, 2)
        x = self.revin(x, 'n')
        x = x.transpose(1, 2)
        
        # 进行patch分割
        x_patch = self.do_patching(x) 
        x_patch  = self.patch_norm(x_patch)
        # patch嵌入
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch)) 
        
        # 通过两个Mixer层处理
        out =  self.mixer1(x_emb) 
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out) 
        
        # 通过输出头生成最终预测
        out = self.head(out) 
        out = out.transpose(1, 2)
        # 对输出进行RevIN反归一化
        out = self.revin(out, 'd')
        out = out.transpose(1, 2)
        return out
    
    def do_patching(self, x):
        # 进行patch分割的实现
        x_end = x[:, :, -1:]  # 获取最后一个元素
        x_padding = x_end.repeat(1, 1, self.patch_stride)  # 重复填充
        x_new = torch.cat((x, x_padding), dim = -1)  # 拼接
        # 使用unfold进行滑动窗口分割
        x_patch = x_new.unfold(dimension = -1, size = self.patch_len, step = self.patch_stride) 
        return x_patch 
        
        
class Mixer(nn.Module):
    def __init__(self, 
                 input_seq = [],           # 输入序列长度
                 out_seq = [],             # 输出序列长度
                 batch_size = [],          # 批次大小
                 channel = [],             # 通道数
                 d_model = [],             # 模型维度
                 dropout = [],             # dropout比率
                 tfactor = [],             # 时间因子（patch mixer的扩展因子）
                 dfactor = []):            # 维度因子（embedding mixer的扩展因子）
        super(Mixer, self).__init__()
        # 初始化各种参数
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor # patch mixer的扩展因子
        self.dfactor = dfactor # embedding mixer的扩展因子
        
        # Token混合器
        self.tMixer = TokenMixer(input_seq = self.input_seq, batch_size = self.batch_size, channel = self.channel, pred_seq = self.pred_seq, dropout = self.dropout, factor = self.tfactor, d_model = self.d_model)
        # dropout层
        self.dropoutLayer = nn.Dropout(self.dropout)
        # 归一化层
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.channel)
        
        # Embedding混合器
        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_model, self.d_model * self.dfactor),
                                            nn.GELU(), 
                                            nn.Dropout(self.dropout),
                                            nn.Linear(self.d_model * self.dfactor, self.d_model))
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : 输入: [Batch, Channel, Patch_number, d_model]

        Returns
        -------
        x: 输出: [Batch, Channel, Patch_number, d_model]

        '''
        # 第一次归一化
        x = self.norm1(x)
        # 转置以适应Token混合器
        x = x.permute(0, 3, 1, 2)
        # 通过Token混合器
        x = self.dropoutLayer(self.tMixer(x))
        # 转置回原格式
        x = x.permute(0, 2, 3, 1) 
        # 第二次归一化
        x = self.norm2(x) 
        # 残差连接和Embedding混合器
        x = x + self.dropoutLayer(self.embeddingMixer(x)) 
        return x 
    
    
class TokenMixer(nn.Module):
    def __init__(self, input_seq = [], batch_size = [], channel = [], pred_seq = [], dropout = [], factor = [], d_model = []):
        super(TokenMixer, self).__init__()
        # 初始化参数
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model
        
        # dropout层
        self.dropoutLayer = nn.Dropout(self.dropout)
        # 线性层序列
        self.layers = nn.Sequential(nn.Linear(self.input_seq, self.pred_seq * self.factor),
                                   nn.GELU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.pred_seq * self.factor, self.pred_seq)
                                   )

        
    def forward(self, x):
        # 转置并进行线性变换
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x