# 导入必要的模块和类
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # 一维小波变换的前向和反向变换
from utils.RevIN import RevIN  # RevIN归一化模块

class Decomposition(nn.Module):
    def __init__(self,
                 input_length = [],        # 输入序列长度
                 pred_length = [],         # 预测序列长度
                 wavelet_name = [],        # 小波名称
                 level = [],               # 小波分解层数
                 batch_size = [],          # 批次大小
                 channel = [],             # 通道数
                 d_model = [],             # 模型维度
                 tfactor = [],             # 时间因子
                 dfactor = [],             # 维度因子
                 device = [],              # 计算设备
                 no_decomposition = [],    # 是否不进行分解
                 use_amp = []):            # 是否使用自动混合精度
        super(Decomposition, self).__init__()
        # 初始化各种参数
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.device = device
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.eps = 1e-5  # 防止除零的小常数
        
        # 初始化小波正向和反向变换器
        # self.dwt = DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp)
        self.dwt = DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp).cuda() 
        # self.idwt = DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp)
        self.idwt = DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp).cuda() 
        # 获取分解后的输入和预测序列维度
        self.input_w_dim = self._dummy_forward(self.input_length) if not self.no_decomposition else [self.input_length] # 分解后输入序列的长度
        self.pred_w_dim = self._dummy_forward(self.pred_length) if not self.no_decomposition else [self.pred_length] # 分解后预测序列所需的长度
        
        self.tfactor = tfactor
        self.dfactor = dfactor
        #################################
        # 仿射变换和RevIN归一化标志
        self.affine = False
        self.rev_ins_normalization = False
        #################################
        
        # 如果启用仿射变换，则初始化参数
        if self.affine:
            self._init_params()
        # 如果启用RevIN归一化，则初始化RevIN模块
        if self.rev_ins_normalization:
            self.revin = nn.ModuleList([RevIN(self.channel) for i in range(self.level + 1)])
            
    def transform(self, x):
        # 输入: x 形状: batch, channel, seq
        if not self.no_decomposition:
            # 进行小波分解
            yl, yh = self._wavelet_decompose(x)
        else:
            # 不分解: 返回相同的值在yl中
            yl, yh = x, []
        return yl, yh
    
    def inv_transform(self, yl, yh):
        # 小波反变换重构
        if not self.no_decomposition:
            x = self._wavelet_reverse_decompose(yl, yh)
        else:
            # 不分解: 返回相同的值在x中
            x = yl
        return x
           
    def _dummy_forward(self, input_length):
        # 使用虚拟输入获取分解后的维度信息
        dummy_x = torch.ones((self.batch_size, self.channel, input_length)).to(self.device)
        yl, yh = self.dwt(dummy_x)
        l = []
        # 添加近似系数的长度
        l.append(yl.shape[-1])
        # 添加各层细节系数的长度
        for i in range(len(yh)):
            l.append(yh[i].shape[-1])
        return l
    
    def _init_params(self):
        # 初始化仿射变换参数
        self.affine_weight = nn.Parameter(torch.ones((self.level + 1, self.channel)))
        self.affine_bias = nn.Parameter(torch.zeros((self.level + 1, self.channel)))
    
    def _wavelet_decompose(self, x):
        # 输入: x 形状: batch, channel, seq
        # 进行小波分解
        yl, yh = self.dwt(x)
        
        # 如果启用仿射变换，则对系数进行变换
        if self.affine:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = yl * self.affine_weight[0]
            yl = yl + self.affine_bias[0]
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ * self.affine_weight[i + 1]
                yh_ = yh_ + self.affine_bias[i + 1]
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq
                
        # 如果启用RevIN归一化，则对系数进行归一化
        if self.rev_ins_normalization:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = self.revin[0](yl, 'norm')
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = self.revin[i + 1](yh_, 'norm')
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq
        return yl, yh
    
    def _wavelet_reverse_decompose(self, yl, yh):
        # 如果启用仿射变换，则对系数进行反变换
        if self.affine:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = yl - self.affine_bias[0]
            yl = yl / (self.affine_weight[0] + self.eps)
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ - self.affine_bias[i + 1]
                yh_ = yh_ / (self.affine_weight[i + 1] + self.eps)
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq
                
        # 如果启用RevIN归一化，则对系数进行反归一化
        if self.rev_ins_normalization:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = self.revin[0](yl, 'denorm')
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = self.revin[i + 1](yh_, 'denorm')
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq
                
        # 进行小波反变换重构
        x = self.idwt((yl, yh))
        return x # 形状: batch, channel, seq