import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import math


class PositionalEmbedding(nn.Module):
    """
        为什么加上位置编码，模型就知道位置了呢?
        用颜色来理解位置编码
            想象每个字都是一种基础颜色：
                "我" = 红色
                "爱" = 蓝色
                "你" = 绿色
            现在我们要给每个位置也分配一种颜色：
                位置1 = 黄色
                位置2 = 紫色
                位置3 = 橙色
            关键操作：颜色混合（相当于向量相加）
            当我们把字和位置"混合"：
                句子 "我爱你"：
                    位置1的"我" = 红色 + 黄色 = 橙红色
                    位置2的"爱" = 蓝色 + 紫色 = 紫蓝色
                    位置3的"你" = 绿色 + 橙色 = 黄绿色
                句子 "你爱我"：
                    位置1的"你" = 绿色 + 黄色 = 黄绿色（注意：和上面不同！）
                    位置2的"爱" = 蓝色 + 紫色 = 紫蓝色（相同）
                    位置3的"我" = 红色 + 橙色 = 红橙色（注意：和上面不同！）
            模型学会识别混合后的颜色模式
                训练后，模型学会了：
                    看到"橙红色" → 这是"我"在句首，通常是主语
                    看到"红橙色" → 这是"我"在句尾，通常是宾语
                    看到"黄绿色"在句首 → 这是"你"在句首，通常是主语
                    看到"黄绿色"在句尾 → 这是"你"在句尾，通常是宾语
            为什么混合（相加）有效？
                关键在于：混合后的颜色是独一无二的！
        同一个字在不同位置 → 产生不同的混合色
            模型能从混合色中同时"看出"原始颜色（语义）和位置颜色（位置）

        数字版本的简单例子
            "我" = 10
            "爱" = 20
            "你" = 30
            位置1 = 1
            位置2 = 100
            位置3 = 10000
        混合后：
            "我爱你": [11, 120, 10030]
            "你爱我": [31, 120, 10010]
        模型学会：
            看到11 → 知道是"我"(10)在位置1(+1)
            看到10010 → 知道是"我"(10)在位置3(+10000)
            看到31 → 知道是"你"(30)在位置1(+1)

        说白了，本质上就是要让训练的数据的数值分布不一样.位置编码的核心就是：让相同的词在不同位置产生不同的数值分布。
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        """
            torch.arange(0, d_model, 2)生成偶数索引
            假设 d_model = 512：
            div_term[0] = 1 / 10000^(0/512) = 1.0
            div_term[1] = 1 / 10000^(2/512) ≈ 0.955
            div_term[2] = 1 / 10000^(4/512) ≈ 0.912
            div_term 本质上就是一个从1到很小值的递减序列，用于生成不同频率的正弦和余弦波
        """
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # 偶数维
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # 注册为buffer，不参与训练但会保存在模型中
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        # 一维数据填充的层，通过复制输入张量的边界值来扩展数据长度，常用于卷积神经网络中保持输出尺寸或保护边界信息
        # (0, padding)代表 左侧不填充（填充量为0）右侧填充padding个元素
        # 若输入为[A, B, C]，padding=2时输出为[A, B, C, C, C]（左侧不填充，右侧复制两次C）
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # 将序列映射到高维
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # 位置编码
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        # 填充边缘
        x = self.padding_patch_layer(x)
        """
            patch数量 = (seq_len - patch_len) // stride + 1 的核心逻辑是：
            确定可用范围: seq_len - patch_len 是最后一个补丁可以开始的位置
            计算步数: 在这个范围内能走多少个 stride 步长
            包含起点: 加1是因为第一个patch从位置0开始
            [例]
                第1步：原始数据
                数据: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                长度: 10
                第2步：填充后
                填充后: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]
                长度: 11
                第3步：滑动窗口取patch
                用长度为3的窗口，每次移动2步：
                第1个patch: 位置0-2  → [1, 2, 3]
                第2个patch: 位置2-4  → [3, 4, 5]  
                第3个patch: 位置4-6  → [5, 6, 7]
                第4个patch: 位置6-8  → [7, 8, 9]
                第5个patch: 位置8-10 → [9, 10, 10]
                计算验证
                
                总长度：11
                最后一个patch的起始位置最大：11 - 3 = 8
                从位置0到位置8，每次移动2步：0, 2, 4, 6, 8
                总共5个起始位置 = 5个patch
                
                用公式：(11 - 3) ÷ 2 + 1 = 8 ÷ 2 + 1 = 4 + 1 = 5
                
                为什么是从位置0到位置8，不是从位置1到位置8？
                第一个patch必须从位置0开始，这是unfold函数的默认行为
            
            在PatchTST中，patch_len=16，stride=8，padding=stride, 也就是说只有最后一个Patch中含有无用信息
            对于unfold函数，如果(seq_len - patch_len) // stride不能整除，末尾不足一个窗口的部分会被直接丢弃
        """
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Transpose(nn.Module):
    """
        Transpose 类的价值在于：
            模块化：将转置操作封装为标准的nn.Module
            Sequential友好：可以轻松插入到Sequential容器中
        为啥要判断if self.contiguous呢，什么情况下contiguous为True?
            这个判断是为了性能优化和避免不必要的内存拷贝
            [例]    x = torch.randn(1000, 1000, 1000)  # 很大的张量
                    y = x.transpose(0, 2)
                    # 情况1：不需要连续性
                    result1 = y.transpose(0, 2)  # 几乎无开销
                    # 情况2：强制连续性
                    result2 = y.transpose(0, 2).contiguous()  # 可能需要拷贝整个张量！
            如果后续有reshape操作，则必须传入contiguous=True
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        """
            BatchNorm1d要求输入格式为 (batch_size, features, length) features就是d_model
            nn.BatchNorm1d(configs.d_model) 中的参数不是指定在哪个维度归一化，而是指定特征的数量。
            BatchNorm(归一化维度：batch_size)：
                序列模型中不同位置的特征分布可能差异很大
                变长序列处理困难
                小batch时性能不稳定
            LayerNorm(归一化维度：d_model)的优势：
                更适合序列数据的特性
                在Transformer中广泛使用
                对batch大小不敏感
        """
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2))
        )

        # Prediction Head
        patch_count = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf = configs.d_model * patch_count
        if self.task_name == 'ultra_short_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'long_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = x_enc[:, :, 1:]  # 取功率
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs, nvars, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder 映射成时间序列
        dec_out = self.head(enc_out)  # z: [bs, nvars, target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'ultra_short_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None
