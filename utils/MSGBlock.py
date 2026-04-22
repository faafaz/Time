from math import sqrt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.masking import TriangularCausalMask

class Predict(nn.Module):
    """
    预测模块，用于将序列转换为预测结果
    """
    def __init__(self, individual, c_out, seq_len, pred_len, dropout):
        super(Predict, self).__init__()
        self.individual = individual  # 是否对每个输出维度使用独立的线性层
        self.c_out = c_out  # 输出特征维度

        if self.individual:
            # 为每个输出维度创建独立的线性变换和dropout层
            self.seq2pred = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for i in range(self.c_out):
                self.seq2pred.append(nn.Linear(seq_len, pred_len))  # 序列到预测的线性变换
                self.dropout.append(nn.Dropout(dropout))  # Dropout层
        else:
            # 使用共享的线性变换层
            self.seq2pred = nn.Linear(seq_len, pred_len)
            self.dropout = nn.Dropout(dropout)

    # 输入形状: (B, c_out, seq)
    def forward(self, x):
        if self.individual:
            out = []
            for i in range(self.c_out):
                per_out = self.seq2pred[i](x[:, i, :])  # 对每个维度进行独立变换
                per_out = self.dropout[i](per_out)  # 应用dropout
                out.append(per_out)
            out = torch.stack(out, dim=1)  # 合并结果
        else:
            out = self.seq2pred(x)  # 共享变换
            out = self.dropout(out)  # 应用dropout

        return out


class Attention_Block(nn.Module):
    """
    注意力块，结合自注意力机制和前馈网络
    """
    def __init__(self, d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model  # 前馈网络隐藏层维度，默认为d_model的4倍
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)  # 自注意力机制
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)  # 前馈网络第一层
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)  # 前馈网络第二层
        self.norm1 = nn.LayerNorm(d_model)  # 第一层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.activation = F.relu if activation == "relu" else F.gelu  # 激活函数

    def forward(self, x, attn_mask=None):
        # 自注意力计算
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)  # 残差连接

        y = x = self.norm1(x)  # 第一层归一化
        # 前馈网络处理
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 激活和变换
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # 第二层变换

        return self.norm2(x + y)  # 第二层归一化和残差连接


class self_attention(nn.Module):
    """
    自注意力机制实现
    """
    def __init__(self, attention, d_model, n_heads):
        super(self_attention, self).__init__()
        d_keys = d_model // n_heads  # 每个头的key维度
        d_values = d_model // n_heads  # 每个头的value维度

        self.inner_attention = attention(attention_dropout=0.1)  # 内部注意力机制
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # Query投影
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)  # Key投影
        self.value_projection = nn.Linear(d_model, d_values * n_heads)  # Value投影
        self.out_projection = nn.Linear(d_values * n_heads, d_model)  # 输出投影
        self.n_heads = n_heads  # 注意力头数

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape  # 批量大小，查询序列长度
        _, S, _ = keys.shape  # 键序列长度
        H = self.n_heads  # 头数
        
        # 投影到多头形式
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 内部注意力计算
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)  # 重塑输出
        out = self.out_projection(out)  # 输出投影
        return out, attn


class FullAttention(nn.Module):
    """
    完整注意力机制实现
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale  # 缩放因子
        self.mask_flag = mask_flag  # 是否使用掩码
        self.output_attention = output_attention  # 是否输出注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # 注意力dropout

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # 批量大小，序列长度，头数，维度
        _, S, _, D = values.shape  # 值序列长度和维度
        scale = self.scale or 1. / sqrt(E)  # 计算缩放因子
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # 计算注意力分数
        
        if self.mask_flag:
            # 应用掩码（如因果掩码）
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 注意力权重
        V = torch.einsum("bhls,bshd->blhd", A, values)  # 加权求和
        
        if self.output_attention:
            return (V.contiguous(), A)  # 返回结果和注意力权重
        else:
            return (V.contiguous(), None)  # 仅返回结果


class GraphBlock(nn.Module):
    """
    图神经网络块，用于建模节点间的关系
    """
    def __init__(self, c_out, d_model, conv_channel, skip_channel,
                 gcn_depth, dropout, propalpha, seq_len, node_dim):
        super(GraphBlock, self).__init__()

        # 可学习的节点嵌入向量，用于构建自适应邻接矩阵
        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        
        # 起始卷积层
        self.start_conv = nn.Conv2d(1, conv_channel, (d_model - c_out + 1, 1))
        # 图卷积层
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()  # GELU激活函数
        # 结束卷积层
        self.end_conv = nn.Conv2d(skip_channel, seq_len, (1, seq_len))
        # 线性投影层
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)  # 层归一化

    # 输入形状: (B, T, d_model)
    # 使用MLP拟合复杂的映射函数f(x)
    def forward(self, x, adj_bias=None):
        # 基础邻接 logits（非负）
        adp_logits = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        # 时间门控偏置：A_t = softmax(base + bias)
        if adj_bias is not None:
            adp_logits = adp_logits + adj_bias.to(adp_logits.device)
        adp = F.softmax(adp_logits, dim=1)

        # 卷积处理
        out = x.unsqueeze(1).transpose(2, 3)  # 增加维度并转置
        out = self.start_conv(out)  # 起始卷积
        out = self.gelu(self.gconv1(out, adp))  # 图卷积和激活
        out = self.end_conv(out).squeeze()  # 结束卷积
        out = self.linear(out)  # 线性变换

        return self.norm(x + out)  # 残差连接和归一化


class nconv(nn.Module):
    """
    图卷积操作实现
    """
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # 执行图卷积: x是输入特征，A是邻接矩阵
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    """
    线性变换层，使用1x1卷积实现
    """
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    """
    混合图卷积传播层
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()  # 图卷积操作
        self.mlp = linear((gdep+1)*c_in, c_out)  # 线性变换
        self.gdep = gdep  # GCN深度（迭代次数）
        self.dropout = dropout  # dropout率
        self.alpha = alpha  # 混合系数

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # 添加自连接
        d = adj.sum(1)  # 计算度矩阵
        h = x
        out = [h]  # 存储各层输出
        a = adj / d.view(-1, 1)  # 归一化邻接矩阵
        
        # 多层图卷积传播
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)  # 混合传播
            out.append(h)
        
        ho = torch.cat(out, dim=1)  # 连接各层输出
        ho = self.mlp(ho)  # 线性变换
        return ho


class simpleVIT(nn.Module):
    """
    简化的视觉Transformer实现
    """
    def __init__(self, in_channels, emb_size, patch_size=2, depth=1, num_heads=4, dropout=0.1, init_weight=True):
        super(simpleVIT, self).__init__()
        self.emb_size = emb_size  # 嵌入维度
        self.depth = depth  # Transformer层数
        
        # 将输入转换为补丁嵌入
        self.to_patch = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, 2 * patch_size + 1, padding=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        # Transformer编码器层
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(emb_size),  # 层归一化
                MultiHeadAttention(emb_size, num_heads, dropout),  # 多头注意力
                FeedForward(emb_size, emb_size)  # 前馈网络
            ]))

        if init_weight:
            self._initialize_weights()  # 初始化权重

    def _initialize_weights(self):
        # 初始化卷积层权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, _, P = x.shape
        x = self.to_patch(x)  # 转换为补丁嵌入
        
        # 通过Transformer层
        for norm, attn, ff in self.layers:
            x = attn(norm(x)) + x  # 注意力和残差连接
            x = ff(x) + x  # 前馈和残差连接

        x = x.transpose(1, 2).reshape(B, self.emb_size, -1, P)  # 重塑输出
        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.keys = nn.Linear(emb_size, emb_size)  # Key变换
        self.queries = nn.Linear(emb_size, emb_size)  # Query变换
        self.values = nn.Linear(emb_size, emb_size)  # Value变换
        self.att_drop = nn.Dropout(dropout)  # 注意力dropout
        self.projection = nn.Linear(emb_size, emb_size)  # 输出投影

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 重塑为多头形式
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        
        # 计算注意力分数
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)  # 缩放因子
        att = F.softmax(energy, dim=-1) / scaling  # 注意力权重
        att = self.att_drop(att)  # Dropout
        
        # 加权求和
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")  # 重塑
        out = self.projection(out)  # 输出投影
        return out


class FeedForward(nn.Module):
    """
    前馈网络
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, hidden_dim),  # 第一层线性变换
            nn.GELU(),  # GELU激活
            nn.Linear(hidden_dim, dim),  # 第二层线性变换
        )
        
    def forward(self, x):
        return self.net(x)  # 前向传播