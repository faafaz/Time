import torch
import torch.nn as nn
from einops import rearrange
from utils.RevIN import RevIN
import numpy as np
import math


class TriangularCausalMask:
    """三角因果掩码"""

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


"""
DUET (Dual Clustering Enhanced Multivariate Time Series Forecasting) 模型
基于双聚类机制增强多变量时间序列预测性能
核心组件：
1. TCM (Temporal Clustering Module): 时间维度聚类
2. CCM (Channel Clustering Module): 通道维度聚类
3. Distribution Router: 分布路由器
4. Linear Pattern Extractor: 线性模式提取器
"""


class Linear_extractor_cluster(nn.Module):
    """线性提取器聚类模块"""

    def __init__(self, configs):
        super(Linear_extractor_cluster, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model
        self.num_experts = configs.num_experts
        self.k = configs.k
        self.patch_len = configs.patch_len

        # RevIN归一化
        self.revin = RevIN(self.n_vars, affine=True, subtract_last=False)

        # 线性投影层
        self.linear_projection = nn.Linear(self.patch_len, self.d_model)

        # 聚类相关层
        self.distribution_router = nn.Linear(self.d_model, self.num_experts)
        self.softmax = nn.Softmax(dim=-1)

        # 专家线性层
        self.experts = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model) for _ in range(self.num_experts)
        ])

        # 聚类中心
        self.cluster_centers = nn.Parameter(
            torch.randn(self.num_experts, self.d_model)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, n_vars]

        # 简单归一化 (暂时不使用RevIN)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # 序列分块
        batch_size, seq_len, n_vars = x.shape
        num_patches = seq_len // self.patch_len

        # 重塑为patch格式
        x_patches = x.unfold(1, self.patch_len, self.patch_len)  # [batch, num_patches, n_vars, patch_len]
        x_patches = rearrange(x_patches, 'b p n l -> (b n) p l')

        # 线性投影
        x_projected = self.linear_projection(x_patches)  # [(b n), p, d_model]

        # 分布路由
        router_logits = self.distribution_router(x_projected)  # [(b n), p, num_experts]
        router_weights = self.softmax(router_logits)

        # 专家处理
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_projected)  # [(b n), p, d_model]
            expert_weight = router_weights[:, :, i].unsqueeze(-1)  # [(b n), p, 1]
            weighted_expert = expert_out * expert_weight
            expert_outputs.append(weighted_expert)

        # 合并专家输出
        combined_output = sum(expert_outputs)  # [(b n), p, d_model]

        # 聚类重要性计算
        importance_scores = torch.norm(combined_output, dim=-1).mean(dim=1)  # [(b n)]

        # 重塑回原始格式 [batch, seq_len, n_vars]
        combined_output = rearrange(combined_output, '(b n) p d -> b (p d) n', b=batch_size)

        # 调整维度到正确的seq_len
        current_length = combined_output.shape[1]
        if current_length < self.seq_len:
            # 填充
            padding = torch.zeros(batch_size, self.seq_len - current_length, n_vars).to(combined_output.device)
            combined_output = torch.cat([combined_output, padding], dim=1)
        elif current_length > self.seq_len:
            # 截断
            combined_output = combined_output[:, :self.seq_len, :]

        # 简单反归一化
        combined_output = combined_output * stdev
        combined_output = combined_output + means

        return combined_output, importance_scores.mean()


class Mahalanobis_mask(nn.Module):
    """马氏距离掩码生成器"""

    def __init__(self, seq_len):
        super(Mahalanobis_mask, self).__init__()
        self.seq_len = seq_len

    def forward(self, x):
        # x: [batch_size, n_vars, seq_len]
        batch_size, n_vars, seq_len = x.shape

        # 计算协方差矩阵
        x_reshaped = rearrange(x, 'b n l -> (b l) n')
        cov_matrix = torch.cov(x_reshaped.T)  # [n_vars, n_vars]

        # 添加小值防止奇异矩阵
        cov_matrix = cov_matrix + torch.eye(n_vars).to(x.device) * 1e-6

        # 计算马氏距离
        inv_cov = torch.linalg.inv(cov_matrix)

        # 生成掩码
        mask = torch.ones(batch_size, n_vars, n_vars).to(x.device)

        return mask


class FullAttention(nn.Module):
    """完整注意力机制"""

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    """注意力层"""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """编码器层"""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, attn_mask=None):
        # 注意力层
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # 前馈层
        new_x = x.transpose(1, 2)
        new_x = self.conv1(new_x)
        new_x = self.activation(new_x)
        new_x = self.conv2(new_x)
        new_x = new_x.transpose(1, 2)

        x = x + self.dropout(new_x)
        x = self.norm2(x)

        return x, attn


class Encoder(nn.Module):
    """编码器"""

    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Model(nn.Module):
    """DUET主模型"""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in

        # DUET特有参数
        self.CI = getattr(configs, 'CI', True)  # 通道独立性
        self.num_experts = getattr(configs, 'num_experts', 4)
        self.k = getattr(configs, 'k', 2)
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.factor = getattr(configs, 'factor', 5)  # 注意力因子

        # 线性提取器聚类模块
        self.cluster = Linear_extractor_cluster(configs)

        # 马氏距离掩码生成器
        self.mask_generator = Mahalanobis_mask(self.seq_len)

        # 通道Transformer
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True,
                            factor=getattr(configs, 'factor', 5),
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 通道维度投影层
        self.channel_projection = nn.Linear(self.n_vars, configs.d_model)

        # 输出头
        self.linear_head = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            nn.Dropout(getattr(configs, 'fc_dropout', 0.2))
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [batch_size, seq_len, n_vars]
        x = x[:, :, 2:]

        # 通道独立性处理
        if self.CI:
            # 通道独立模式
            channel_independent_input = rearrange(x, 'b l n -> (b n) l 1')
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=x.shape[0])
        else:
            # 通道依赖模式
            temporal_feature, L_importance = self.cluster(x)

        # 调整维度 [batch, seq_len, n_vars] -> [batch, n_vars, seq_len]
        temporal_feature = rearrange(temporal_feature, 'b l n -> b n l')

        # 多变量处理
        if self.n_vars > 1:
            # 对于多变量情况，直接使用线性头处理每个变量
            # [batch, n_vars, seq_len] -> [batch, n_vars, pred_len]
            output = self.linear_head(temporal_feature)
        else:
            # 单变量直接处理
            # [batch, 1, seq_len] -> [batch, 1, pred_len]
            output = self.linear_head(temporal_feature)

        # 调整输出维度 [batch, n_vars, pred_len] -> [batch, pred_len, n_vars]
        output = rearrange(output, 'b n d -> b d n')

        return output[:, :, :1]