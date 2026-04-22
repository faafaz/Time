import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from utils.DetailEnhancementModule import EnhancedDLinear1DHead


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # (batch_size, seq_len, n_vars) -> (batch_size, n_vars, seq_len)
        x = x.permute(0, 2, 1)
        if x_mark is None:
            # 直接进行值嵌入 (batch_size, n_vars, seq_len) -> (batch_size, embed_dim, seq_len)
            x = self.value_embedding(x)
        else:
            # (batch_size, seq_len, mark_features) -> (batch_size, mark_features, seq_len)
            x_mark = x_mark.permute(0, 2, 1)
            # 拼接: (batch_size, seq_len + mark_features, seq_len) -> (batch_size, embed_dim, seq_len)
            x = self.value_embedding(torch.cat([x, x_mark], 1))
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        # Encoder

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
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'ultra_short_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'long_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.configs = configs

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1.RevIn
        # (batch_size, seq_len, 1) -> (batch_size, 1, 1) 计算序列均值用于归一化
        means = x_enc.mean(1, keepdim=True).detach()
        # (batch_size, seq_len, 1) 减去 (batch_size, 1, 1) 减去均值进行中心化
        x_enc = x_enc - means
        # (batch_size, seq_len, 1) -> (batch_size, 1, 1) 计算序列标准差
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # (batch_size, seq_len, 1) 除以 (batch_size, 1, 1) 除以标准差进行归一化
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape
        # 检查数据中是否包含nan值

        # 2.Embedding
        # (batch_size, seq_len, 1) -> (batch_size, embed_dim , seq_len)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 3.编码器处理
        # (batch_size, embed_dim, seq_len) -> (batch_size, embed_dim, d_model)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 4.投影到预测长度并调整维度
        # (batch_size, embed_dim, d_model) -> (batch_size, embed_dim, pred_len)
        dec_out = self.projection(enc_out)
        # (batch_size, embed_dim, pred_len) -> (batch_size, pred_len, embed_dim)
        dec_out = dec_out.permute(0, 2, 1)
        # (batch_size, pred_len, embed_dim) -> (batch_size, pred_len, N)
        # (batch_size, pred_len, 1+5) -> (batch_size, pred_len, 1) 1为n_vars,5为时间embeddings
        dec_out = dec_out[:, :, :N]

        # 6.反RevIn
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'ultra_short_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, :, self.in_num_features - 1:self.in_num_features]  # [B, L, D]

        return None
    def dfs_regularization(self):
        """L1正则项（对DFS掩码概率）"""
        return self.dfs.l1_regularization() if hasattr(self, 'dfs') else 0.0

    def anneal_dfs_temperature(self, rate: float = 0.95, min_temp: float = 0.1):
        """对DFS温度进行退火"""
        if hasattr(self, 'dfs'):
            self.dfs.anneal(rate=rate, min_temperature=min_temp)

