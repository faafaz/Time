import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
import numpy as np


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


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        # (1, n_vars, 1, d_model) -> (1 * batch_size, n_vars, 1, d_model)
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        # 将时间序列数据分割成patches
        # (batch_size, n_vars, seq_len) -> (batch_size, n_vars, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        # 合并batch和变量维度，便于批处理embedding
        # (batch_size, n_vars, n_patches, patch_len) -> (batch_size * n_vars, n_patches, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # 添加值嵌入和位置嵌入
        # (batch_size * n_vars, n_patches, patch_len) -> (batch_size * n_vars, n_patches, d_model)
        x = self.value_embedding(x) + self.position_embedding(x)

        # 恢复batch和变量维度的分离
        # (batch_size * n_vars, n_patches, d_model) -> (batch_size, n_vars, n_patches, d_model)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))

        # 在patch维度上拼接全局特征
        # (batch_size, n_vars, n_patches, d_model) -> (batch_size, n_vars, n_patches + glb_patches, d_model)
        x = torch.cat([x, glb], dim=2)

        # 再次合并batch和变量维度，准备输入Transformer
        # (batch_size, n_vars, n_patches + glb_patches, d_model) -> (batch_size * n_vars, n_patches + glb_patches, d_model)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        # res = self.self_attention(
        #     x, x, x,
        #     attn_mask=x_mask,
        #     tau=tau, delta=None
        # )[0]
        res = self.self_attention(
            x, x, x,
            attn_mask=x_mask,
        )[0]
        x = x + self.dropout(res)
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        # x_glb_attn = self.dropout(self.cross_attention(
        #     x_glb, cross, cross,
        #     attn_mask=cross_mask,
        #     tau=tau, delta=delta
        # )[0])
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        # 对不重复的Patch进行数量获取
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        # 内生变量
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        # 外生变量
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1.RevIn
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # 2.内生变量嵌入。数据集中需要除时间列外，第一个变量是内生变量(预测目标变量)
        # x_enc[:, :, 0] -> (batch_size, seq_len)
        # .unsqueeze(-1) -> (batch_size, seq_len, 1)
        # .permute(0, 2, 1) -> (batch_size, 1, seq_len)
        # en_embed -> (batch_size * 1, n_patches + 1, d_model) 其中n_vars为1
        en_embed, n_vars = self.en_embedding(x_enc[:, :, 0].unsqueeze(-1).permute(0, 2, 1))

        # 3.外生变量(协变量)嵌入。
        # ex_embed (batch_size, embed_dim, seq_len)
        ex_embed = self.ex_embedding(x_enc[:, :, 1:], x_mark_enc)

        # 4.编码器处理
        # en_embed: (batch_size * 1, n_patches + 1, d_model)
        # ex_embed: (batch_size, embed_dim, seq_len)
        # enc_out -> (batch_size * 1, n_patches + 1, d_model)
        enc_out = self.encoder(en_embed, ex_embed)

        # 恢复batch和变量维度的分离
        # (batch_size * 1, n_patches + 1, d_model) -> (batch_size, 1, n_patches + 1, d_model)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        # 调整维度顺序，准备送入预测头
        # (batch_size, 1, n_patches + 1, d_model) -> (batch_size, 1, d_model, n_patches + 1)
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 5.预测头处理：展平并线性映射到目标长度
        # (batch_size, 1, d_model, n_patches + 1) -> (batch_size, 1, pred_len)
        dec_out = self.head(enc_out)
        # (batch_size, 1, pred_len) -> (batch_size, pred_len, 1)
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc = x_enc[:, :, 1:]
        if self.task_name == 'ultra_short_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'long_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            return None
