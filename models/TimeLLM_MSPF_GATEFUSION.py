from math import sqrt
import torch
import torch.nn as nn
from layers.StandardNorm import Normalize
from torch import Tensor
from layers.LoadLLM import load_llm
from layers.Prompt import get_simple_prompt, get_calculate_prompt
import torch.nn.functional as F


class SimpleFusionStrategy1(nn.Module):
    """
    策略1: 直接拼接所有patches

    最简单的方案：把所有尺度的patches直接连在一起
    短期(47) + 中期(15) + 长期(7) = 69个patches
    """

    def __init__(self, embed_dim=64):
        super(SimpleFusionStrategy1, self).__init__()

        # 简单的线性嵌入
        self.short_embed = nn.Linear(4, embed_dim)
        self.medium_embed = nn.Linear(12, embed_dim)
        self.long_embed = nn.Linear(24, embed_dim)

    def forward(self, patches_dict):
        short_patches = patches_dict['short']  # [B, 47, 4]
        medium_patches = patches_dict['medium']  # [B, 15, 12]
        long_patches = patches_dict['long']  # [B, 7, 24]

        # 嵌入到相同维度
        short_embedded = self.short_embed(short_patches)  # [B, 47, embed_dim]
        medium_embedded = self.medium_embed(medium_patches)  # [B, 15, embed_dim]
        long_embedded = self.long_embed(long_patches)  # [B, 7, embed_dim]

        # 直接拼接
        fused = torch.cat([short_embedded, medium_embedded, long_embedded], dim=1)

        return fused  # [B, 69, embed_dim]


class GatedFusionStrategy(nn.Module):
    """
    门控融合：根据辐射量强度动态调整融合权重
    """

    def __init__(self, embed_dim=64):
        super().__init__()

        # 单变量嵌入
        self.irradiance_embed = SimpleFusionStrategy1(embed_dim)
        self.power_embed = SimpleFusionStrategy1(embed_dim)

        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # 交互融合
        self.interaction_layer = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, irradiance_patches, power_patches):
        irr_features = self.irradiance_embed(irradiance_patches)  # [B, P, embed_dim]
        pow_features = self.power_embed(power_patches)  # [B, P, embed_dim]

        # 门控权重：基于辐射量和功率的联合信息
        gate_input = torch.cat([irr_features, pow_features], dim=-1)
        gate_weights = self.gate_net(gate_input)  # [B, P, 1]

        # 门控融合
        gated_irr = gate_weights * irr_features
        yushu = (1 - gate_weights)
        gated_pow = yushu * pow_features

        # 交互融合
        interaction = self.interaction_layer(torch.cat([gated_irr, gated_pow], dim=-1))

        return interaction



class MultiScalePatcher:
    """多尺度Patch生成器 - 针对太阳能功率预测优化"""

    def __init__(self):
        # 三尺度参数设置
        self.scales = {
            'short': {'len': 4, 'stride': 2},  # 云层瞬时变化
            'medium': {'len': 12, 'stride': 6},  # 太阳高度角变化
            'long': {'len': 24, 'stride': 12}  # 天气系统变化
        }

    def _get_patches(self, x, patch_len, stride):
        """生成单一尺度patches"""
        batch_size, seq_len = x.shape[:2]
        num_patches = (seq_len - patch_len) // stride + 1

        patches = []
        for i in range(num_patches):
            start = i * stride
            patch = x[:, start:start + patch_len]  # [B, patch_len]
            patches.append(patch)

        return torch.stack(patches, dim=1)  # [B, num_patches, patch_len]

    def forward(self, x):
        """
        输入: [batch_size, seq_len, 1]
        输出: {'short': [B, P_s, L_s], 'medium': [B, P_m, L_m], 'long': [B, P_l, L_l]}
        """
        x = x.squeeze(-1)  # [B, seq_len, 1] -> [B, seq_len]

        patches = {}
        for scale_name, params in self.scales.items():
            patches[scale_name] = self._get_patches(
                x, params['len'], params['stride']
            )

        return patches


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        # self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        print()
        self.config = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.d_llm
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.num_tokens = configs.num_tokens
        model_, tokenizer_ = load_llm(configs.llm_name, configs.llm_layers)
        self.llm_model = model_
        self.tokenizer = tokenizer_
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.patch_nums = 69  #
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(1, self.head_nf, self.pred_len, head_dropout=configs.dropout)

        self.normalize_layers = Normalize(1, affine=False)
        # 多尺度分patch
        self.msp = MultiScalePatcher()

        # 层次化特征融合
        self.gate_fusion = GatedFusionStrategy(embed_dim=configs.d_model)

    def forward(self, x_enc, batch_x_mark=None, dec_inp=None, batch_y_mark=None):
        x_enc = self.normalize_layers(x_enc, 'norm')
        x_power = x_enc[:, :, 0:1]  # 取功率
        x_irr = x_enc[:, :, 1:2]  # 取辐射量

        # 获取提示词的embeddings
        prompt = globals().get(self.config.prompt_func)(x_enc, self.seq_len, self.pred_len)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # 多尺度分patch
        x_power_patches = self.msp.forward(x_power)
        x_irr_patches = self.msp.forward(x_irr)

        # 门控网络融合
        enc_out = self.gate_fusion(x_irr_patches, x_power_patches)

        # 所有的尺度与文本原型计算注意力
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, 1, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = dec_out[:, :, 0:1]  # 只要第一列的功率

        dec_out = dec_out[:, -self.pred_len:, :]
        return dec_out


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
