import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.Transformer_EncDec1 import Encoder, EncoderLayer
from layers.SelfAttention_Family1 import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_wo_pos
from layers.LoadLLM import load_llm


class moving_avg(nn.Module):
    """
        实现移动平均操作，用于提取时间序列的趋势成分。
        为什么要填充？
            1. 保持序列长度不变
                详细解释见DLinear
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # 平均池化（Average Pooling）是一种下采样操作，它通过计算滑动窗口内所有值的平均值来减少数据量，同时保留重要特征。
        # padding=0/1/2 在序列两端各添加0/1/2个零
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 计算需要的填充长度 并在前端填充和后端填充
        padding_length = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding_length, 1)
        end = x[:, -1:, :].repeat(1, padding_length, 1)
        x = torch.cat([front, x, end], dim=1)
        # (batch_size,seq_len,n_vars) -> (batch_size,n_vars,seq_len)
        x = x.permute(0, 2, 1)
        # 假如(64,2,96) 2为辐射量，功率。2和96合起来理解就是，要对辐射量、功率两个特征都进行池化。
        x = self.avg(x)
        # (batch_size,n_vars,seq_len) -> (batch_size,seq_len,n_vars)
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
        序列分解模块，用于将时间序列分解为趋势和季节性成分。
        为什么季节性成分是原序列减去趋势的残差?
            详细解释见DLinear
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # 趋势Trend(t)
        res = x - moving_mean  # 季节性成分Seasonal(t)
        return res, moving_mean


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
        从高层次到低层次逐步混合或整合的趋势模式
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """
        多尺度
    """

    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()

        # 层归一化
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        self.channel_independence = configs.channel_independence

        # 将一个时间序列分解为几个更简单的成分
        self.decompsition = series_decomp(configs.moving_avg)

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class ContinuousScalingEmbedding(nn.Module):
    """
       对输入数据进行连续缩放嵌入
    """

    def __init__(self):
        super(ContinuousScalingEmbedding, self).__init__()
        self.nub_features = 7
        self.embedding_dim = 96
        self.embedding_weights = nn.Parameter(torch.randn(self.nub_features, self.embedding_dim))
        self.transform = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, input_data):
        # print(input_data.shape)
        _, sequence_length, _ = input_data.shape
        transform = self.transform(input_data.float())
        # print('transform', transform.shape)
        # print('self.embedding_weights', self.embedding_weights.shape)

        embedded_data = transform * self.embedding_weights[:sequence_length].unsqueeze(0)
        return embedded_data


class TModel(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(TModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len  # + configs.label_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.c_out

        self.Linear1 = nn.Linear(self.pred_len, self.pred_len)
        self.Linear = nn.Linear(self.pred_len, self.pred_len)

        # self.transform = TabularBertPredictionHeadTransform(config)
        # print('decoder_dropout', decoder_dropout)
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # print(x.shape)
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        x = x + seq_last

        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        x = x + seq_last
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])
        self.channel_independence = configs.channel_independence
        self.preprocess = series_decomp(configs.moving_avg)
        self.layer = configs.e_layers
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_future_temporal_feature = configs.use_future_temporal_feature
        # 位置编码
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed_type, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed_type, configs.freq,
                                                      configs.dropout)
        # RevIN归一化
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(configs.enc_in, affine=True, non_norm=True)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        # 预测层
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                )
                for i in range(configs.down_sampling_layers + 1)
            ])
            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

        # self.continuousScalingEmbedding = ContinuousScalingEmbedding()
        self.top_k = 5
        # self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        self.description = ("There is a positive correlation between power and total radiation intensity, "
                            "The power output usually increases with the increase of radiation intensity, "
                            "but there is a brief lag effect.")
        # 解码器
        self.decoder = TModel(configs)

        # 大模型和分词器
        model_, tokenizer_ = load_llm(configs.llm_name, configs.llm_layers)
        self.llm_model = model_
        self.tokenizer = tokenizer_
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        self.llm_hidden_size = self.llm_model.config.hidden_size
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, 0.001,
                                      output_attention=None), 768, 8),
                    768,
                    100,
                    dropout=0.001,
                    activation='gelu'
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(768)
        )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=1, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print('x_enc',x_enc.shape)
        # x_enc = self.ContinuousScalingEmbedding(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)
        # print('tab_embedding_res',tab_embedding_res.shape)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        # prompt_embeddings = self.transformer.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        # print('prompt_embeddings', prompt_embeddings.shape)

        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        # print(type(x_enc), 'x_enc')

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous()
                    x = x.reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)
        # Past Decomposable Mixing as encoder for past

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        # Future Multipredictor Mixing as decoder for future
        hidden_list = self.future_multi_mixing(B, enc_out_list, x_list)

        hidden_info = torch.stack(hidden_list, dim=-1).sum(-1)

        inputs_embeds = hidden_info.permute(0, 2, 1)

        old_pad = inputs_embeds.shape[2]
        inputs_embeds = F.pad(inputs_embeds, (0, self.llm_hidden_size - old_pad))
        prompt_embeddings = prompt_embeddings[0:inputs_embeds.shape[0], :, :]
        inputs_embeds = torch.cat([prompt_embeddings, inputs_embeds], dim=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds)

        hidden_states = outputs.last_hidden_state

        dec_out = hidden_states[:, prompt_embeddings.shape[1]:, :old_pad].permute(0, 2, 1)
        dec_out = self.decoder(dec_out) + hidden_info

        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc = x_enc[:, :, 0:1]  # 取功率
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
