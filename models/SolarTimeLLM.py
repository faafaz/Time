from math import sqrt
import torch
import torch.nn as nn
from layers.StandardNorm import Normalize
from torch import Tensor
from layers.LoadLLM import load_llm
from layers.Prompt import get_simple_prompt, get_calculate_prompt


class TokenEmbedding(nn.Module):
    # patch_len=64 d_model为Time-LLM的隐藏层维度32
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        """
            对每个patch再进行卷积提取特征 每个patch长度为16，把16个点映射成32维的向量，即每个patch接下来都由一个32维的向量来表示了
            通常用于自然语言处理（NLP）或时间序列分析等任务中，用于处理序列数据。通过一维卷积层，可以提取序列中的局部特征，从而对序列数据进行特征提取和变换。
                nn.Conv1d是PyTorch中的一个类，用于创建一维卷积层。一维卷积层通常用于处理序列数据，例如文本或时间序列数据。
                in_channels=c_in：指定输入信号的通道数，即输入数据的特征数。
                out_channels=d_model：指定输出信号的通道数，即卷积层输出的特征数。
                kernel_size=3：指定卷积核的大小，这里是一个长度为3的卷积核。
                padding=padding：指定输入信号的填充方式。padding是一个变量，通常用来控制填充的长度。
                padding_mode='circular'：指定填充模式为循环填充。循环填充意味着在输入信号的边缘部分，填充的内容是输入信号的反向部分。
                bias=False：指定不使用偏置项。在某些情况下，偏置项可能不需要，或者为了简化模型，可以选择不使用偏置项。
        """
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            # 检查当前模块是否是 nn.Conv1d 类型
            if isinstance(m, nn.Conv1d):
                """
                    使用Kaiming初始化方法来初始化卷积层的权重。kaiming_normal_用于生成符合Kaiming分布的权重。
                        mode='fan_in' 表示使用输入通道数的方差
                        nonlinearity='leaky_relu' 表示使用Leaky ReLU作为激活函数。

                """
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1))
        x = x.transpose(1, 2)
        return x


class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding  # (0, stride)

    def forward(self, input: Tensor) -> Tensor:
        # input(1,7,512) -> replicate_padding(1,7)
        replicate_padding = input[:, :, -1]
        # (1,7) -> (1,7,1)
        replicate_padding = replicate_padding.unsqueeze(-1)
        stride = self.padding[-1]
        # (1,7,1) -> (1,7,8) repeat(1,1,8)表示第一个维度重复1次，第二个维度重复1次，第三个维度重复8次
        replicate_padding = replicate_padding.repeat(1, 1, stride)
        # (1,7,512 + 8)
        # 最好是不用padding，因为这里padding只是把最后一个数据重复了8次，我觉得是破坏了序列
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class PatchEmbedding(nn.Module):
    # Time-LLM的隐层维度32, patch长度, 步长, dropout
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        # 补充序列长度以便分patch
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        # 将16个点的patch投影成32维
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        """
            经过padding_patch_layer后，x的shape从(4,2,512)=>(4,2,520)
        """
        x = self.padding_patch_layer(x)
        """
            unfold方法切割patch，patch个数 = (seq_len - patch_len) / stride + 1  即(520 - 16) / 8 + 1 = 64
        """
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        """
           (4,2,64,16)=>(8,64,16)
        """
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        """
           核心：对每个patch再进行卷积提取特征 每个patch长度为16，把16个点映射成32维的向量，即每个patch接下来都由一个32维的向量来表示了
           执行完下一行 x (8,64,16)=>(8,64,32)
        """
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


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


class CrossScaleAttention(nn.Module):
    """
    日间周期性增强的尺度间交互模块
    整合不同时间尺度的特征，并增强对日间周期性的感知
    """

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        """
        初始化日间周期性增强的尺度间交互模块

        参数:
            d_model: 模型隐藏层维度
            n_heads: 注意力头数
            dropout: Dropout比率
        """
        super(CrossScaleAttention, self).__init__()

        # 时间编码层
        # self.time_encoding = nn.Linear(1, d_model)

        # 尺度间注意力机制
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, scales):
        """
        前向传播函数

        参数:
            short_scale: 短期尺度特征
            medium_scale: 中期尺度特征
            long_scale: 长期尺度特征
            time_of_day: 一天中的时间信息 (0-24小时，归一化到0-1)

        返回:
            整合了不同时间尺度，并增强了日间周期性感知的特征
        """
        # 尺度间注意力机制
        scales_transposed = scales.transpose(0, 1)  # 调整为注意力机制所需的形状
        attn_output, _ = self.cross_attn(scales_transposed, scales_transposed, scales_transposed)
        attn_output = attn_output.transpose(0, 1)  # 调整回原始形状

        # 残差连接和层归一化
        output1 = self.norm1(scales + self.dropout(attn_output))

        # 前馈神经网络
        ffn_output = self.ffn(output1)
        output2 = self.norm2(output1 + self.dropout(ffn_output))

        return output2


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.prompt_func = configs.prompt_func
        print(configs.load_llm_func)
        model_, tokenizer_ = globals().get(configs.load_llm_func)(configs.llm_layers)
        # model_, tokenizer_ = load_llama_7b(configs.llm_layers)
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

        # 自注意力
        self.crossScaleAttention = CrossScaleAttention(d_model=configs.d_llm, n_heads=configs.n_heads,
                                                       dropout=configs.dropout)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 100
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # self.reprogramming_layer = ReprogrammingLayer(d_patch=configs.d_model, d_llm=configs.d_llm,
        #                                               n_heads=configs.n_heads, d_ff=self.d_ff, dropout=configs.dropout)
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        # 多变量特征融合
        self.jointRepresentation = JointRepresentation(num_vars=configs.n_features, d_llm=self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.output_projection = FlattenHead(n_vars=2, nf=self.d_ff * self.patch_nums, target_window=self.pred_len,
                                             head_dropout=configs.dropout)
        self.normalize_layers = Normalize(configs.n_features, affine=False)

    def forward(self, x_enc, batch_y, batch_x_date):  # (B,SEQ_LEN, N)
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 获取提示词的embeddings
        prompt = globals().get(self.prompt_func)(B, self.seq_len, self.pred_len)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # 分patch
        x_enc = x_enc.reshape(B, N, T)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # 计算patch与文本原型的注意力
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out_list = []
        for i in range(n_vars):
            enc_out_item = self.reprogramming_layer(enc_out[:, i, :, :], source_embeddings, source_embeddings)
            enc_out_list.append(enc_out_item)

        # 多变量特征融合 和prompt_embeddings拼起来丢给大模型
        z_joint = self.jointRepresentation(enc_out_list)

        # 自注意力机制
        z_joint = self.crossScaleAttention(z_joint)

        llama_enc_out = torch.cat([prompt_embeddings, z_joint], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = self.output_projection(dec_out[:, -self.patch_nums:, :]).unsqueeze(-1)

        dec_out = self.normalize_layers(dec_out, 'denorm')
        # dec_out = dec_out[:, -self.pred_len:, :]
        dec_out = dec_out[:, :, 0:1]  # 只要第一列的功率
        return dec_out


class JointRepresentation(nn.Module):
    def __init__(self, num_vars, d_llm, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(num_vars * d_llm, d_llm)  # 融合多变量特征
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_list):
        # 在特征维度拼接
        z_combined = torch.cat(z_list, dim=2)  # (batch_size, patch_nums, num_vars*d_model)
        # 生成联合表征
        z_joint = self.fc(z_combined)  # (batch_size, patch_nums, d_model)
        return z_joint


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
