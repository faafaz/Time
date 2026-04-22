import torch
import torch.nn as nn
from utils.MSGBlock import GraphBlock
from models.iTransformer_xLSTM_VMD_Preprocessed1 import VMDDecomposer
from layers.Embed import TimeFeatureEmbedding
from models.iTransformer_xLSTM import LongConvMix




"""
DLinear_Graph: 融合图神经网络和DLinear的时间序列预测模型

核心思想：
1. DLinear部分：时间序列分解（趋势+季节性）+ 线性预测
2. GraphBlock部分：自动学习变量间依赖关系，增强多变量建模能力
3. LongConvMix部分：对目标变量进行多尺度时序增强
4. 融合策略：在分解后、预测前加入图建模与时序增强，让变量间信息交互并强化目标变量表达

架构流程：
输入 [B, seq_len, n_vars]
  ↓
RevIN 归一化
  ↓
VMD 自适应频带分解（可选）
  ↓
DLinear 分解 (趋势 + 季节性)
  ↓
GraphBlock 建模变量关系 (分别对趋势和季节性)
  ↓
LongConvMix 目标变量时序增强（可选）
  ↓
线性预测头
  ↓
反 RevIN
  ↓
输出 [B, pred_len, 1]
"""


class moving_avg(nn.Module):
    """移动平均模块，用于提取趋势成分"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 边界填充（重复边界值）
        padding_length = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding_length, 1)
        end = x[:, -1:, :].repeat(1, padding_length, 1)
        x = torch.cat([front, x, end], dim=1)

        # 平均池化
        x = x.permute(0, 2, 1)  # [B, seq_len, n_vars] -> [B, n_vars, seq_len]
        x = self.avg(x)
        x = x.permute(0, 2, 1)  # [B, n_vars, seq_len] -> [B, seq_len, n_vars]
        return x


class series_decomp(nn.Module):
    """序列分解模块：分解为趋势和季节性成分"""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # 趋势成分
        res = x - moving_mean             # 季节性成分
        return res, moving_mean


class Model(nn.Module):
    """
    DLinear_Graph 模型

    参数说明：
        configs.seq_len: 输入序列长度
        configs.pred_len: 预测序列长度
        configs.n_input_features: 输入特征数（变量数）
        configs.moving_avg: 移动平均窗口大小
        configs.individual: 是否为每个变量使用独立的线性层
        configs.graph_conv_channel: 图卷积通道数（默认16）
        configs.graph_skip_channel: 图跳跃连接通道数（默认32）
        configs.gcn_depth: 图卷积深度（默认2）
        configs.graph_node_dim: 节点嵌入维度（默认10）
        configs.graph_propalpha: 图传播系数（默认0.05）
        configs.use_graph: 是否使用图模块（默认True）
        configs.enable_target_longconv: 是否对目标变量使用LongConvMix增强（默认True）
        configs.longconv_kernels: LongConvMix卷积核大小（默认(5, 11, 23)）
        configs.longconv_dropout: LongConvMix的dropout率（默认0.1）
        configs.longconv_hidden: LongConvMix的隐藏维度（默认32）
        configs.enable_vmd_preprocessing: 是否使用VMD分解（默认True）
        configs.enable_time_adj_gate: 是否使用时间感知邻接门控（默认True）
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # 输入变量数与元数据列处理
        self.meta_cols = getattr(configs, 'meta_cols_to_drop', 0)
        self.base_channels = max(1, configs.n_input_features - self.meta_cols)
        self.individual = configs.individual

        # 自适应频带分解（VMD）配置
        self.enable_vmd_preprocessing = getattr(configs, 'enable_vmd_preprocessing', True)
        if self.enable_vmd_preprocessing:
            vmd_k = getattr(configs, 'vmd_k', 8)
            vmd_impl = getattr(configs, 'vmd_impl', 'fftbank')
            vmd_alpha = getattr(configs, 'vmd_alpha', 483.5238203360982)
            vmd_tau = getattr(configs, 'vmd_tau', 0.0)
            vmd_dc = getattr(configs, 'vmd_dc', 0)
            vmd_init = getattr(configs, 'vmd_init', 1)
            vmd_tol = getattr(configs, 'vmd_tol', 1e-7)
            vmd_max_iter = getattr(configs, 'vmd_max_iter', 500)
            print(f"impl:{vmd_impl}")
            self.vmd_decomposer = VMDDecomposer(
                K=vmd_k, impl=vmd_impl, alpha=vmd_alpha, tau=vmd_tau,
                DC=vmd_dc, init=vmd_init, tol=vmd_tol, max_iter=vmd_max_iter
            )
            self.vmd_k = vmd_k
        else:
            self.vmd_decomposer = None
            self.vmd_k = 0


        # VMD 拼接后的通道数
        self.channels = self.base_channels + self.vmd_k

        # 时间感知图门控（方案3）：基于时间标记生成邻接矩阵偏置
        self.enable_time_adj_gate = getattr(configs, 'enable_time_adj_gate', True)
        if self.enable_time_adj_gate:
            freq = getattr(configs, 'freq', '15min')
            time_d_model = getattr(configs, 'time_d_model', 32)
            time_hidden = getattr(configs, 'time_hidden', 64)
            self.time_gate_scale = getattr(configs, 'time_gate_scale', 0.1)
            # 将时间标记 x_mark_enc [B, L, d_inp] 映射为紧凑时间嵌入 [B, L, time_d_model]
            self.time_embed = TimeFeatureEmbedding(time_d_model, freq=freq)
            # 池化后的时间表示 -> 邻接偏置矩阵 [channels, channels]
            self.time_adj_bias_mlp = nn.Sequential(
                nn.Linear(time_d_model, time_hidden), nn.GELU(), nn.Linear(time_hidden, self.channels * self.channels)
            )

        # DLinear 分解模块
        self.decompsition = series_decomp(configs.moving_avg)

        # 是否使用图模块（方便消融实验）
        self.use_graph = getattr(configs, 'use_graph', True)

        if self.use_graph:
            # 图模块参数（可配置）
            graph_conv_channel = getattr(configs, 'graph_conv_channel', 16)
            graph_skip_channel = getattr(configs, 'graph_skip_channel', 32)
            gcn_depth = getattr(configs, 'gcn_depth', 2)
            graph_node_dim = getattr(configs, 'graph_node_dim', 10)
            graph_propalpha = getattr(configs, 'graph_propalpha', 0.05)
            dropout = getattr(configs, 'dropout', 0.2)

            # 特征嵌入层（将输入特征映射到图模型的维度）
            # 这样可以让 GraphBlock 的 d_model 独立于输入特征数
            self.d_model = 64
            self.feature_embedding = nn.Linear(self.channels, self.d_model)

            # GraphBlock - 季节性成分的图建模
            self.graph_seasonal = GraphBlock(
                c_out=self.channels,
                d_model=self.d_model,
                conv_channel=graph_conv_channel,
                skip_channel=graph_skip_channel,
                gcn_depth=gcn_depth,
                dropout=dropout,
                propalpha=graph_propalpha,
                seq_len=self.seq_len,
                node_dim=graph_node_dim
            )

            # GraphBlock - 趋势成分的图建模
            self.graph_trend = GraphBlock(
                c_out=self.channels,
                d_model=self.d_model,
                conv_channel=graph_conv_channel,
                skip_channel=graph_skip_channel,
                gcn_depth=gcn_depth,
                dropout=dropout,
                propalpha=graph_propalpha,
                seq_len=self.seq_len,
                node_dim=graph_node_dim
            )

            # 图输出投影层（将 d_model 映射回 channels）
            self.graph_projection = nn.Linear(self.d_model, self.channels)

        # 目标变量时序增强模块（LongConvMix）
        self.enable_target_longconv = getattr(configs, 'enable_target_longconv', False)
        if self.enable_target_longconv:
            longconv_kernels = getattr(configs, 'longconv_kernels', (5, 11, 23))
            longconv_dropout = getattr(configs, 'longconv_dropout', 0.1)
            longconv_hidden = getattr(configs, 'longconv_hidden', 32)  # 投影到的隐藏维度

            # 投影层：将目标变量从1维投影到hidden维
            self.target_proj_in = nn.Linear(1, longconv_hidden)

            # 对季节性和趋势分别建立 LongConvMix
            self.target_seasonal_conv = LongConvMix(
                hidden=longconv_hidden,
                kernels=longconv_kernels,
                dropout=longconv_dropout
            )
            self.target_trend_conv = LongConvMix(
                hidden=longconv_hidden,
                kernels=longconv_kernels,
                dropout=longconv_dropout
            )

            # 投影层：将hidden维投影回1维
            self.target_proj_out = nn.Linear(longconv_hidden, 1)

        # DLinear 预测头（线性层）
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x[:, :, self.meta_cols:]  # [B, seq_len, n_vars]

        # RevIN 归一化
        means = x.mean(1, keepdim=True).detach()  # [B, 1, n_vars]
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # VMD：对目标变量进行自适应频带分解并与原特征拼接
        if self.enable_vmd_preprocessing and self.vmd_decomposer is not None:
            tgt_idx0 = self.base_channels - 1  # 原始目标变量索引（拼接前）
            target_var = x[:, :, tgt_idx0:tgt_idx0+1]  # [B, L, 1]
            imf_components = self.vmd_decomposer(target_var)      # [B, L, K]
            x = torch.cat([x, imf_components], dim=2)             # [B, L, base+K]

        # DLinear 分解：季节性 + 趋势
        seasonal_init, trend_init = self.decompsition(x)  # [B, seq_len, n_vars]

        # ========== 关键创新：图建模阶段 ==========
        if self.use_graph:
            # 嵌入到图模型维度
            seasonal_emb = self.feature_embedding(seasonal_init)  # [B, seq_len, d_model]
            trend_emb = self.feature_embedding(trend_init)        # [B, seq_len, d_model]

            # 基于时间标记的邻接门控（方案3）
            adj_bias = None
            if self.enable_time_adj_gate and (x_mark_enc is not None):
                t_emb = self.time_embed(x_mark_enc)  # [B, L, time_d_model]
                t_ctx = t_emb.mean(dim=(0, 1))       # [time_d_model] 跨 batch/time 的全局上下文
                adj_bias_vec = self.time_adj_bias_mlp(t_ctx)  # [channels*channels]
                adj_bias = self.time_gate_scale * torch.tanh(adj_bias_vec).view(self.channels, self.channels)

            # 通过图模块增强变量间关系（带时间门控的邻接）
            seasonal_graph = self.graph_seasonal(seasonal_emb, adj_bias)    # [B, seq_len, d_model]
            trend_graph = self.graph_trend(trend_emb, adj_bias)             # [B, seq_len, d_model]

            # 投影回原始特征维度
            seasonal_enhanced = self.graph_projection(seasonal_graph)  # [B, seq_len, n_vars]
            trend_enhanced = self.graph_projection(trend_graph)        # [B, seq_len, n_vars]

            # 残差连接（保留原始分解信息）
            seasonal_init = seasonal_init + seasonal_enhanced
            trend_init = trend_init + trend_enhanced

        # ========== 目标变量时序增强阶段 ==========
        if self.enable_target_longconv:
            tgt_idx = self.base_channels - 1
            # 提取目标变量 [B, seq_len, 1]
            seasonal_target = seasonal_init[:, :, tgt_idx:tgt_idx+1]  # [B, seq_len, 1]
            trend_target = trend_init[:, :, tgt_idx:tgt_idx+1]        # [B, seq_len, 1]

            # 投影到高维空间 [B, seq_len, 1] -> [B, seq_len, hidden]
            seasonal_target_proj = self.target_proj_in(seasonal_target)  # [B, seq_len, hidden]
            trend_target_proj = self.target_proj_in(trend_target)        # [B, seq_len, hidden]

            # 应用 LongConvMix 进行多尺度时序增强
            seasonal_target_conv = self.target_seasonal_conv(seasonal_target_proj)  # [B, seq_len, hidden]
            trend_target_conv = self.target_trend_conv(trend_target_proj)          # [B, seq_len, hidden]

            # 投影回1维 [B, seq_len, hidden] -> [B, seq_len, 1]
            seasonal_target_enhanced = self.target_proj_out(seasonal_target_conv)  # [B, seq_len, 1]
            trend_target_enhanced = self.target_proj_out(trend_target_conv)        # [B, seq_len, 1]

            # 替换回原张量（仅增强目标变量）
            seasonal_init = seasonal_init.clone()
            trend_init = trend_init.clone()
            seasonal_init[:, :, tgt_idx:tgt_idx+1] = seasonal_target_enhanced
            trend_init[:, :, tgt_idx:tgt_idx+1] = trend_target_enhanced

        # ========== DLinear 预测阶段 ==========
        # 调整维度用于线性预测 [B, seq_len, n_vars] -> [B, n_vars, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            # 为每个变量独立预测
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # 所有变量共享线性层
            seasonal_output = self.Linear_Seasonal(seasonal_init)  # [B, n_vars, pred_len]
            trend_output = self.Linear_Trend(trend_init)

        # 合并季节性和趋势预测
        x = seasonal_output + trend_output  # [B, n_vars, pred_len]
        x = x.permute(0, 2, 1)              # [B, pred_len, n_vars]

        # 仅对目标变量进行反归一化并输出
        tgt_idx = self.base_channels - 1
        y = x[:, :, tgt_idx:tgt_idx+1]  # [B, pred_len, 1]
        y = y * stdev[:, :, tgt_idx:tgt_idx+1] + means[:, :, tgt_idx:tgt_idx+1]
        return y
