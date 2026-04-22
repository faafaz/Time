import torch
import torch.nn as nn
import torch.nn.functional as F

"""
S-MoLE: Stationarity-aware Mixture of Linear Experts

基于DLinear的多专家模型，通过可训练的平稳性判别路由（Router）为不同平稳性等级选择对应专家：
- Level 1 (Stable): 标准DLinear
- Level 2 (Semi): 趋势-残差分解，预测残差 + 趋势持久化
- Level 3 (NonStationary): 一阶差分 + 残差修正MLP + 方差估计（训练中未参与损失，仅供不确定性分析）

返回形状: [B, pred_len, 1]，与现有训练框架对齐。
评估阶段(model.eval())采用硬路由（hard gating）；训练阶段采用软路由（soft gating）。
"""

# -----------------
# 基础: DLinear分解模块
# -----------------
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        padding_len = (self.kernel_size - 1) // 2
        if padding_len > 0:
            front = x[:, 0:1, :].repeat(1, padding_len, 1)
            end = x[:, -1:, :].repeat(1, padding_len, 1)
            x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend

# -----------------
# Router: 平稳性特征 + 两层MLP
# -----------------
class StationarityRouter(nn.Module):
    """
    路由器：基于可解释的统计特征判别平稳性等级。
    特征设计遵循平稳性检验启发：
      - ACF(1), ACF(2): 自相关，对应Ljung-Box检验的关键量
      - 谱熵 Spectral Entropy: 序列复杂度/随机性（越大越接近白噪声）
      - 近似ADF: unit-root φ ≈ argmin ||x_t - φ x_{t-1}||
      - 差分方差 var(Δx): 非平稳强度的简单 proxy
      - 趋势强度 TrendStrength = var(Trend(x))（基于均值平滑）
      - KPSS proxy = var(cumsum(x - Trend(x))) / var(x)
      - LBQ proxy = sum_{k=1..h} r_k^2（h≈min(10, L/4)）
      - 主要频率能量占比 peak_psd_ratio
    以上特征均对平移缩放不敏感（在RevIN后计算更稳定）。
    """
    def __init__(self, num_levels=3, hidden_dim=64, ma_window=25, seq_len=96, temperature=1.0, dropout=0.0):
        super().__init__()
        self.num_levels = int(num_levels)
        self.ma_window = int(ma_window)
        self.seq_len = int(seq_len)
        self.temperature = float(temperature)
        self.dropout = nn.Dropout(float(dropout))
        # 用于趋势估计
        self._ma = moving_avg(self.ma_window, stride=1)
        # 特征维度固定为9
        self.input_dim = 9
        self.norm = nn.LayerNorm(self.input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_levels)
        )

    @staticmethod
    def _spectral_entropy(x, eps=1e-8):
        Xf = torch.fft.rfft(x, dim=1)
        psd = (Xf.real ** 2 + Xf.imag ** 2).clamp(min=eps)
        psd = psd / psd.sum(dim=1, keepdim=True)
        ent = -(psd * (psd + eps).log()).sum(dim=1)
        max_ent = torch.log(torch.tensor(psd.shape[1], device=x.device, dtype=x.dtype) + eps)
        return (ent / (max_ent + eps)).clamp(0, 1)

    @staticmethod
    def _acf(x, lag=1, eps=1e-8):
        x_mean = x.mean(dim=1, keepdim=True)
        x0 = x[:, :-lag] - x_mean
        x1 = x[:, lag:] - x_mean
        num = (x0 * x1).sum(dim=1)
        den = (x - x_mean).pow(2).sum(dim=1) + eps
        return (num / den).clamp(-1, 1)

    @staticmethod
    def _unit_root_phi(x, eps=1e-8):
        x_t = x[:, 1:]
        x_tm1 = x[:, :-1]
        num = (x_t * x_tm1).sum(dim=1)
        den = (x_tm1.pow(2).sum(dim=1) + eps)
        phi = (num / den).clamp(-1, 1)
        return phi

    @staticmethod
    def _peak_psd_ratio(x, eps=1e-8):
        Xf = torch.fft.rfft(x, dim=1)
        psd = (Xf.real ** 2 + Xf.imag ** 2).clamp(min=eps)
        total = psd.sum(dim=1, keepdim=True) + eps
        peak = psd.max(dim=1).values
        return (peak / total.squeeze(1)).clamp(0, 1)

    @staticmethod
    def _lbq_proxy(x, h=10, eps=1e-8):
        # Q ≈ sum r_k^2，忽略缩放因子
        B, L = x.shape
        h = max(1, min(h, L - 1))
        r_list = []
        for k in range(1, h + 1):
            r_list.append(StationarityRouter._acf(x, lag=k))
        r = torch.stack(r_list, dim=1)  # [B, h]
        return r.pow(2).sum(dim=1)

    def extract_features(self, x_n):
        """基于RevIN后的目标序列x_n: [B, L] 提取9维特征"""
        B, L = x_n.shape
        # 趋势与残差
        x_n3 = x_n.unsqueeze(-1)  # [B, L, 1]
        trend = self._ma(x_n3).squeeze(-1)  # [B, L]
        resid = x_n - trend
        # ACF、谱熵、peak能量占比、unit-root φ、差分方差
        acf1 = self._acf(x_n, lag=1)
        acf2 = self._acf(x_n, lag=2)
        spent = self._spectral_entropy(x_n)
        phi = self._unit_root_phi(x_n)
        diff = x_n[:, 1:] - x_n[:, :-1]
        diff_var = diff.var(dim=1, unbiased=False)
        # 趋势强度（在标准化域下相当于var(trend)）
        trend_strength = trend.var(dim=1, unbiased=False)
        # KPSS proxy
        cumsum_resid = torch.cumsum(resid, dim=1)
        kpss_proxy = (cumsum_resid.var(dim=1, unbiased=False) / (x_n.var(dim=1, unbiased=False) + 1e-8))
        # LBQ proxy
        h = max(1, min(10, L // 4))
        lbq = self._lbq_proxy(x_n, h=h)
        # 主要频率能量占比
        peak_ratio = self._peak_psd_ratio(x_n)
        feats = torch.stack([
            acf1, acf2, spent, phi, diff_var, trend_strength, kpss_proxy, lbq, peak_ratio
        ], dim=1)  # [B, 9]
        feats = self.norm(feats)
        return feats

    def forward(self, x_n_target, hard=False):
        feats = self.extract_features(x_n_target)
        logits = self.mlp(feats)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        weights = F.softmax(logits, dim=-1)
        if hard:
            idx = torch.argmax(weights, dim=-1)
            onehot = F.one_hot(idx, num_classes=self.num_levels).to(weights.dtype)
            return onehot, feats
        return weights, feats

# -----------------
# 专家实现
# -----------------
class DLinearStableExpert(nn.Module):
    def __init__(self, seq_len, pred_len, channels, individual=False, moving_avg_win=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        self.decomp = series_decomp(moving_avg_win)
        if individual:
            self.lin_sea = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
            self.lin_trd = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
        else:
            self.lin_sea = nn.Linear(seq_len, pred_len)
            self.lin_trd = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C] (已标准化)
        seasonal, trend = self.decomp(x)
        sea, trd = seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)  # [B, C, L]
        if self.individual:
            sea_o = x.new_zeros(x.size(0), self.channels, self.pred_len)
            trd_o = x.new_zeros(x.size(0), self.channels, self.pred_len)
            for i in range(self.channels):
                sea_o[:, i, :] = self.lin_sea[i](sea[:, i, :])
                trd_o[:, i, :] = self.lin_trd[i](trd[:, i, :])
        else:
            sea_o = self.lin_sea(sea)
            trd_o = self.lin_trd(trd)
        y = (sea_o + trd_o).permute(0, 2, 1)  # [B, pred_len, C]
        # 仅输出目标变量（假设最后一列为目标）
        return y[:, :, -1:]

class DLinearSemiExpert(nn.Module):
    def __init__(self, seq_len, pred_len, channels, individual=False, moving_avg_win=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        self.decomp = series_decomp(moving_avg_win)
        if individual:
            self.lin_res = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
        else:
            self.lin_res = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C] (已标准化)
        residual, trend = self.decomp(x)
        res = residual.permute(0, 2, 1)  # [B, C, L]
        if self.individual:
            res_o = x.new_zeros(x.size(0), self.channels, self.pred_len)
            for i in range(self.channels):
                res_o[:, i, :] = self.lin_res[i](res[:, i, :])
        else:
            res_o = self.lin_res(res)
        # 趋势持久化: 使用最后一个趋势值延拓
        last_trend = trend[:, -1:, :]  # [B, 1, C]
        trend_persist = last_trend.repeat(1, self.pred_len, 1)  # [B, pred_len, C]
        y = (res_o.permute(0, 2, 1) + trend_persist)  # [B, pred_len, C]
        return y[:, :, -1:]

class DLinearNonStationaryExpert(nn.Module):
    def __init__(self, seq_len, pred_len, feature_dim_for_mlp=16):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # 差分线性映射：仅针对目标变量（最后一列）
        self.lin_diff = nn.Linear(seq_len - 1, pred_len)
        # 残差修正与方差估计（基于Router特征）
        hd = max(32, feature_dim_for_mlp)
        self.residual_mlp = nn.Sequential(
            nn.Linear(feature_dim_for_mlp, hd), nn.ReLU(inplace=True), nn.Linear(hd, pred_len)
        )
        self.var_mlp = nn.Sequential(
            nn.Linear(feature_dim_for_mlp, hd), nn.ReLU(inplace=True), nn.Linear(hd, pred_len)
        )
        self.softplus = nn.Softplus()
        self.last_var = None  # [B, pred_len, 1]

    def forward(self, x, router_feats):
        # x: [B, L, C], router_feats: [B, F]
        x_tgt = x[:, :, -1]  # [B, L]
        diff = x_tgt[:, 1:] - x_tgt[:, :-1]  # [B, L-1]
        y_diff = self.lin_diff(diff)  # [B, pred_len]
        # 递推还原：从最后一个观测值开始累加
        last = x_tgt[:, -1:].detach()  # [B, 1]
        y_level = last + torch.cumsum(y_diff, dim=1)  # [B, pred_len]
        # 残差修正
        delta = self.residual_mlp(router_feats)  # [B, pred_len]
        y_level = y_level + delta
        # 方差估计（用于不确定性评估，训练不参与损失）
        var = self.softplus(self.var_mlp(router_feats)).unsqueeze(-1)  # [B, pred_len, 1]
        self.last_var = var
        return y_level.unsqueeze(-1)  # [B, pred_len, 1]

# -----------------
# 主模型
# -----------------
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.n_input_features
        self.individual = getattr(configs, 'individual', False)
        self.moving_avg = getattr(configs, 'moving_avg', 25)

        # 输出尺度：True=返回反归一化后的真实尺度；False=返回归一化尺度（便于与batch_y直接对齐）
        self.output_denorm = bool(getattr(configs, 'output_denorm', True))
        # 路由计算/训练策略：'hard' = 训练与推理都使用硬路由且仅计算被选中专家；'soft' = 训练软路由、推理硬路由
        self.router_mode = str(getattr(configs, 'router_mode', 'hard')).lower()

        # Router配置
        self.num_levels = int(getattr(configs, 'router_num_levels', 3))
        self.router_hidden = int(getattr(configs, 'router_hidden', 64))
        self.router_temperature = float(getattr(configs, 'router_temperature', 1.0))
        self.router_dropout = float(getattr(configs, 'router_dropout', 0.0))
        # 平衡正则权重（外部训练循环中使用）
        self.balance_lambda = float(getattr(configs, 'router_balance_lambda', 0.0))

        # Router（使用9维理论驱动特征）
        self.router = StationarityRouter(
            num_levels=self.num_levels,
            hidden_dim=self.router_hidden,
            ma_window=self.moving_avg,
            seq_len=self.seq_len,
            temperature=self.router_temperature,
            dropout=self.router_dropout,
        )

        # 构建与等级数严格一致的专家集合
        self.experts, self.expert_types = self._build_experts()

        # 记录值 (for interpretability)
        self.last_router_weights = None  # [B, K]
        self.last_router_level = None    # [B]
        self.last_var = None             # 来自非平稳专家

    def _normalize(self, x):
        # RevIN风格归一化: 按时间维度减均值/除方差
        means = x.mean(1, keepdim=True).detach()
        x_n = x - means
        stdev = torch.sqrt(torch.var(x_n, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_n = x_n / stdev
        return x_n, means, stdev

    def _denormalize_target(self, y, means, stdev):
        # 仅用目标变量（最后一列）的统计量反归一化，保持与DLinear一致
        y = y * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        y = y + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return y

    def _mixture(self, w, y_list):
        # w: [B, K], y_list: list of K tensors each [B, H, 1]
        B, H = y_list[0].shape[0], y_list[0].shape[1]
        y = 0.0
        for k, yk in enumerate(y_list):
            y = y + yk * w[:, k].view(B, 1, 1)
        return y

    def get_balance_loss(self):
        """KL(p || U) 对batch取平均，避免所有样本塌缩到某一专家"""
        if self.last_router_weights is None:
            return None
        w = self.last_router_weights.clamp_min(1e-8)
        K = w.shape[-1]
        # KL(p||U) = sum p*log(p/(1/K)) = sum p*log p + log K
        kl = (w * w.log()).sum(dim=-1) + torch.log(torch.tensor(float(K), device=w.device, dtype=w.dtype))
        return kl.mean()

    def _build_experts(self):
        """
        根据等级数构建专家列表（与Router等级数严格一致）。
        K=3: ['stable', 'semi', 'diff']
        K=4: ['stable_short', 'stable', 'semi', 'diff']
        K=5: ['stable_short', 'stable', 'semi', 'diff', 'stable_long']
        """
        assert 3 <= self.num_levels <= 5, f"S-MoLE 目前支持等级数为3~5，收到: {self.num_levels}"
        types = []
        if self.num_levels == 3:
            types = ['stable', 'semi', 'diff']
        elif self.num_levels == 4:
            types = ['stable_short', 'stable', 'semi', 'diff']
        else:  # 5
            types = ['stable_short', 'stable', 'semi', 'diff', 'stable_long']
        # kernel windows
        win_short = 11
        win_base = int(self.moving_avg)
        win_long = max(win_base * 2 + 1, 51)
        exps = []
        for t in types:
            if t == 'stable':
                exps.append(DLinearStableExpert(self.seq_len, self.pred_len, self.channels, self.individual, win_base))
            elif t == 'stable_short':
                exps.append(DLinearStableExpert(self.seq_len, self.pred_len, self.channels, self.individual, win_short))
            elif t == 'stable_long':
                exps.append(DLinearStableExpert(self.seq_len, self.pred_len, self.channels, self.individual, win_long))
            elif t == 'semi':
                exps.append(DLinearSemiExpert(self.seq_len, self.pred_len, self.channels, self.individual, win_base))
            elif t == 'diff':
                exps.append(DLinearNonStationaryExpert(self.seq_len, self.pred_len, feature_dim_for_mlp=9))
            else:
                raise ValueError(f"Unknown expert type: {t}")
        return nn.ModuleList(exps), types

    def _expert_forward(self, expert, x_n, feats):
        if isinstance(expert, DLinearNonStationaryExpert):
            return expert(x_n, feats)
        return expert(x_n)


    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [B, L, C]
        x_n, means, stdev = self._normalize(x)
        # 路由使用RevIN后的目标变量序列（对平移/缩放不敏感）
        x_tgt_n = x_n[:, :, -1]
        # 始终计算soft权重用于解释与均衡正则；根据router_mode/训练状态决定是否硬计算
        weights_soft, feats = self.router(x_tgt_n, hard=False)  # [B, K], [B, 9]
        levels = torch.argmax(weights_soft, dim=-1)  # [B]
        self.last_router_weights = weights_soft.detach()
        self.last_router_level = levels.detach()

        hard_compute = (self.router_mode == 'hard') or (not self.training)

        # 检查路由等级与专家数量一致
        if weights_soft.shape[-1] != len(self.experts):
            raise ValueError(f"Router levels ({weights_soft.shape[-1]}) != experts ({len(self.experts)}). 请保持一致。")

        if hard_compute:
            # 仅计算被选中的专家（按样本分组），并回填到对应位置
            B = x_n.size(0)
            device = x_n.device
            y_n = torch.zeros(B, self.pred_len, 1, device=device, dtype=x_n.dtype)
            self.last_var = None
            var_buf = None
            for k, exp in enumerate(self.experts):
                idx = (levels == k).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                x_sub = x_n.index_select(0, idx)
                feats_sub = feats.index_select(0, idx)
                if isinstance(exp, DLinearNonStationaryExpert):
                    yk = exp(x_sub, feats_sub)
                    # 记录方差，仅对被该专家选中的样本填充
                    if var_buf is None:
                        var_buf = torch.zeros(B, self.pred_len, 1, device=device, dtype=x_n.dtype)
                    var_buf.index_copy_(0, idx, exp.last_var)
                else:
                    yk = exp(x_sub)
                y_n.index_copy_(0, idx, yk)
            self.last_var = var_buf
        else:
            # 软路由：计算所有专家并加权混合（仅用于训练且router_mode='soft'）
            y_list = [self._expert_forward(exp, x_n, feats) for exp in self.experts]
            # 记录非平稳专家的方差（若存在）
            self.last_var = None
            for exp in self.experts:
                if isinstance(exp, DLinearNonStationaryExpert):
                    self.last_var = exp.last_var
                    break
            y_n = self._mixture(weights_soft, y_list)  # [B, H, 1]

        # 输出尺度控制
        if self.output_denorm:
            y = self._denormalize_target(y_n, means, stdev)
            return y
        else:
            return y_n

