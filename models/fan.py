"""
Frequency Attention Network (FAN) - 频率注意力网络。

FAN 首先从原始序列中分离出低频主成分，然后对残差和主成分分别预测，
更好地捕捉不同频率成分的特征。
"""

import torch
import torch.nn as nn


def main_freq_part(x, k, rfft=True):
    """
    提取主要频率成分。

    Args:
        x: 输入序列 [B, L, N]
        k: 保留top-k频率分量
        rfft: 是否使用rfft

    Returns:
        norm_input: 残差 (输入 - 主要频率成分)
        x_filtered: 提取出的主要频率成分
    """
    # freq normalization
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    k_values = torch.topk(xf.abs(), k, dim=1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    norm_input = x - x_filtered
    return norm_input, x_filtered


class FAN(nn.Module):
    """
    FAN - 先从原始序列中减去bottom k频率分量，再分别预测。

    Args:
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        enc_in: 输入特征数
        freq_topk: 保留top-k主要频率分量，默认20
        rfft: 是否使用实数FFT，默认True
    """
    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20, rfft=True, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        print("freq_topk : ", self.freq_topk)
        self.rfft = rfft

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)

    def loss(self, true):
        """计算频率分解损失"""
        B, O, N = true.shape
        residual, pred_main = main_freq_part(true, self.freq_topk, self.rfft)

        lf = nn.functional.mse_loss
        return lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual)

    def normalize(self, input):
        """归一化：分解输入"""
        # (B, T, N)
        bs, length, dim = input.shape
        norm_input, x_filtered = main_freq_part(input, self.freq_topk, self.rfft)
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1, 2), input.transpose(1, 2)).transpose(1, 2)

        return norm_input.reshape(bs, length, dim)

    def denormalize(self, input_norm):
        """反归一化：重组输出"""
        # input: (B, O, N)
        bs, length, dim = input_norm.shape
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal

        return output.reshape(bs, length, dim)

    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x)


class MLPfreq(nn.Module):
    """MLP预测频率成分"""
    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )

        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)
