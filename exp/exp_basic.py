import logging
import os
from abc import abstractmethod, ABC
from torch import optim
import torch
from torch.optim import lr_scheduler

from models import Transformer, TimeXer, TimesNet, PatchTST, iTransformer, Informer, DLinear, Autoformer, TimeLLM, \
    LLMMixer, TimeLLM_MSPF_GATEFUSION, WPMixer, DUET, DLinear_Graph, iTransformer_xLSTM, iTransformer_xLSTM_VMD, S_MoLE, DLinear_ABDM, i_transformer_xlstm_vmd_pre, itransformer_vmd_refinement
from data_provider.data_loader import get_data_loader
import torch.nn as nn
from utils.log import create_file_handler
from accelerate import Accelerator
from third_party.DILATE_master.loss import dilate_loss
from utils.tools import adjust_learning_rate
from third_party.DILATE_master.loss.dilate_loss import dilate_loss as dilate_loss_func  # 重命名避免冲突
import torch.nn.functional as F



class DilateLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=0.01):
        super(DilateLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred和target的形状: [batch_size, seq_len, 1] (根据dilate_loss的要求)
        # 注意：dilate_loss函数需要outputs, targets, alpha, gamma, device
        # 这里pred和target的形状需要是(batch_size, N_output, 1)，其中N_output是预测长度
        # 但是我们的模型输出可能是(batch_size, pred_len, num_features)，而num_features=1（单变量预测）
        # 所以直接使用
        device = pred.device
        # 注意：dilate_loss函数返回三个值，我们只需要第一个（总损失）
        loss, _, _ = dilate_loss_func(pred, target, self.alpha, self.gamma, device)
        return loss

class SpectralRampLoss(nn.Module):
    def __init__(self, w_ramp=2.0, w_freq=0.1, catch_lag=True):
        """
        w_ramp: 爬坡/差分损失的权重
        w_freq: 频域损失的权重
        catch_lag: 是否开启针对滞后的方向惩罚 (Direction Penalty)
        """
        super(SpectralRampLoss, self).__init__()
        self.w_ramp = w_ramp
        self.w_freq = w_freq
        self.catch_lag = catch_lag
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        """
        输入形状: [Batch, Seq_Len, Features] 或 [Batch, Seq_Len]
        建议 Seq_Len 为时间维度
        """
        # ==========================
        # Part 1: 基础数值损失 (Time Domain Value)
        # ==========================
        loss_mse = self.mse(preds, targets)

        # ==========================
        # Part 2: 物理爬坡损失 (Ramp / Physics)
        # ==========================
        # 计算一阶差分: P(t) - P(t-1)
        delta_pred = preds[:, 1:] - preds[:, :-1]
        delta_target = targets[:, 1:] - targets[:, :-1]

        # 2.1 差分幅度的 MSE (让变化的快慢一致)
        loss_ramp_basic = F.mse_loss(delta_pred, delta_target)
        
        loss_ramp = loss_ramp_basic

        # 2.2 抗滞后方向惩罚 (可选，推荐开启)
        if self.catch_lag:
            # 符号相反说明方向预测错了
            diff_sign = torch.sign(delta_pred) * torch.sign(delta_target)
            direction_error_mask = (diff_sign < 0).float()
            # 对方向错误的点给予额外 2 倍权重的惩罚
            loss_direction = (direction_error_mask * (delta_pred - delta_target)**2).mean()
            loss_ramp = loss_ramp + 2.0 * loss_direction

        # ==========================
        # Part 3: 频域一致性损失 (Frequency Domain)
        # ==========================
        # 使用 RFFT (实数快速傅里叶变换)
        # dim=1 假设是时间维度 [Batch, Time, Feat]
        pred_fft = torch.fft.rfft(preds, dim=1)
        target_fft = torch.fft.rfft(targets, dim=1)

        # 我们比较频谱的"振幅" (Amplitude)
        # 这一步能消除"阶梯状"，因为阶梯波和真实波的频谱差异巨大
        loss_spectral = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        # ==========================
        # Total Loss
        # ==========================
        total_loss = 0.5 * loss_mse + self.w_ramp * loss_ramp + self.w_freq * loss_spectral
        # print(f"Loss: {total_loss:.4f}, Loss_MSE: {loss_mse:.4f}, Loss_Ramp: {loss_ramp:.4f}, Loss_Spectral: {loss_spectral:.4f}")
        
        return total_loss


class AntiLagRampLoss(nn.Module):
    def __init__(self, alpha=5.0, beta=2.0):
        """
        alpha: 爬坡数值误差的权重
        beta:  方向惩罚的权重 (通常设置大一点，如 2.0 或 5.0)
        """
        super(AntiLagRampLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        # 1. 基础 MSE
        loss_mse = self.mse(preds, targets)

        # 2. 计算差分 (变化量)
        delta_pred = preds[:, 1:] - preds[:, :-1]
        delta_target = targets[:, 1:] - targets[:, :-1]

        # 3. 爬坡数值误差 (让变化幅度一致)
        loss_ramp_mag = F.mse_loss(delta_pred, delta_target)

        # 4. 方向一致性惩罚 (Direction Penalty)
        # torch.sign 返回 -1, 0, 1
        # 如果符号不一致，sign_product 为 -1，我们希望惩罚它
        # 构造 mask: 当 sign(pred) != sign(target) 时，给予额外惩罚
        # 使用 soft sign (tanh) 可以保持梯度可导，但在简单场景下直接用 ReLU 逻辑更直观
        
        # 逻辑：如果方向相反，diff_sign 为负。
        diff_sign = torch.sign(delta_pred) * torch.sign(delta_target)
        
        # 找出方向错误的点 (diff_sign < 0 的位置)
        direction_error_mask = (diff_sign < 0).float()
        
        # 计算方向惩罚：在方向错误的时刻，误差被 beta 放大
        # 这里我们重用 magnitude 误差，但在方向错的时候乘以 beta
        loss_direction = (direction_error_mask * (delta_pred - delta_target)**2).mean()

        # 5. 总损失
        # 结构: 值误差 + alpha * (形态误差 + beta * 方向错误惩罚)
        
        total_loss = 0.5 * loss_mse + self.alpha * (loss_ramp_mag + self.beta * loss_direction)
        # print(f"Loss: {total_loss:.4f}, Loss_MSE: {loss_mse:.4f}, Loss_Ramp_Mag: {loss_ramp_mag:.4f}, Loss_Direction: {loss_direction:.4f}")
        return total_loss


class RMSELoss(nn.Module):
    """
    RMSE (Root Mean Squared Error) 损失函数
    计算公式: RMSE = sqrt(MSE) = sqrt(mean((pred - target)^2))
    """
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值
            target: 真实值
        Returns:
            RMSE损失值
        """
        return torch.sqrt(self.mse(pred, target))


class Exp_Basic(ABC):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'TimeXer': TimeXer,
            'TimesNet': TimesNet,
            "PatchTST": PatchTST,
            "iTransformer": iTransformer,
            "Informer": Informer,
            "DLinear": DLinear,
            "DLinear_Graph": DLinear_Graph,
            "Autoformer": Autoformer,
            "TimeLLM": TimeLLM,
            "LLMMixer": LLMMixer,
            "TimeLLM_MSPF_GATEFUSION": TimeLLM_MSPF_GATEFUSION,
            "WPMixer": WPMixer,
            "DUET": DUET,
            "iTransformer_xLSTM": iTransformer_xLSTM,
            "iTransformer_xLSTM_VMD": iTransformer_xLSTM_VMD,
            "i_transformer_xlstm_vmd_pre": i_transformer_xlstm_vmd_pre,
            "itransformer_vmd_refinement": itransformer_vmd_refinement,
            "S_MoLE": S_MoLE,
            "DLinear_ABDM": DLinear_ABDM,
        }
        # 日志
        logger = logging.getLogger('solar')
        if args.run_type == 0:
            log_path = os.path.join(args.cur_model_save_path, f"{args.model_name}_run.log")
            logger.addHandler(create_file_handler(log_path))
        elif args.run_type == 1:
            log_path = os.path.join(args.test_folder_path, f"{args.model_name}_test.log")
            logger.addHandler(create_file_handler(log_path))

        self.logger = logger
        # 加速器
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        # 模型
        self.model = self._build_model().to(self.device)
        # 训练的时候记录模型参数
        if args.run_type == 0:
            self._log_model_param()
        # 优化器
        self.model_optim = self._select_optimizer()
        # 损失函数
        self.criterion = self._select_criterion(getattr(self.args, 'loss', 'MSE'))

    @abstractmethod
    def _build_model(self):
        pass

    def _log_model_param(self):
        # 记录训练参数
        trained_parameters = []
        numel_params = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)
                num_params = p.numel()
                numel_params = numel_params + num_params
                self.logger.info(f"Layer: {name}, Parameters: {num_params}")
        self.logger.info(f"All Parameters Num: {len(trained_parameters)}, All Numel:{numel_params}")
        return trained_parameters

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate,
            weight_decay=1e-4,  # 权重衰减 (NEW)
            betas=(0.9, 0.999)
        )
        return model_optim

    def _select_lr_scheduler(self, train_loader):
        scheduler = None
        if self.args.lradj == "adjust_fuc":
            scheduler = lambda optimizer, epoch, lr: adjust_learning_rate(optimizer, epoch, lr, "default")
        elif self.args.lradj == "type1":
            scheduler = lambda optimizer, epoch, lr: adjust_learning_rate(optimizer, epoch, lr, "type1")
        elif self.args.lradj == "type2":
            scheduler = lambda optimizer, epoch, lr: adjust_learning_rate(optimizer, epoch, lr, "type2")
        elif self.args.lradj == "type3":
            scheduler = lambda optimizer, epoch, lr: adjust_learning_rate(optimizer, epoch, lr, "type3")
        elif self.args.lradj == "constant":
            scheduler = lambda optimizer, epoch, lr: adjust_learning_rate(optimizer, epoch, lr, "constant")
        elif self.args.lradj == "OneCycleLR":
            scheduler = lr_scheduler.OneCycleLR(optimizer=self.model_optim,
                                                steps_per_epoch=len(train_loader),
                                                pct_start=0.2,
                                                div_factor=10,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        return scheduler

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'RMSE':
            return RMSELoss()
        elif loss_name == 'MAE':
            return nn.L1Loss()
        elif loss_name == 'huber':
            return nn.HuberLoss(delta=0.5)
        elif loss_name == 'dilate':
            return DilateLoss(alpha=self.args.alpha, gamma=self.args.gamma)
        elif loss_name == 'AntiLagRampLoss':
            return AntiLagRampLoss()
        elif loss_name == "SpectralRampLoss":
            return SpectralRampLoss()
        else:
            raise NotImplementedError(f"Loss function {loss_name} not supported")

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = get_data_loader(self.args, flag)
        return data_set, data_loader
