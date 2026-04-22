"""
自适应损失加权模块

针对风电功率预测中的随机突变波峰波谷问题，提供多种自适应损失加权策略。

主要方法：
1. RampAwareLoss: 基于爬坡事件检测的自适应加权
2. GradientWeightedLoss: 基于梯度的自适应加权
3. QuantileWeightedLoss: 基于分位数的自适应加权
4. FocalMSELoss: 类Focal Loss的MSE变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RampAwareLoss(nn.Module):
    """
    基于爬坡事件检测的自适应损失加权
    
    对功率变化率大的时刻（波峰波谷）赋予更高的损失权重
    
    Args:
        base_criterion: 基础损失函数（如MSELoss）
        ramp_weight: 爬坡事件的权重倍数，默认3.0
        ramp_threshold_quantile: 爬坡事件阈值分位数，默认0.9（前10%的变化）
        use_target_ramp: 是否使用真实值的爬坡检测，默认True
    """
    def __init__(self, base_criterion=None, ramp_weight=3.0, 
                 ramp_threshold_quantile=0.9, use_target_ramp=True):
        super().__init__()
        self.base_criterion = base_criterion if base_criterion is not None else nn.MSELoss(reduction='none')
        self.ramp_weight = ramp_weight
        self.ramp_threshold_quantile = ramp_threshold_quantile
        self.use_target_ramp = use_target_ramp
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, T, C] 预测值
            target: [B, T, C] 真实值
        
        Returns:
            loss: 标量损失值
        """
        # 计算基础损失 [B, T, C]
        loss = self.base_criterion(pred, target)
        
        # 检测爬坡事件
        if self.use_target_ramp:
            # 使用真实值检测爬坡
            ramp_signal = target
        else:
            # 使用预测值检测爬坡
            ramp_signal = pred
        
        # 计算功率变化率 [B, T-1, C]
        if ramp_signal.shape[1] > 1:
            power_diff = torch.abs(ramp_signal[:, 1:, :] - ramp_signal[:, :-1, :])
            
            # 计算阈值（使用分位数）
            threshold = torch.quantile(power_diff, self.ramp_threshold_quantile)
            
            # 识别爬坡事件 [B, T-1, C]
            is_ramp = power_diff > threshold
            
            # 构建权重矩阵 [B, T, C]
            weights = torch.ones_like(loss)
            # 对爬坡时刻及其后一个时刻赋予更高权重
            weights[:, 1:, :][is_ramp] *= self.ramp_weight  # 爬坡发生时刻
            weights[:, :-1, :][is_ramp] *= self.ramp_weight  # 爬坡前一时刻
        else:
            weights = torch.ones_like(loss)
        
        # 加权损失
        weighted_loss = loss * weights
        
        return weighted_loss.mean()


class GradientWeightedLoss(nn.Module):
    """
    基于梯度的自适应损失加权
    
    根据功率变化的梯度大小动态调整权重，梯度越大权重越高
    
    Args:
        base_criterion: 基础损失函数
        alpha: 梯度权重系数，默认2.0
        smooth: 是否对梯度进行平滑，默认True
    """
    def __init__(self, base_criterion=None, alpha=2.0, smooth=True):
        super().__init__()
        self.base_criterion = base_criterion if base_criterion is not None else nn.MSELoss(reduction='none')
        self.alpha = alpha
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, T, C] 预测值
            target: [B, T, C] 真实值
        
        Returns:
            loss: 标量损失值
        """
        # 计算基础损失 [B, T, C]
        loss = self.base_criterion(pred, target)
        
        if target.shape[1] > 1:
            # 计算一阶梯度（变化率）[B, T-1, C]
            grad1 = torch.abs(target[:, 1:, :] - target[:, :-1, :])
            
            if self.smooth:
                # 对梯度进行平滑（移动平均）
                if grad1.shape[1] > 2:
                    grad1_smooth = F.avg_pool1d(
                        grad1.permute(0, 2, 1),  # [B, C, T-1]
                        kernel_size=3, stride=1, padding=1
                    ).permute(0, 2, 1)  # [B, T-1, C]
                    grad1 = grad1_smooth
            
            # 归一化梯度到[0, 1]
            grad1_norm = grad1 / (grad1.max() + 1e-8)
            
            # 构建权重：1 + alpha * gradient
            # 梯度大的地方权重高
            weights = torch.ones_like(loss)
            weights[:, 1:, :] = 1.0 + self.alpha * grad1_norm
            weights[:, :-1, :] = torch.maximum(
                weights[:, :-1, :],
                1.0 + self.alpha * grad1_norm
            )
        else:
            weights = torch.ones_like(loss)
        
        # 加权损失
        weighted_loss = loss * weights
        
        return weighted_loss.mean()


class QuantileWeightedLoss(nn.Module):
    """
    基于分位数的自适应损失加权
    
    对极端值（高分位数和低分位数）赋予更高权重
    
    Args:
        base_criterion: 基础损失函数
        extreme_quantiles: 极端值分位数阈值，默认(0.1, 0.9)
        extreme_weight: 极端值权重倍数，默认3.0
    """
    def __init__(self, base_criterion=None, extreme_quantiles=(0.1, 0.9), extreme_weight=3.0):
        super().__init__()
        self.base_criterion = base_criterion if base_criterion is not None else nn.MSELoss(reduction='none')
        self.extreme_quantiles = extreme_quantiles
        self.extreme_weight = extreme_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, T, C] 预测值
            target: [B, T, C] 真实值
        
        Returns:
            loss: 标量损失值
        """
        # 计算基础损失 [B, T, C]
        loss = self.base_criterion(pred, target)
        
        # 计算分位数阈值
        q_low = torch.quantile(target, self.extreme_quantiles[0])
        q_high = torch.quantile(target, self.extreme_quantiles[1])
        
        # 识别极端值
        is_extreme = (target < q_low) | (target > q_high)
        
        # 构建权重
        weights = torch.ones_like(loss)
        weights[is_extreme] *= self.extreme_weight
        
        # 加权损失
        weighted_loss = loss * weights
        
        return weighted_loss.mean()


class FocalMSELoss(nn.Module):
    """
    类Focal Loss的MSE变体
    
    对难以预测的样本（损失大的样本）赋予更高权重
    
    Args:
        gamma: 聚焦参数，默认2.0
        alpha: 缩放参数，默认1.0
    """
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, T, C] 预测值
            target: [B, T, C] 真实值
        
        Returns:
            loss: 标量损失值
        """
        # 计算MSE损失 [B, T, C]
        mse_loss = F.mse_loss(pred, target, reduction='none')
        
        # 归一化损失到[0, 1]
        mse_norm = mse_loss / (mse_loss.max() + 1e-8)
        
        # Focal权重：损失越大，权重越高
        focal_weight = (1 + self.gamma * mse_norm) ** self.gamma
        
        # 加权损失
        weighted_loss = self.alpha * focal_weight * mse_loss
        
        return weighted_loss.mean()

