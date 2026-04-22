import numpy as np
import torch
import logging


# 让JSON属性可以通过.来访问, 让代码看起来更整洁
class DotDict(dict):
    """一个简单的类，使得字典属性可以通过点号访问"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def adjust_learning_rate(optimizer, epoch, learning_rate, lradj_type="default"):
    """
    学习率调整函数，支持多种调整策略
    """
    if lradj_type == "type1":
        # DUET type1策略
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj_type == "type2":
        # DUET type2策略
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj_type == "type3":
        # DUET type3策略
        lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif lradj_type == "constant":
        # 恒定学习率
        lr_adjust = {epoch: learning_rate}
    else:
        # 默认策略
        lr_adjust = {epoch: learning_rate * (1.0 ** ((epoch - 1) // 1))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_mode=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode
        self.solar_logger = logging.getLogger('solar')

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.solar_logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.solar_logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


import numpy as np
import torch
import torch.nn as nn
import random
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('agg')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

   
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    

def save_to_csv(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    data = pd.DataFrame({'true': true, 'preds': preds})
    data.to_csv(name, index=False, sep=',')


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_weights(weights, name='./pic/test.pdf'):
    """
    Weights visualization
    """
    fig, ax = plt.subplots()
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap='YlGnBu')
    fig.colorbar(im, pad=0.03, location='top')
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims  # The new order of dimensions

    def forward(self, x):
        return x.permute(*self.dims)
    
class Reshape(nn.Module):
    def __init__(self, *dims):
        super(Reshape, self).__init__()
        self.dims = dims  # The new order of dimensions

    def forward(self, x):
        return x.reshape(*self.dims)

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# def set_random_seed(random_seed):
#     torch.manual_seed(random_seed)
#     # torch.cuda.manual_seed(random_seed)
#     # torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
#     # torch.backends.cudnn.deterministic = True
#     # torch.backends.cudnn.benchmark = False
#     np.random.seed(random_seed)
#     random.seed(random_seed)