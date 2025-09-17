import logging
import os
from abc import abstractmethod, ABC
from torch import optim
import torch
from torch.optim import lr_scheduler

from models import Transformer, TimeXer, TimesNet, PatchTST, iTransformer, Informer, DLinear, Autoformer, TimeLLM, \
    LLMMixer, TimeLLM_MSPF_GATEFUSION, WPMixer, DualSignalModel
from models.Q1_2019_CNN_LSTM import PV_CNN, PV_LSTM,PV_CNNLSTM
from data_provider.data_loader import get_data_loader
import torch.nn as nn
from utils.log import create_file_handler
from accelerate import Accelerator

from utils.tools import adjust_learning_rate


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
            "Autoformer": Autoformer,
            "TimeLLM": TimeLLM,
            "LLMMixer": LLMMixer,
            "TimeLLM_MSPF_GATEFUSION": TimeLLM_MSPF_GATEFUSION,
            "PV_CNN": PV_CNN,
            "PV_LSTM": PV_LSTM,
            "PV_CNNLSTM": PV_CNNLSTM,
            "WPMixer": WPMixer,
            "DualSignalModel": DualSignalModel
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
        self.criterion = self._select_criterion()

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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_lr_scheduler(self, train_loader):
        scheduler = None
        if self.args.lradj == "adjust_fuc":
            scheduler = adjust_learning_rate
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
        else:
            raise NotImplementedError

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
