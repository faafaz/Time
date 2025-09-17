import os
from pathlib import Path
from exp.exp_basic import Exp_Basic
import torch.nn as nn
from utils.tools import EarlyStopping
from tqdm import tqdm
import torch
import time
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class Exp_Ultra_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Ultra_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # accelerate会自动处理模型的分布式包装，无需手动DataParallel
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        return model

    def _do_same(self, args_tuple):
        batch_x, batch_y, batch_x_mark, batch_y_mark = args_tuple
        batch_x = batch_x.float().to(self.accelerator.device)
        batch_y = batch_y.float().to(self.accelerator.device)
        batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
        batch_y_mark = batch_y_mark.float().to(self.accelerator.device)
        # batch_x = batch_x[:, :, 0:1]  # 只要功率
        # 由于label_len=0，batch_y[:, :0, :]会返回一个空张量
        # 与全零张量dec_inp在dim=1维度拼接后，结果就是原始的全零张量dec_inp本身
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.accelerator.device)
        # print(batch_x.shape)
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # print(f"预测值: {outputs.shape}, 真实值: {batch_y.shape}")
        # 计算损失
        # if self.args.get_pred_type == 'last':
        #     # 取预测的最后一个时间步计算损失
        #     pred_last = outputs[:, -1, :]  # 取最后一个时间步
        #     true_last = batch_y[:, -1, :]  # 取真实值的最后一个时间步
        #     loss = self.criterion(pred_last, true_last)
        # elif self.args.get_pred_type == 'first':
        #     # 取第一个时间步计算损失
        #     pred_first = outputs[:, 0, :]  # 取第一个时间步
        #     true_first = batch_y[:, 0, :]  # 取真实值的第一个时间步
        #     loss = self.criterion(pred_first, true_first)
        # elif self.args.get_pred_type == 'all':
        #     # 取所有时间步计算损失
        #     loss = self.criterion(outputs, batch_y)
        # print(f"预测值: {outputs[:10]}, 真实值: {batch_y[:10]}")
        loss = self.criterion(outputs, batch_y)
        return loss, outputs

    def train(self):
        # 获取训练数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # 初始化学习率调度器
        lr_scheduler = self._select_lr_scheduler(train_loader)
        train_loader, vali_loader, test_loader, self.model, self.model_optim = self.accelerator.prepare(
            train_loader, vali_loader, test_loader, self.model, self.model_optim)
        # 训练过程中监控模型的性能，并在性能不再提升时提前停止训练
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        train_loss_epoch_list = []  # 保存每个epoch的平均损失
        val_loss_epoch_list = []  # 保存每个epoch的验证集损失
        test_loss_epoch_list = []  # 保存每个epoch的测试集损失
        for epoch in range(self.args.train_epochs):
            # 设置为训练模式
            self.model.train()

            # 记录每次预测模型的损失
            train_loss_list = []
            epoch_time = time.time()  # epoch开始时间
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader),
                                                                          total=len(train_loader)):
                self.model_optim.zero_grad()  # 将模型的参数的梯度归零

                loss, outputs = self._do_same((batch_x, batch_y, batch_x_mark, batch_y_mark))  # 预测并返回损失
                train_loss_list.append(loss.item())  # 保存当前损失

                self.accelerator.backward(loss)  # 对损失值进行反向传播，计算模型参数的梯度
                self.model_optim.step()  # 根据梯度更新模型参数

                # OneCycleLR 的设计理念是在整个训练过程中形成一个完整的学习率周期,所以每个batch都调整
                if self.args.lradj == "OneCycleLR":
                    lr_scheduler.step()

            # 训练集、验证集和测试集损失（当前epoch平均）
            train_loss_a = np.average(train_loss_list)
            vali_loss_a = self._vali(vali_loader)
            test_loss_a = self._test(test_loader) 
            # 只在主进程打印日志
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Epoch: {epoch + 1} cost time:{((time.time() - epoch_time) / 60):.2} min | "
                    f"Train: {train_loss_a:.7f} Vali: {vali_loss_a:.7f} test: {test_loss_a:.7f}")

            # 根据验证集损失来决定是否早停，停止后保存模型
            early_stopping(vali_loss_a, self.model, self.args.cur_model_save_path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            # 按不同的epoch动态调整学习率
            if self.args.lradj == "adjust_fuc":
                lr_scheduler(self.model_optim, epoch + 1, self.args.learning_rate)  # 保存每个epoch的平均损失

            # 测试打印各参数组学习率
            for i, group in enumerate(self.model_optim.param_groups):
                # print(f"Group {i} lr: {group['lr']:.4f}")
                self.logger.info(f"Group {i} lr: {group['lr']:.10f}")

            # 保存 训练集、验证集和测试集损失（当前epoch平均）
            train_loss_epoch_list.append(train_loss_a)
            val_loss_epoch_list.append(vali_loss_a)
            test_loss_epoch_list.append(test_loss_a)

        return train_loss_epoch_list, val_loss_epoch_list, test_loss_epoch_list

    @torch.no_grad()
    def _vali(self, vali_loader):
        # 设置模型为评估模式
        self.model.eval()

        # 记录验证集中每次预测模型的损失
        loss_list = []

        # 预测
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader), total=len(vali_loader)):
            loss, outputs = self._do_same((batch_x, batch_y, batch_x_mark, batch_y_mark))
            loss_list.append(loss.item())
        average = np.average(loss_list)

        # 将模型改回训练模型
        self.model.train()
        return average

    @torch.no_grad()
    def _test(self, test_loader):
        # 设置模型为评估模式
        self.model.eval()

        # 记录验证集中每次预测模型的损失
        loss_list = []

        # 预测
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader)):
            loss, outputs = self._do_same((batch_x, batch_y, batch_x_mark, batch_y_mark))
            loss_list.append(loss.item())
        average = np.average(loss_list)

        # 将模型改回训练模型
        self.model.train()
        return average

    @torch.no_grad()
    def predict(self):
        # 获取测试数据
        test_data, test_loader = self._get_data(flag='test')

        # 加载模型权重
        weight_file_path = Path(self.args.test_folder_path).parent
        checkpoint_path = os.path.join(weight_file_path, 'checkpoint')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # 设置模型为评估模式
        self.model.eval()

        pred_list = []
        true_list = []
        # 预测
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader)):
            loss, outputs = self._do_same((batch_x, batch_y, batch_x_mark, batch_y_mark))
    
            # 反标准化拿真实功率
            for _i in range(outputs.shape[0]):  # 遍历batch_size
                # 预测值反标准化 - 整个序列(16,1)
                pred_seq = outputs[_i].cpu().detach().numpy()  # 形状: (16, 1)
                pred_inverse = test_data.inverse_transform(pred_seq)  # 反标准化整个序列

                # 真实值反标准化 - 整个序列(16,1)
                true_seq = batch_y[_i].cpu().detach().numpy()  # 形状: (16, 1)
                true_inverse = test_data.inverse_transform(true_seq)  # 反标准化整个序列

                if self.args.get_pred_type == 'last':
                    pre_one = pred_inverse[-1, 0]  # 取最后一个值
                    true_one = true_inverse[-1, 0]  # 取最后一个值
                    pred_list.append(pre_one)
                    true_list.append(true_one)
                elif self.args.get_pred_type == 'first':
                    pre_one = pred_inverse[0, 0]  # 取第一个值
                    true_one = true_inverse[0, 0]  # 取第一个值
                    pred_list.append(pre_one)
                    true_list.append(true_one)
                elif self.args.get_pred_type == 'all':
                    # 取所有值 - 将整个序列展平并添加到列表中
                    pred_all = pred_inverse.flatten()  # 展平为一维数组
                    true_all = true_inverse.flatten()  # 展平为一维数组
                    pred_list.extend(pred_all)  # 使用extend添加所有元素
                    true_list.extend(true_all)  # 使用extend添加所有元素
        return pred_list, true_list
