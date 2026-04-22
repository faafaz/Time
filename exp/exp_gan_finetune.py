"""
GAN微调实验类

用于对预训练的DLinear_Graph模型进行GAN对抗微调
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from exp.exp_basic import Exp_Basic
from tqdm import tqdm
import time
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class Exp_GAN_Finetune(Exp_Basic):
    """
    GAN微调实验类
    
    训练流程:
    1. 加载预训练的DLinear_Graph模型作为生成器
    2. 阶段1: 冻结生成器,训练判别器
    3. 阶段2: 联合训练,微调生成器
    """
    
    def __init__(self, args):
        super(Exp_GAN_Finetune, self).__init__(args)
        
        # GAN训练参数
        self.n_critic = getattr(args, 'gan_n_critic', 5)  # 判别器训练次数
        self.pretrain_epochs = getattr(args, 'gan_pretrain_epochs', 10)  # 判别器预训练轮数
        
    def _build_model(self):
        """构建GAN模型"""
        # 导入GAN模型
        from models import DLinear_Graph_GAN
        model = DLinear_Graph_GAN.Model(self.args).float()
        
        # 加载预训练的生成器权重
        if hasattr(self.args, 'pretrained_checkpoint') and self.args.pretrained_checkpoint:
            model.load_pretrained_generator(self.args.pretrained_checkpoint)
        else:
            self.logger.warning("[GAN] 未指定预训练模型路径,将从头训练生成器!")
        
        return model
    
    def _select_optimizer(self):
        """
        为生成器和判别器分别创建优化器
        
        Returns:
            g_optimizer: 生成器优化器
            d_optimizer: 判别器优化器
        """
        # 生成器优化器 - 使用较小的学习率进行微调
        g_lr = getattr(self.args, 'gan_g_lr', self.args.learning_rate * 0.1)
        g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=g_lr,
            betas=(0.5, 0.999)
        )
        
        # 判别器优化器
        d_lr = getattr(self.args, 'gan_d_lr', self.args.learning_rate)
        d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=d_lr,
            betas=(0.5, 0.999)
        )
        
        self.logger.info(f"[GAN] 生成器学习率: {g_lr}, 判别器学习率: {d_lr}")
        
        return g_optimizer, d_optimizer
    
    def _do_prediction(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        执行一次预测
        
        Returns:
            outputs: 模型预测输出
            batch_y: 真实标签
        """
        batch_x = batch_x.float().to(self.accelerator.device)
        batch_y = batch_y.float().to(self.accelerator.device)
        batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
        batch_y_mark = batch_y_mark.float().to(self.accelerator.device)
        
        # 构造decoder输入
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.accelerator.device)
        
        # 前向传播
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        return outputs, batch_y
    
    def train_discriminator(self, train_loader, d_optimizer, epoch):
        """
        训练判别器
        
        Args:
            train_loader: 训练数据加载器
            d_optimizer: 判别器优化器
            epoch: 当前轮数
        
        Returns:
            avg_d_loss: 平均判别器损失
            avg_real_acc: 平均真实数据准确率
            avg_fake_acc: 平均生成数据准确率
        """
        self.model.train()
        self.model.freeze_generator()  # 冻结生成器
        
        d_loss_list = []
        real_acc_list = []
        fake_acc_list = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), 
                                                                       total=len(train_loader),
                                                                       desc=f"Epoch {epoch} - 训练判别器"):
            # 获取预测结果
            with torch.no_grad():
                fake_data, real_data = self._do_prediction(batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            # 计算判别器损失
            d_loss, real_acc, fake_acc = self.model.compute_discriminator_loss(real_data, fake_data)
            
            # 反向传播
            d_optimizer.zero_grad()
            self.accelerator.backward(d_loss)
            d_optimizer.step()
            
            # 记录
            d_loss_list.append(d_loss.item())
            real_acc_list.append(real_acc)
            fake_acc_list.append(fake_acc)
        
        return np.mean(d_loss_list), np.mean(real_acc_list), np.mean(fake_acc_list)
    
    def train_generator(self, train_loader, g_optimizer, epoch):
        """
        训练生成器
        
        Args:
            train_loader: 训练数据加载器
            g_optimizer: 生成器优化器
            epoch: 当前轮数
        
        Returns:
            avg_g_loss: 平均生成器总损失
            avg_pred_loss: 平均预测损失
            avg_adv_loss: 平均对抗损失
        """
        self.model.train()
        self.model.unfreeze_generator()  # 解冻生成器
        
        g_loss_list = []
        pred_loss_list = []
        adv_loss_list = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader),
                                                                       total=len(train_loader),
                                                                       desc=f"Epoch {epoch} - 训练生成器"):
            # 获取预测结果
            fake_data, real_data = self._do_prediction(batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            # 计算预测损失
            pred_loss = self.criterion(fake_data, real_data)
            
            # 计算生成器损失(预测损失 + 对抗损失)
            g_loss, adv_loss = self.model.compute_generator_loss(fake_data, real_data, pred_loss)
            
            # 反向传播
            g_optimizer.zero_grad()
            self.accelerator.backward(g_loss)
            g_optimizer.step()
            
            # 记录
            g_loss_list.append(g_loss.item())
            pred_loss_list.append(pred_loss.item())
            adv_loss_list.append(adv_loss.item())
        
        return np.mean(g_loss_list), np.mean(pred_loss_list), np.mean(adv_loss_list)
    
    def train(self):
        """
        GAN微调主训练流程
        
        Returns:
            训练历史记录
        """
        # 获取数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # 创建优化器
        g_optimizer, d_optimizer = self._select_optimizer()
        
        # Accelerator准备
        train_loader, vali_loader, test_loader, self.model, g_optimizer, d_optimizer = \
            self.accelerator.prepare(train_loader, vali_loader, test_loader, 
                                     self.model, g_optimizer, d_optimizer)
        
        # 训练历史
        history = {
            'd_loss': [],
            'g_loss': [],
            'pred_loss': [],
            'adv_loss': [],
            'real_acc': [],
            'fake_acc': [],
            'val_loss': []
        }
        
        self.logger.info("=" * 80)
        self.logger.info("[GAN微调] 开始训练")
        self.logger.info(f"判别器预训练轮数: {self.pretrain_epochs}")
        self.logger.info(f"总训练轮数: {self.args.train_epochs}")
        self.logger.info(f"判别器训练频率: 每 {self.n_critic} 次生成器训练")
        self.logger.info("=" * 80)
        
        # 阶段1: 预训练判别器
        self.logger.info("\n[阶段1] 预训练判别器...")
        for epoch in range(self.pretrain_epochs):
            epoch_time = time.time()
            
            d_loss, real_acc, fake_acc = self.train_discriminator(train_loader, d_optimizer, epoch + 1)
            
            self.logger.info(
                f"预训练 Epoch {epoch + 1}/{self.pretrain_epochs} | "
                f"D_loss: {d_loss:.6f} | Real_acc: {real_acc:.4f} | Fake_acc: {fake_acc:.4f} | "
                f"Time: {(time.time() - epoch_time):.2f}s"
            )
        
        # 阶段2: 联合训练
        self.logger.info("\n[阶段2] 联合训练生成器和判别器...")
        best_val_loss = float('inf')
        
        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            
            # 训练判别器
            if epoch % self.n_critic == 0:
                d_loss, real_acc, fake_acc = self.train_discriminator(train_loader, d_optimizer, epoch + 1)
                history['d_loss'].append(d_loss)
                history['real_acc'].append(real_acc)
                history['fake_acc'].append(fake_acc)
            else:
                d_loss, real_acc, fake_acc = history['d_loss'][-1], history['real_acc'][-1], history['fake_acc'][-1]
            
            # 训练生成器
            g_loss, pred_loss, adv_loss = self.train_generator(train_loader, g_optimizer, epoch + 1)
            history['g_loss'].append(g_loss)
            history['pred_loss'].append(pred_loss)
            history['adv_loss'].append(adv_loss)
            
            # 验证
            val_loss = self._vali(vali_loader)
            history['val_loss'].append(val_loss)
            
            # 日志
            self.logger.info(
                f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                f"G_loss: {g_loss:.6f} | Pred_loss: {pred_loss:.6f} | Adv_loss: {adv_loss:.6f} | "
                f"D_loss: {d_loss:.6f} | Val_loss: {val_loss:.6f} | "
                f"Time: {(time.time() - epoch_time) / 60:.2f}min"
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.accelerator.is_main_process:
                    save_path = os.path.join(self.args.cur_model_save_path, 'checkpoint_gan_best')
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f"✓ 保存最佳模型 (Val_loss: {val_loss:.6f})")
        
        return history
    
    @torch.no_grad()
    def _vali(self, vali_loader):
        """验证"""
        self.model.eval()
        loss_list = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            outputs, real_data = self._do_prediction(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = self.criterion(outputs, real_data)
            loss_list.append(loss.item())
        
        self.model.train()
        return np.mean(loss_list)

