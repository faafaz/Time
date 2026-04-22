import os
from pathlib import Path
from exp.exp_basic import Exp_Basic
import torch.nn as nn
from utils.tools import EarlyStopping
from tqdm import tqdm
import torch
import torch.optim as optim
import time
import numpy as np
import warnings
import pandas as pd
from utils.vmd_loss import vmd_loss

warnings.filterwarnings('ignore')


class Exp_Ultra_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Ultra_Short_Term_Forecast, self).__init__(args)
        # Plus-model specific switch
        self.plus_mode = (args.model_name == 'iTransformerPlus')

    def _build_model(self):
        # accelerate会自动处理模型的分布式包装
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        return model

    def _select_optimizer(self):
        """
        实现分层学习率：
        1. VMD 模块参数 -> 使用 vmd_learning_rate
        2. 其他模型参数 -> 使用全局 learning_rate
        """
        import torch.optim as optim

        # 获取 VMD 专用学习率
        vmd_lr = getattr(self.args, 'vmd_learning_rate', 0.001)
        weight_decay = getattr(self.args, 'weight_decay', 0.0)
        
        vmd_params = []
        other_params = []
        
        # 遍历参数进行分组
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 判断是否属于 VMD 模块
            if 'vmd_' in name or 'decomposer' in name:
                vmd_params.append(param)
                if not hasattr(self, '_vmd_params_logged'):
                    print(f"  >>> [Optimizer] VMD Parameter detected (LR={vmd_lr}): {name}")
            else:
                other_params.append(param)
        
        self._vmd_params_logged = True

        # 创建参数组
        optim_groups = [
            {'params': other_params, 'lr': self.args.learning_rate},
            {'params': vmd_params, 'lr': vmd_lr}
        ]
        
        if len(vmd_params) > 0 and not hasattr(self, '_vmd_groups_logged'):
            print(f"  >>> [Optimizer] Created groups: VMD params ({len(vmd_params)}), Other params ({len(other_params)})")
            self._vmd_groups_logged = True

        model_optim = optim.Adam(optim_groups, weight_decay=weight_decay)
        return model_optim

    def _do_same(self, args_tuple):
        batch_x, batch_y, batch_x_mark, batch_y_mark = args_tuple
        batch_x = batch_x.float().to(self.accelerator.device)
        batch_y = batch_y.float().to(self.accelerator.device)
        batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
        batch_y_mark = batch_y_mark.float().to(self.accelerator.device)
        
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.accelerator.device)

        # if self.args.model_name == 'iTransformer_xLSTM_VMD_Preprocessed':
        #     outputs, modes = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        #     v_loss = vmd_loss(batch_x[:, :, 1:2], modes)
        #     loss = self.criterion(outputs, batch_y)
        #     loss = loss + v_loss
        # else:
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        loss = self.criterion(outputs, batch_y)
        
        return loss, outputs

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.model_optim = self._select_optimizer()
        lr_scheduler = self._select_lr_scheduler(train_loader)

        train_loader, vali_loader, test_loader, self.model, self.model_optim = self.accelerator.prepare(
            train_loader, vali_loader, test_loader, self.model, self.model_optim)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        train_loss_epoch_list = []  
        val_loss_epoch_list = []  
        test_loss_epoch_list = []
        
        lambda_sparsity = getattr(self.args, 'vmd_sparsity_lambda', 0.1)

        vmd_alpha_history = [] 
        vmd_sigma_history = []
        
        def get_vmd_stats(m):
            raw_m = m.module if hasattr(m, 'module') else m
            if hasattr(raw_m, 'vmd_decomposer') and raw_m.vmd_decomposer is not None:
                alpha_val = raw_m.vmd_decomposer.vmd_alpha.detach().cpu()
                alpha_eff = torch.nn.functional.softplus(alpha_val) + 1e-6
                n_freq = self.args.seq_len // 2 + 1
                k_fixed = raw_m.vmd_decomposer.K
                sigma_val = n_freq / (k_fixed * alpha_eff)
                return alpha_val.numpy(), sigma_val.numpy()
            return None, None

        init_a, init_s = get_vmd_stats(self.model)
        if init_a is not None:
            vmd_alpha_history.append(init_a.copy())
            vmd_sigma_history.append(init_s.copy())

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss_list = []
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), total=len(train_loader)):
                self.model_optim.zero_grad()
                loss, outputs = self._do_same((batch_x, batch_y, batch_x_mark, batch_y_mark)) 
                
                vmd_sparsity_loss = torch.tensor(0.0, device=self.accelerator.device)
                raw_model = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(raw_model, 'get_vmd_sparsity_loss'):
                    vmd_sparsity_loss = raw_model.get_vmd_sparsity_loss()
                
                total_loss = loss + lambda_sparsity * vmd_sparsity_loss
                train_loss_list.append(total_loss.item()) 

                self.accelerator.backward(total_loss) 
                self.model_optim.step()

                if self.args.lradj == "OneCycleLR":
                    lr_scheduler.step()

            train_loss_a = np.average(train_loss_list)
            vali_loss_a = self._vali(vali_loader)
            test_loss_a = self._test(test_loader)
            
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Epoch: {epoch + 1} | Train: {train_loss_a:.7f} Vali: {vali_loss_a:.7f} test: {test_loss_a:.7f}")
                
                curr_a, curr_s = get_vmd_stats(self.model)
                if curr_a is not None:
                    vmd_alpha_history.append(curr_a.copy())
                    vmd_sigma_history.append(curr_s.copy())

            train_loss_epoch_list.append(train_loss_a)
            val_loss_epoch_list.append(vali_loss_a)
            test_loss_epoch_list.append(test_loss_a)

            early_stopping(vali_loss_a, self.model, self.args.cur_model_save_path) 
            if early_stopping.early_stop:
                self.logger.info("Early stopping triggered.")
                break

            if self.args.lradj == "adjust_fuc":
                lr_scheduler(self.model_optim, epoch + 1, self.args.learning_rate)

        if self.accelerator.is_main_process and len(vmd_alpha_history) > 0:
            save_dir = self.args.cur_model_save_path
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            
            def save_vmd_csv(data, name):
                arr = np.array(data)
                cols = [f'IMF_{k+1}' for k in range(arr.shape[1])]
                df = pd.DataFrame(arr, columns=cols)
                df.index.name = 'Epoch'
                df.to_csv(os.path.join(save_dir, name))
            
            save_vmd_csv(vmd_alpha_history, 'vmd_alpha_history.csv')
            save_vmd_csv(vmd_sigma_history, 'vmd_sigma_bandwidth_history.csv')

        return train_loss_epoch_list, val_loss_epoch_list, test_loss_epoch_list

    @torch.no_grad()
    def _vali(self, vali_loader):
        self.model.eval()
        loss_list = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader), total=len(vali_loader)):
            loss, outputs = self._do_same((batch_x, batch_y, batch_x_mark, batch_y_mark))
            loss_list.append(loss.item())
        average = np.average(loss_list)
        self.model.train()
        return average

    @torch.no_grad()
    def _test(self, test_loader):
        self.model.eval()
        loss_list = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader)):
            loss, outputs = self._do_same((batch_x, batch_y, batch_x_mark, batch_y_mark))
            loss_list.append(loss.item())
        average = np.average(loss_list)
        self.model.train()
        return average

    @torch.no_grad()
    def predict(self):
        test_data, test_loader = self._get_data(flag='test')

        # 1. 加载权重
        weight_file_path = Path(self.args.test_folder_path).parent
        checkpoint_path = os.path.join(weight_file_path, 'checkpoint')
        if not os.path.exists(checkpoint_path):
             checkpoint_path = os.path.join(self.args.checkpoints, self.args.setting, 'checkpoint.pth')
        
        if os.path.exists(checkpoint_path):
            self.logger.info(f"[Test] Loading model from: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            self.logger.warning(f"No checkpoint found. Using random weights.")

        self.model.eval()

        # 2. 准备 VMD 追踪
        vmd_module = None
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(raw_model, 'vmd_decomposer') and raw_model.vmd_decomposer is not None:
            vmd_module = raw_model.vmd_decomposer
            vmd_module.trace_spectrum = True
        
        collected_spectra = [] 

        # ================= [修改重点] 准备 LongConvMix 权重追踪 =================
        mslstm_weights_list = []
        input_power_list_denorm = [] # [NEW] 存储反归一化后的功率
        
        # 辅助函数：递归查找含有 last_weights 的 LongConvMix 模块 (更强力版本)
        def find_and_get_weights(module):
            # 使用 named_modules() 扁平化遍历所有子层，无视层级深度
            for name, m in module.named_modules():
                # 只要该模块有 last_weights 且不为 None，就是我们要找的
                if hasattr(m, 'last_weights') and m.last_weights is not None:
                    return m.last_weights
            return None

        pred_list_denormalized = []
        true_list_denormalized = []

        self.logger.info(">>> Starting inference with Multi-Scale Weight Tracking...")
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader)):
            loss, outputs = self._do_same((batch_x, batch_y, batch_x_mark, batch_y_mark))

            # --- A. 收集 VMD ---
            if vmd_module is not None and vmd_module.last_spectral_data is not None:
                batch_spectra = vmd_module.last_spectral_data['mode_spectra'].cpu().numpy()
                collected_spectra.append(batch_spectra)
            
            # --- B. 收集 LongConvMix 权重 ---
            w = find_and_get_weights(raw_model)
            if w is not None:
                # w shape: [Batch, Seq_Len, K]
                mslstm_weights_list.append(w)
                
                # [NEW] 收集并反归一化输入功率
                # 假设输入的最后一列是目标变量 Power
                
                batch_x_np = batch_x.cpu().numpy() # [B, L, C]
                B, L, C = batch_x_np.shape
                
                # 展平以便 scaler 处理: [B*L, C]
                batch_x_flat = batch_x_np.reshape(-1, C)
                
                # ================= [修改重点] =================
                # 错误写法: test_data.scaler.inverse_transform(...)
                # 正确写法: 直接调用数据集提供的 inverse_transform 方法
                batch_x_dn_flat = test_data.inverse_transform(batch_x_flat)
                # ============================================
                
                batch_x_dn = batch_x_dn_flat.reshape(B, L, C)
                
                # 提取反归一化后的最后一列 (Power)
                input_power_dn = batch_x_dn[:, :, -1]
                input_power_list_denorm.append(input_power_dn)

            # --- C. 收集预测 ---
            for _i in range(outputs.shape[0]):  
                pred_seq_norm = outputs[_i].cpu().detach().numpy()  
                true_seq_norm = batch_y[_i].cpu().detach().numpy()  
                pred_dn = test_data.inverse_transform(pred_seq_norm)
                true_dn = test_data.inverse_transform(true_seq_norm)

                if self.args.get_pred_type == 'all':
                    pred_list_denormalized.extend(pred_dn.flatten())
                    true_list_denormalized.extend(true_dn.flatten())
                else:
                    pred_list_denormalized.append(pred_dn[-1, 0])
                    true_list_denormalized.append(true_dn[-1, 0])

        # ================= [保存结果] =================
        save_dir = Path(self.args.test_folder_path)
        if not save_dir.exists(): save_dir.mkdir(parents=True)

        # 1. 保存 VMD
        if vmd_module is not None and len(collected_spectra) > 0:
            all_spectra = np.concatenate(collected_spectra, axis=0)
            freq_axis = vmd_module.last_spectral_data['freq_idx'].cpu().numpy()
            np.save(save_dir / 'test_set_all_modes_spectra.npy', all_spectra)
            np.save(save_dir / 'test_set_freq_axis.npy', freq_axis)
            vmd_module.trace_spectrum = False

        # 2. 保存 Multi-Scale Weights 并生成 Case Study
        if len(mslstm_weights_list) > 0:
            # 拼接
            all_weights = np.concatenate(mslstm_weights_list, axis=0) # [Total_Samples, Seq_Len, 3]
            all_powers_dn = np.concatenate(input_power_list_denorm, axis=0) # [Total_Samples, Seq_Len]
            
            # [NEW] 保存反归一化后的功率数据
            np.save(save_dir / 'mslstm_gate_weights.npy', all_weights)
            np.save(save_dir / 'mslstm_input_power.npy', all_powers_dn)
            self.logger.info(f">>> Weights & Power saved. Power shape: {all_powers_dn.shape}")
            
            # 自动找最大方差样本 (Case Study)
            # 使用反归一化后的功率计算方差也是没问题的，或者用权重方差
            # 这里用功率波动最大的样本
            variances = np.var(all_powers_dn, axis=1)
            high_var_idx = np.argmax(variances)
            
            case_weights = all_weights[high_var_idx] # [Seq_Len, 3]
            case_power = all_powers_dn[high_var_idx] # [Seq_Len]
            
            # [NEW] 保存 CSV，包含样本索引，方便定位
            df_case = pd.DataFrame({
                'Sample_Index': high_var_idx, # 记录这是测试集第几个样本
                'Time_Step': range(len(case_power)),
                'Power_Denorm': case_power, # 反归一化的真实功率
                'Weight_Small_k7': case_weights[:, 0],
                'Weight_Medium_k15': case_weights[:, 1],
                # 'Weight_Large_k25': case_weights[:, 2]
            })
            csv_path = save_dir / 'case_study_mslstm.csv'
            df_case.to_csv(csv_path, index=False)
            self.logger.info(f">>> [Auto-Case Study] Sample {high_var_idx} saved to: {csv_path}")
        else:
            self.logger.error("No MSLSTM weights collected. Check model configuration.")

        return pred_list_denormalized, true_list_denormalized, [], []