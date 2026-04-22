from pathlib import Path

import torch
import numpy as np
import random
from datetime import datetime
import os
import shutil
from exp.exp_ultra_short_term_forecasting import Exp_Ultra_Short_Term_Forecast
import argparse
import utils.plot_tools as plt
import ast
import json
from utils.run_tools import save_pred_csv
from utils.calculate_tools import (
    calculate_global_metrics_from_arrays,
    print_global_metrics,
    calculate_state_grid_accuracy
)
import os

# ----------------------------
# HPO (Optuna) helpers (lazy import)
# ----------------------------

def _lazy_import_optuna():
    try:
        import optuna  # type: ignore
        from optuna.samplers import TPESampler  # type: ignore
        from optuna.pruners import SuccessiveHalvingPruner  # type: ignore
        return optuna, TPESampler, SuccessiveHalvingPruner
    except Exception as e:
        raise ImportError(
            "未安装 Optuna。请先安装后再使用 --use_vmd_hpo 开关: pip install optuna"
        ) from e


def _evaluate_val_denorm(exp, args):
    """Evaluate on validation loader using denormalized predictions and targets.
    Returns (rmse, stategrid_acc).
    """
    import numpy as _np
    # Load best checkpoint if exists

    ckpt_path = os.path.join(args.cur_model_save_path, 'checkpoint')
    if os.path.exists(ckpt_path):
        exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))
    exp.model.eval()
    # Build val loader
    val_set, val_loader = exp._get_data(flag='val')
    pred_denorm, true_denorm = [], []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
            batch_x = batch_x.float().to(exp.accelerator.device)
            batch_y = batch_y.float().to(exp.accelerator.device)
            batch_x_mark = batch_x_mark.float().to(exp.accelerator.device)
            batch_y_mark = batch_y_mark.float().to(exp.accelerator.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.accelerator.device)
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # denorm output
            # to numpy
            out_np = outputs.detach().cpu().numpy()  # [B, pred_len, 1]
            y_np = batch_y.detach().cpu().numpy()     # normalized
            # inverse transform true
            # flatten per get_pred_type
            B = out_np.shape[0]
            for b in range(B):
                pred_seq = out_np[b]  # [pred_len, 1], already denorm
                print(pred_seq.size)
                true_seq_norm = y_np[b]  # [pred_len, 1]
                true_seq = val_set.inverse_transform(true_seq_norm)
                if args.get_pred_type == 'last':
                    pred_denorm.append(pred_seq[-1, 0])
                    true_denorm.append(true_seq[-1, 0])
                elif args.get_pred_type == 'first':
                    pred_denorm.append(pred_seq[0, 0])
                    true_denorm.append(true_seq[0, 0])
                else:  # all
                    pred_denorm.extend(pred_seq.flatten())
                    true_denorm.extend(true_seq.flatten())
    # compute metrics
    metrics = calculate_global_metrics_from_arrays(true_denorm, pred_denorm)
    sg = calculate_state_grid_accuracy(true_denorm, pred_denorm, capacity=args.capacity)
    rmse = float(metrics.get('RMSE', _np.nan))
    acc = float(sg.get('StateGrid_Accuracy', 0.0))
    return rmse, acc


def run_vmd_hpo(args):
    """Run Optuna TPE-based HPO for VMD hyperparameters on the Preprocessed model.
    Searches over vmd_k (categorical) and vmd_alpha (log-uniform). Trains with a
    reduced epoch budget per trial, evaluates validation RMSE/StateGrid accuracy
    on denormalized scale, and optimizes a scalarized objective.
    """
    optuna, TPESampler, SuccessiveHalvingPruner = _lazy_import_optuna()

    hpo_dir = os.path.join(args.cur_model_save_path, 'hpo_trials')
    os.makedirs(hpo_dir, exist_ok=True)

    sampler = TPESampler(seed=2021)
    # Note: Without in-loop reporting, ASHA will behave like no-prune. Kept here for future extension.
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)

    def objective(trial):
        # search space
        vmd_k = trial.suggest_categorical('vmd_k', [4, 6, 8, 12, 16])
        vmd_alpha = trial.suggest_float('vmd_alpha', 1, 4, log=True)
        # prepare per-trial args and folder
        trial_dir = os.path.join(hpo_dir, f"trial_{trial.number:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        trial_args = argparse.Namespace(**vars(args))
        trial_args.cur_model_save_path = trial_dir
        trial_args.vmd_k = int(vmd_k)
        trial_args.vmd_alpha = float(vmd_alpha)
        # Limit epochs for HPO
        max_epochs = getattr(args, 'hpo_max_epochs', 15)
        trial_args.train_epochs = int(max_epochs)
        # Train
        exp = Exp_Ultra_Short_Term_Forecast(trial_args)
        try:
            exp.train()
            # Evaluate on validation set with denormalized metrics
            rmse, acc = _evaluate_val_denorm(exp, trial_args)
        finally:
            # cleanup accelerator/GPU cache per trial
            try:
                if args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception:
                pass
            del exp
        # Scalarized objective: lower is better
        objective_value = rmse - 0.1 * (acc / 100.0)
        return objective_value

    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=getattr(args, 'hpo_trials', 40), show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value
    # Persist best
    with open(os.path.join(hpo_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
        json.dump({'best_params': best_params, 'best_objective': best_value}, f, ensure_ascii=False, indent=2)
    return best_params, best_value

"""
根据时间范围，PVPOF分为超短期、短期、中期和长期4种类型。
随后，超短期涵盖秒到一小时[16]，短期跨越小时到一天，中期延伸到一个月，持久预测解决在一个月到一年[22]左右。
"""

# 保证实验的可重复性
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser()

# 公共参数 模型启动相关
parser.add_argument("--run_type", type=int, default=0, required=True)  # 0/1 train/test
parser.add_argument("--model_name", type=str, default="TEST", required=True)  # 模型名称
parser.add_argument("--weight_foldername", type=str)
parser.add_argument("--data_path_list", type=ast.literal_eval, default=[""], required=True)
parser.add_argument('--get_pred_type', type=str, default="first", required=True)
parser.add_argument('--is_set_zero', action='store_true')

parser.add_argument("--capacity", type=float, default=80, help="Rated capacity (same unit as power).")


# 公共参数 模型训练相关
parser.add_argument("--task_name", type=str, default="ultra_short_term_forecast")  # 任务名称
parser.add_argument("--seq_len", type=int, default=720, required=True)  # 输入序列时间步
parser.add_argument("--pred_len", type=int, default=16, required=True)  # 预测时间步
parser.add_argument("--label_len", type=int, default=0, required=True)  #
parser.add_argument("--dropout", type=float, default=0.3, required=True)  # 丢失学习率
parser.add_argument("--freq", type=str, default="15min", required=True)  # 数据分辨率
parser.add_argument('--lradj', type=str, required=True)  # 学习率调整方式
parser.add_argument('--learning_rate', type=float, required=True)  # 学习率
parser.add_argument('--patience', type=int, default=10, required=True)  # 早停触发的训练轮数
parser.add_argument('--batch_size', type=int, required=True)  # 训练批次大小
parser.add_argument('--train_epochs', type=int, required=True)  # 训练轮数
# 根据设备判断获取gpu还是cpu
# parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# 训练设备相关
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument("--use_gpu", default=True)
parser.add_argument("--gpu_type", default='cuda')
# 多GPU训练设备相关
parser.add_argument("--use_multi_gpu", default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU显卡设置')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

"""
    模型公共结构参数
"""
# SolarTimeLLM/TimeLLM/TimeXer/PatchTST
parser.add_argument("--patch_len", type=int, default=24)
# SolarTimeLLM/TimeLLM/PatchTST
parser.add_argument("--stride", type=int, default=8)
# Informer/Autoformer
# VMD/HPO switches and params
parser.add_argument("--use_vmd_hpo", action='store_true', default=False,
                    help="Enable Optuna TPE HPO for VMD (only for iTransformer_xLSTM_VMD_Preprocessed). Default: off")
parser.add_argument("--hpo_trials", type=int, default=40, help="Number of HPO trials")
parser.add_argument("--hpo_max_epochs", type=int, default=50, help="Epochs per trial during HPO")
# VMD params (used when HPO disabled)
parser.add_argument("--vmd_k", type=int, default=8, help="VMD modes K (used when HPO disabled)")
parser.add_argument("--vmd_alpha", type=float, default=0.54132, help="VMD alpha initial value (used when HPO disabled)")#0.54132
parser.add_argument("--vmd_impl", type=str, default="vmdpy", help="VMD implementation: fftbank|auto|vmdpy")

parser.add_argument("--factor", type=int, default=1)  # attn factor
# SolarTimeLLM/TimeLLM/TimeXer/PatchTST/iTransformer/Informer/Transformer/Autoformer
parser.add_argument("--n_heads", type=int, default=8)
# SolarTimeLLM/TimeLLM/TimeXer/PatchTST/iTransformer/TimesNet/LLMMixer/Informer/Transformer/Autoformer
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--d_ff", type=int, default=64)
# TimeXer/PatchTST/iTransformer/TimesNet/LLMMixer/Informer/Transformer/Autoformer
parser.add_argument("--enc_in", type=int, default=5)# n_vars,也就是模型输入特征
parser.add_argument("--e_layers", type=int, default=1)  # encoder层数
# Informer/Transformer/Autoformer
parser.add_argument("--dec_in", type=int, default=1)  # decoder解码器输入特征
parser.add_argument("--d_layers", type=int, default=1)  # encoder层数
# TimesNet/LLMMixer/Transformer/Autoformer
parser.add_argument("--c_out", type=int, default=1)  # 模型预测的特征数
# TimeXer/PatchTST/iTransformer/Informer/Transformer/Autoformer
parser.add_argument("--activation", type=str, default="gelu")  # 激活函数
# TimesNet/LLMMixer/Informer/Transformer
# 时间嵌入方式,虽然/TimeXer/PatchTST/iTransformer有这个参数,但是都没有使用到,就删掉了
parser.add_argument("--embed_type", type=str, default="timeF")
# DLinear/LLMMixer/Autoformer
parser.add_argument("--moving_avg", type=int, default=11)  # moving_avg类用于提取时间序列的趋势成分,主要是AvgPool1d的池化核大小
# TimeLLM/SolarTimeLLM/LLMMixer
parser.add_argument("--llm_layers", type=int, default=32)
parser.add_argument("--llm_name", type=str, default="BERT")
parser.add_argument("--vmd_sparsity_lambda", type=float, default=0.05)
"""
    TimeLLM与SolarTimeLLM参数
"""
parser.add_argument("--d_llm", type=int, default=768)
parser.add_argument("--prompt_func", type=str, default="get_simple_prompt")  # get_simple_prompt/get_calculate_prompt
parser.add_argument("--num_tokens", type=int, default=100)
"""
    LLMMixer参数
"""
parser.add_argument("--channel_independence", type=int, default=1)  # 采样方法
parser.add_argument("--down_sampling_method", type=str, default="avg")  # avg/max/conv
parser.add_argument("--use_future_temporal_feature", type=int, default=0)  # 是否使用未来时间特征
parser.add_argument("--down_sampling_window", type=int, default=2)  # 采样率
parser.add_argument("--down_sampling_layers", type=int, default=3)  # 采样层数
"""
    TimeXer参数
"""
parser.add_argument("--features", type=str, default="MS")  # S为预测单变量

"""
    DUET参数
"""
parser.add_argument("--CI", type=int, default=0)  # 通道独立性 (1=启用, 0=禁用)
parser.add_argument("--num_experts", type=int, default=4)  # 专家数量
parser.add_argument("--k", type=int, default=2)  # 聚类数量
parser.add_argument("--fc_dropout", type=float, default=0.2)  # 全连接层dropout
parser.add_argument("--loss", type=str, default="RMSE")  # 损失函数 (/RMSE/MAE/huber/dilate/AntiLagRampLoss)
"""
    TimesNet参数
"""
parser.add_argument("--top_k", type=int, default=5)  # FFT相关度前5
parser.add_argument("--num_kernels", type=int, default=6)  # Inception核心数量
"""
    DLinear参数
"""
parser.add_argument("--individual", type=bool, default=False)  # False为预测单变量
parser.add_argument("--n_input_features", type=int, default=2)  # 模型输入特征数量
parser.add_argument("--output_denorm", type=bool, default=True)
parser.add_argument("--meta_cols", type=int, default=0)  # 元数据列数
parser.add_argument("--enable_two_stage_pred", type=bool, default=True)  # 是否启用两阶段预测

"""
DILATE 参数
"""
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--gamma", type=float, default=0.001)

# """
#     DLinear_Graph / DLinear_PreGraph 图模块参数
# """
# parser.add_argument("--use_graph", type=bool, default=True)  # 是否使用图模块
# parser.add_argument("--graph_conv_channel", type=int, default=16)  # 图卷积通道数
# parser.add_argument("--graph_skip_channel", type=int, default=32)  # 图跳跃连接通道数
# parser.add_argument("--gcn_depth", type=int, default=2)  # 图卷积深度
# parser.add_argument("--graph_node_dim", type=int, default=64)  # 节点嵌入维度
# parser.add_argument("--graph_propalpha", type=float, default=0.05)  # 图传播混合系数
# parser.add_argument("--enable_time_adj_gate", type=bool, default=True)  # 是否启用时间感知邻接门控
# parser.add_argument("--time_gate_scale", type=float, default=0.1)  # 时间门控缩放系数
# parser.add_argument("--time_d_model", type=int, default=32)  # 时间嵌入维度
# parser.add_argument("--time_hidden", type=int, default=64)  # 时间MLP隐藏维度

# """
#     DLinear_Graph / DLinear_PreGraph VMD分解参数
# """
# parser.add_argument("--enable_vmd_preprocessing", type=bool, default=True)  # 是否启用VMD预处理
# parser.add_argument("--vmd_K", type=int, default=3)  # VMD分解的模态数（注意大写K）
# parser.add_argument("--vmd_tau", type=float, default=0.0)  # VMD噪声容忍度
# parser.add_argument("--vmd_DC", type=int, default=0)  # VMD是否包含DC分量
# parser.add_argument("--vmd_init", type=int, default=1)  # VMD初始化方式
# parser.add_argument("--vmd_tol", type=float, default=1e-7)  # VMD收敛容忍度

# """
#     DLinear_Graph / DLinear_PreGraph LongConvMix参数
# """
# parser.add_argument("--enable_target_longconv", type=bool, default=True)  # 是否对目标变量使用LongConvMix
# parser.add_argument("--longconv_kernels", type=str, default="(5, 11, 23)")  # LongConvMix卷积核大小（字符串形式）
# parser.add_argument("--longconv_dropout", type=float, default=0.1)  # LongConvMix dropout率
# parser.add_argument("--longconv_hidden", type=int, default=32)  # LongConvMix隐藏维度

# S-MoLE router parameters
parser.add_argument("--router_num_levels", type=int, default=3)
parser.add_argument("--router_mode", type=str, default="hard")

parser.add_argument("--router_hidden", type=int, default=256)
parser.add_argument("--router_balance_lambda", type=float, default=0.01)
parser.add_argument("--router_temperature", type=float, default=1.0)
parser.add_argument("--router_dropout", type=float, default=0.0)


"""
    Informer参数
"""
parser.add_argument("--distil", type=bool, default=True)  # 蒸馏（distilling）技术
"""
    Autoformer参数
"""
parser.add_argument("--enc_embedding", type=str, default="DataEmbedding_wo_pos")  # encoder编码器嵌入类
parser.add_argument("--dec_embedding", type=str, default="DataEmbedding_wo_pos_decoder")  # decoder解码器嵌入类

args = parser.parse_args()

# 处理 longconv_kernels 参数（从字符串转换为元组）
if hasattr(args, 'longconv_kernels') and isinstance(args.longconv_kernels, str):
    try:
        args.longconv_kernels = ast.literal_eval(args.longconv_kernels)
    except:
        args.longconv_kernels = (5, 11, 23)  # 默认值

# 读取YAML文件
# with open(f'./models/run_config/{args.model_name}.yml', 'r', encoding='utf-8') as file:
#     model_args = yaml.safe_load(file)
#
# # 合并属性
# for key, value in model_args.items():
#     setattr(args, key, value)

def _profile_model_performance(exp, args):
    """
    计算模型的参数量(Params)、计算复杂度(FLOPs)和单次推理延迟(Latency)
    """
    try:
        from thop import profile, clever_format
    except ImportError:
        print("\n[警告] 未安装 thop，跳过 FLOPs 和 Params 计算。请运行: pip install thop")
        return

    print("\n" + "="*80)
    print("开始测算模型参数量、计算复杂度 (FLOPs) 与 推理延迟...")
    print("="*80)
    
    # 1. 准备数据
    exp.model.eval()
    test_set, test_loader = exp._get_data(flag='test')
    
    # 获取一个真实的 batch，取第一条数据 (Batch_Size = 1)
    batch_iterator = iter(test_loader)
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(batch_iterator)
    
    device = exp.accelerator.device
    batch_x = batch_x[0:1].float().to(device)
    batch_y = batch_y[0:1].float().to(device)
    batch_x_mark = batch_x_mark[0:1].float().to(device)
    batch_y_mark = batch_y_mark[0:1].float().to(device)
    
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
    
    inputs_tuple = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    # ---------------------------------------------------------
    # 第一阶段：在真实的设备 (GPU) 上测算推理延迟 (Latency)
    # ---------------------------------------------------------
    import time
    with torch.no_grad():
        # GPU 预热 (Warm-up)
        for _ in range(10):
            _ = exp.model(*inputs_tuple)
        
        if args.gpu_type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        run_times = 50  # 跑50次取平均
        for _ in range(run_times):
            _ = exp.model(*inputs_tuple)
            
        if args.gpu_type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_latency = ((end_time - start_time) / run_times) * 1000
        print(f"[*] 平均单次推理延迟 (Latency) : {avg_latency:.2f} ms")

    # ---------------------------------------------------------
    # 第二阶段：转移到 CPU 测算 FLOPs 和 Params (彻底解决 thop 报错)
    # ---------------------------------------------------------
    print("[*] 正在分析模型结构 (计算 FLOPs & Params)...")
    with torch.no_grad():
        # 剥离 accelerate 的分布式包装，并将原始模型移动到 CPU
        base_model = exp.accelerator.unwrap_model(exp.model).cpu()
        
        # 将输入张量也移动到 CPU
        cpu_inputs_tuple = (
            batch_x.cpu(), 
            batch_x_mark.cpu(), 
            dec_inp.cpu(), 
            batch_y_mark.cpu()
        )
        
        # 运行 thop 分析
        flops, params = profile(base_model, inputs=cpu_inputs_tuple, verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        
        print(f"[*] 模型总参数量 (Parameters): {params_str}")
        print(f"[*] 单次推理复杂度 (FLOPs)  : {flops_str}")
        print("="*80 + "\n")


# 时序任务类型
Exp = Exp_Ultra_Short_Term_Forecast

if args.run_type == 0:  # 训练
    # 权重根目录下哪个模型文件夹
    model_save_path = os.path.join("./checkpoints", args.model_name)
    # 此次训练的权重路径
    weight_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.model_name}'
    cur_model_save_path = os.path.join(model_save_path, weight_name)
    os.makedirs(cur_model_save_path)

    # 把数据集拷贝进去
    shutil.copy(args.data_path_list[0], cur_model_save_path)
    # 备份训练参数到文件
    with open(os.path.join(cur_model_save_path, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    args.cur_model_save_path = cur_model_save_path

    # If enabled, run VMD HPO for the Preprocessed model, then final full training with best params
    if args.use_vmd_hpo and args.model_name == 'iTransformer_xLSTM_VMD_Preprocessed':
        print("\n================ HPO: VMD (Optuna TPE) ================")
        print("模型: iTransformer_xLSTM_VMD_Preprocessed  | 开关: 已开启 --use_vmd_hpo")
        try:
            best_params, best_value = run_vmd_hpo(args)
        except ImportError as e:
            print(str(e))
            raise
        print(f"HPO完成: best_obj={best_value:.6f}, best_params={best_params}")
        # 最终全预算训练
        final_dir = os.path.join(cur_model_save_path, 'final_best')
        os.makedirs(final_dir, exist_ok=True)
        # 应用最优参数
        if 'vmd_k' in best_params:
            setattr(args, 'vmd_k', int(best_params['vmd_k']))
        if 'vmd_alpha' in best_params:
            setattr(args, 'vmd_alpha', float(best_params['vmd_alpha']))
        args.cur_model_save_path = final_dir
        with open(os.path.join(final_dir, "args_best.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        exp = Exp(args)
        train_loss_epoch_list, val_loss_epoch_list, test_loss_epoch_list = exp.train()
        plt.plot_loss(train_loss_epoch_list, val_loss_epoch_list, test_loss_epoch_list, final_dir)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        train_loss_epoch_list, val_loss_epoch_list, test_loss_epoch_list = exp.train()
        # 画训练损失图
        plt.plot_loss(train_loss_epoch_list, val_loss_epoch_list, test_loss_epoch_list, cur_model_save_path)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
else:
    # 创建此次测试的文件夹保存测试相关结果和过程
    # 1.创建保存模型的路径
    model_save_path = os.path.join("./checkpoints", args.model_name, args.weight_foldername)

    # 特殊处理：如果是GAN模型，路径在 DLinear_Graph_GAN 目录下
    if not os.path.exists(model_save_path) and "GAN" in args.weight_foldername:
        # 尝试在 {model_name}_GAN 目录下查找
        gan_model_save_path = os.path.join("./checkpoints", f"{args.model_name}_GAN", args.weight_foldername)
        if os.path.exists(gan_model_save_path):
            model_save_path = gan_model_save_path
            print(f"检测到GAN模型，使用路径: {model_save_path}")

    if not os.path.exists(model_save_path):
        raise FileExistsError(f"权重文件夹不存在:{model_save_path}")

    # 2.创建此次测试的文件夹路径
    test_folder_path = os.path.join(model_save_path, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_test')
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)
    # 3.把训练集拷贝进去
    shutil.copy(args.data_path_list[2], test_folder_path)
    # 4.备份测试参数到文件
    with open(os.path.join(test_folder_path, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # 模型预测
    args.test_folder_path = test_folder_path
    exp = Exp(args)  # set experiments


    # 返回格式: (pred_denormalized, true_denormalized, pred_normalized, true_normalized)
    pred_list_denorm, true_list_denorm, pred_list_norm, true_list_norm = exp.predict()

    # 保存预测结果（使用反归一化数据）
    # 1. 把此次测试的文件夹路径 + 数据文件
    test_file_path = os.path.join(test_folder_path, Path(args.data_path_list[2]).name)
    # 2. 调用工具（保存反归一化数据）
    pred_file_path = save_pred_csv(args.is_set_zero, args.get_pred_type, args.model_name, test_file_path,
                                   [args.seq_len, args.pred_len], pred_list_denorm)
    plt.plot_power_comparison(pred_file_path)

    # 计算整体指标（使用实际功率/反归一化数据）
    print(f"预测文件路径: {pred_file_path}")
    print("\n" + "="*80)
    print("使用实际功率（反归一化数据）计算整体指标")
    print("="*80)
   
    metrics = calculate_global_metrics_from_arrays(true_list_denorm, pred_list_denorm)

    # 计算国家电网准确率（使用反归一化数据 + 容量法）
    state_metrics = calculate_state_grid_accuracy(true_list_denorm, pred_list_denorm, capacity=args.capacity)
    # 打印国家电网准确率

    # 合并所有指标
    all_metrics = {**metrics, **state_metrics}

    print_global_metrics(all_metrics)

    # 保存全局指标到JSON文件
    metrics_file_path = os.path.join(test_folder_path, f"{args.model_name}_global_metrics.json")
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n全局指标已保存到: {metrics_file_path}")

    
    _profile_model_performance(exp, args)




