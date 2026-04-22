from pathlib import Path

import torch
import numpy as np
import random
from datetime import datetime
import os
import shutil
from exp.exp_ultra_short_term_forecasting import Exp_Ultra_Short_Term_Forecast
import argparse
import yaml
import utils.plot_tools as plt
import ast
import json
from utils.run_tools import save_pred_csv
from utils.calculate_tools import calculate_daily_metrics, print_daily_metrics
import os

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

# 公共参数 模型训练相关
parser.add_argument("--task_name", type=str, default="ultra_short_term_forecast")  # 任务名称
parser.add_argument("--seq_len", type=int, default=96, required=True)  # 输入序列时间步
parser.add_argument("--pred_len", type=int, default=8, required=True)  # 预测时间步
parser.add_argument("--label_len", type=int, default=0, required=True)  #
parser.add_argument("--dropout", type=float, default=0.1, required=True)  # 丢失学习率
parser.add_argument("--freq", type=str, default="60min", required=True)  # 数据分辨率
parser.add_argument('--lradj', type=str, required=True)  # 学习率调整方式
parser.add_argument('--learning_rate', type=float, required=True)  # 学习率
parser.add_argument('--patience', type=int, default=10, required=True)  # 早停触发的训练轮数
parser.add_argument('--batch_size', type=int, required=True)  # 训练批次大小
parser.add_argument('--train_epochs', type=int, required=True)  # 训练轮数
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
parser.add_argument("--patch_len", type=int, default=16)
# SolarTimeLLM/TimeLLM/PatchTST
parser.add_argument("--stride", type=int, default=8)
# Informer/Autoformer
parser.add_argument("--factor", type=int, default=1)  # attn factor
# SolarTimeLLM/TimeLLM/TimeXer/PatchTST/iTransformer/Informer/Transformer/Autoformer
parser.add_argument("--n_heads", type=int, default=8)
# SolarTimeLLM/TimeLLM/TimeXer/PatchTST/iTransformer/TimesNet/LLMMixer/Informer/Transformer/Autoformer
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--d_ff", type=int, default=128)
# TimeXer/PatchTST/iTransformer/TimesNet/LLMMixer/Informer/Transformer/Autoformer
parser.add_argument("--enc_in", type=int, default=1)  # n_vars,也就是模型输入特征
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
parser.add_argument("--moving_avg", type=int, default=25)  # moving_avg类用于提取时间序列的趋势成分,主要是AvgPool1d的池化核大小
# TimeLLM/SolarTimeLLM/LLMMixer
parser.add_argument("--llm_layers", type=int, default=32)
parser.add_argument("--llm_name", type=str, default="BERT")
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
parser.add_argument("--features", type=str, default="S")  # S为预测单变量
"""
    TimesNet参数
"""
parser.add_argument("--top_k", type=int, default=5)  # FFT相关度前5
parser.add_argument("--num_kernels", type=int, default=6)  # Inception核心数量
"""
    DLinear参数
"""
parser.add_argument("--individual", type=bool, default=False)  # False为预测单变量
parser.add_argument("--n_input_features", type=int, default=13)  # 模型输入特征数量
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
# 读取YAML文件
# with open(f'./models/run_config/{args.model_name}.yml', 'r', encoding='utf-8') as file:
#     model_args = yaml.safe_load(file)
#
# # 合并属性
# for key, value in model_args.items():
#     setattr(args, key, value)

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
    pred_list, true_list = exp.predict()
    # 打印pred_list形状
    # print(f"pred_list: {pred_list}")

    # 保存预测结果
    # 1. 把此次测试的文件夹路径 + 数据文件
    test_file_path = os.path.join(test_folder_path, Path(args.data_path_list[2]).name)
    # 2. 调用工具
    pred_file_path = save_pred_csv(args.is_set_zero, args.get_pred_type, args.model_name, test_file_path,
                                   [args.seq_len, args.pred_len], pred_list)
    plt.plot_power_comparison(pred_file_path)

    # 计算指标
    daily_metrics_df, overall_metrics = calculate_daily_metrics(pred_file_path)
    # 1.使用日志记录每天的详细指标
    print_daily_metrics(daily_metrics_df, overall_metrics)
    # 2.保存每日指标到文件
    metrics_file_path = os.path.join(test_folder_path, f"{args.model_name}_daily_metrics.csv")
    daily_metrics_df.to_csv(metrics_file_path, index=False)
