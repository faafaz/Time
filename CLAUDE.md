# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个时间序列预测项目, 专注于光伏功率预测(PVPOF), 包含超短期、短期、中期和长期预测。项目使用PyTorch框架, 支持多种时间序列模型。

## 核心架构

- **模型层**: `models/` 包含多种时间序列模型 (DLinear, Transformer, TimeXer, TimesNet, PatchTST, iTransformer等)
- **实验层**: `exp/` 包含实验配置和训练逻辑
- **数据层**: `data_provider/` 数据处理和加载器
- **工具层**: `utils/` 包含各种工具函数
- **数据库**: `database/` 数据预处理和数据库操作

## 常用命令

### 单次运行
```bash
# 训练模式
python run.py --run_type 0 --model_name DLinear --data_path_list "['datasets/train_dataset.csv','datasets/test_dataset.csv','datasets/val_dataset.csv']" --seq_len 96 --pred_len 16 --label_len 0 --dropout 0.1 --freq 60min --lradj adjust_fuc --learning_rate 0.0001 --patience 10 --batch_size 128 --train_epochs 100

# 测试模式
python run.py --run_type 1 --model_name DLinear --weight_foldername "20250914_225208_DLinear" --data_path_list "['datasets/train_dataset.csv','datasets/test_dataset.csv','datasets/val_dataset.csv']" --seq_len 96 --pred_len 16 --label_len 0 --dropout 0.1 --freq 60min --lradj adjust_fuc --learning_rate 0.0001 --patience 10 --batch_size 128 --train_epochs 100
```

### 批量运行
```bash
python run_batch.py
```

### 使用Accelerate (多GPU训练)
```bash
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 8 --main_process_port 00097 run.py [参数]
```

## 模型配置

模型参数配置文件位于 `models/run_config/`, 每个模型有对应的YAML文件:
- `DLinear.yml`, `Transformer.yml`, `TimeXer.yml` 等

## 数据格式

- 训练数据: `datasets/train_dataset.csv`
- 测试数据: `datasets/test_dataset.csv`
- 验证数据: `datasets/val_dataset.csv`

## 关键参数

- `seq_len`: 输入序列长度 (默认96)
- `pred_len`: 预测长度 (默认16)
- `label_len`: 标签长度 (默认0)
- `freq`: 数据频率 (60min/30min/15min)
- `run_type`: 运行类型 (0=训练, 1=测试)

## 输出结果

训练和测试结果保存在 `checkpoints/` 目录:
- 训练权重和日志
- 测试预测结果和指标文件
- 最终结果汇总到 `A_result/` 目录

## 依赖环境

核心依赖:
- PyTorch 2.5.1
- Accelerate 1.10.1
- NumPy 1.26.4
- 其他标准科学计算库

## 开发提示

1. 新模型应在 `models/` 目录实现
2. 实验逻辑在 `exp/exp_ultra_short_term_forecasting.py`
3. 数据预处理在 `database/` 和 `data_provider/`
4. 使用 `run_batch.py` 进行批量实验
5. 结果分析使用 `utils/` 中的工具函数