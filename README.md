# 风电功率时间序列预测框架

集成多种先进时间序列预测模型，用于风电功率超短期/短期/中期/长期预测研究。

## 项目结构

```
.
├── third_party/          # 第三方依赖源码
│   ├── DILATE_master/   # DILATE时间对齐损失
│   ├── NFM-main/        # Neural Frequency Modeling
│   ├── pysdtw_main/     # Soft-DTW GPU加速
│   └── pytorch_wavelets/ # PyTorch小波变换
├── data_provider/       # 数据加载和预处理
├── database/           # 原始数据处理脚本
├── dataset/            # 数据集 (建议放在项目外部，这里仅gitignore)
├── exp/                # 实验流程 (训练/测试)
├── layers/             # 共享神经网络层组件
├── models/             # 所有模型定义
│   └── run_config/     # YAML配置文件
├── utils/              # 工具函数
├── scripts/            # 运行脚本
├── A_result/           # 预测结果输出
├── checkpoints*/       # 训练检查点 (gitignore)
├── run.py              # 主运行入口
└── run_batch.py        # 批量运行
```

## 已实现模型

| 模型 | 文件 | 说明 |
|------|------|------|
| Transformer | `Transformer.py` |  vanilla Transformer |
| DLinear | `DLinear.py` | 基线DLinear模型 |
| DLinear_Graph | `DLinear_Graph.py` | DLinear + 图结构 |
| DLinear_ABDM | `DLinear_ABDM.py` | DLinear + 自适应分支分解模块 |
| Autoformer | `Autoformer.py` | Autoformer |
| Informer | `Informer.py` | Informer |
| TimesNet | `TimesNet.py` | TimesNet |
| PatchTST | `PatchTST.py` | PatchTST |
| iTransformer | `iTransformer.py` | 标准iTransformer |
| iTransformer_xLSTM | `iTransformer_xLSTM.py` | iTransformer + xLSTM时序建模 |
| iTransformer_xLSTM_VMD | `iTransformer_xLSTM_VMD.py` | iTransformer_xLSTM + VMD双分支融合 |
| **i_transformer_xlstm_vmd_pre** | `i_transformer_xlstm_vmd_pre.py` | **统一VMD预处理版本** - 支持标准VMD/自适应稀疏VMD，可通过配置切换 |
| itransformer_vmd_refinement | `itransformer_vmd_refinement.py` | VMD分解 + 未来风速修正 |
| DUET | `DUET.py` | 双聚类增强时间序列预测 |
| FAN | `fan.py` | Frequency Attention Network |
| LightTS | `LightTS.py` | LightTS |
| LLMMixer | `LLMMixer.py` | LLM + Mixer |
| S_MoLE | `S_MoLE.py` | S_MoLE模型 |
| SolarTimeLLM | `SolarTimeLLM.py` | 太阳能LLM模型 |
| TimeCMA | `TimeCMA.py` | TimeCMA模型 |
| TimeLLM | `TimeLLM.py` | TimeLLM |
| TimeLLM_MSPF_GATEFUSION | `TimeLLM_MSPF_GATEFUSION.py` | TimeLLM + MSPF门融合 |
| TimeXer | `TimeXer.py` | TimeXer |
| WPMixer | `WPMixer.py` | 小波PatchMixer |
| nfm_module | `nfm_module.py` | NFM模块 |
| xlstm_modules | `xlstm_modules.py` | xLSTM核心模块 (sLSTM实现) |
| feature_selection | `feature_selection.py` | 特征选择模块 |
| vmd_decomposer | `vmd_decomposer.py` | **统一VMD分解模块** - 支持StandardVMD/SparseAdaptiveVMD |

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 训练

```bash
# 训练模式: run_type 0
python run.py --run_type 0 --model_name DLinear --seq_len 96 --pred_len 16
```

### 测试

```bash
# 测试模式: run_type 1
python run.py --run_type 1 --model_name DLinear --weight_foldername "[path_to_checkpoints]"
```

### 批量运行

```bash
python run_batch.py
```

## 关键配置参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `seq_len` | 输入序列长度 | - |
| `pred_len` | 预测长度 | - |
| `batch_size` | 批次大小 | - |
| `learning_rate` | 学习率 | - |
| `num_epochs` | 训练轮数 | - |

### VMD模型配置 (i_transformer_xlstm_vmd_pre)

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `enable_vmd_preprocessing` | 是否启用VMD预处理 | True |
| `vmd_mode` | `standard`/`sparse_adaptive` | `standard` |
| `vmd_k` | 模态数量 (standard) / 最大模态K_max (sparse_adaptive) | 8 |
| `vmd_alpha` | 初始带宽参数 | 2000.0 |
| `xlstm_layers` | xLSTM层数 | 2 |
| `rnn_type` | `gru`/`slstm` | `slstm` |
| `mask_high_freq` | 消融实验-屏蔽高频 | False |
| `mask_low_freq` | 消融实验-屏蔽低频 | False |
| `mask_mid_freq` | 消融实验-屏蔽中频 | False |

## 项目重构说明

本次重构主要改进：

1. **消除代码重复**:
   - 提取统一 `vmd_decomposer.py` - 消除了5个文件中的重复VMD类定义
   - 合并 `iTransformer_xLSTM_VMD_Preprocessed*` 5个变种为单个统一文件，通过配置切换变体
   - 删除重复的 `Transformer_EncDec1.py` 和 `SelfAttention_Family1.py`，使用功能完整的原始文件
   - 删除了 `run_config/` 中重复放置的模型代码文件

2. **整理第三方库**:
   - 创建 `third_party/` 统一目录存放所有内嵌第三方库
   - 更新所有导入路径
   - 根目录更干净

3. **统一命名规范**:
   - 所有模型文件放在 `models/` 目录
   - 遵循蛇形命名规范 (`fan.py` 代替 `FAN.py`)
   - `run_config/` 只存放YAML配置文件

4. **添加基础项目配置**:
   - `requirements.txt` - 依赖列表
   - `.gitignore` - 正确忽略数据集、检查点、pyc缓存
   - `README.md` - 项目说明文档

## 评估指标

支持标准时间序列预测评估指标:
- MSE, MAE, RMSE
- MAPE, SMAPE, WAPE, MSMAPE

## 许可证

本项目用于学术研究。
