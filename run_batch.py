import shutil
import subprocess
from utils.run_tools import check_is_complete, clear_fold

"""
----------------------------------------------------------------需要填写的运行参数---------------------------------------------------------------------------------
"""
# 运行模式
run_type = "0"  # 0:训练 1:测试
is_set_zero = False  # 是否将预测出来的负值置为0
get_pred_type = "all"  # first:取第一个值 last:取最后一个值,all:取所有值

# 训练相关配置
lradj = "adjust_fuc"  # 学习率调整方式 adjust_fuc/OneCycleLR
learning_rate = 0.0001  # 学习率
patience = 10  # 训练耐心值
train_epochs = 100  # 训练轮数
batch_size = 128  # 批大小

# 模型相关配置
num_tokens = 30

# 数据集
farm1 = [
    "datasets/train_dataset.csv",
    "datasets/test_dataset.csv",
    "datasets/val_dataset.csv",
    # "dataset/power/farm1/merged.csv",
    # "dataset/power/farm1/2022/farm_1_20220101_20221231_1479.csv",
    # "dataset/power/farm1/2023/farm_1_20230101_20231231_1479.csv",
    # "F:/TimeSeries/dataset/power/farm2/2023/farm_2_20230101_20231231_1479_hz_ctdre.csv"
]

farm2 = [
    "F:/TimeSeries/dataset/power/farm2/farm_2_20220101_20231231_clean.csv",
    "F:/TimeSeries/dataset/power/farm2/2022/farm_2_20220101_20221231_1479_hz.csv",
    "F:/TimeSeries/dataset/power/farm2/2023/farm_2_20230101_20231231_1479_hz_ctdre.csv"
]

data = farm1

# 配置需要执行的模型和权重
model_dict = {
    # "Autoformer": "20250607_192408_Autoformer",
    "DLinear": "20250914_225208_DLinear",
    # "Informer": "20250607_194222_Informer",
    # "TimesNet": "20250607_195312_TimesNet",
    # "TimeXer": "20250915_131144_TimeXer",
    # "Transformer": "20250607_200954_Transformer",
    # "TimeLLM_MSPF_GATEFUSION": "20250609_223505_TimeLLM_MSPF_GATEFUSION",
    # "LLMMixer": "20250608_001246_LLMMixer",
    # "TimeLLM": "20250609_141514_TimeLLM",
    # "iTransformer": "20250915_154757_iTransformer",
    # "PatchTST": "20250913_012805_PatchTST",
    # "TimeLLM_MSPF_HFUSION": "20250602_021841_TimeLLM_MSPF_GATEFUSION",
    # "TimeLLM_MSPF_FULL": "20250530_182637_TimeLLM_MSPF_FULL"
    # "TimeLLM_MSPF_AWAREMVF": "20250531_152538_TimeLLM_MSPF_AWAREMVF",
    # "PV_CNN": "20250727_231547_PV_CNN",
    # "PV_LSTM": "20250730_230805_PV_LSTM",
    # "PV_CNNLSTM": "20250731_134031_PV_CNNLSTM",
    # "WPMixer": "20250916_135315_WPMixer"
}

# ------------------------------------------------------------------------------------------------------------------------------------------------
# 1.执行模型
for model_name, weight_folder in model_dict.items():
    print(f"执行模型 {model_name}...")
    cmd = [
        "accelerate",
        "launch",
        "./run.py",
        # 公共参数 模型启动相关
        "--run_type", run_type,
        "--model_name", model_name,
        "--weight_foldername", weight_folder,
        "--data_path_list", f"['{data[0]}','{data[1]}','{data[2]}']",
        "--get_pred_type", get_pred_type,
        # 公共参数 模型训练相关
        "--seq_len", str(96),
        "--pred_len", str(8),
        "--label_len", str(0),
        "--dropout", str(0.1), 
        "--freq", "60min",
        "--lradj", str(lradj),
        "--learning_rate", str(learning_rate),
        "--patience", str(patience),
        "--batch_size", str(batch_size),
        "--train_epochs", str(train_epochs),
        # SolarTimeLLM
        "--num_tokens", str(num_tokens)
    ]
    # 修改后的代码
    if is_set_zero:
        cmd.append("--is_set_zero")
    print(str(cmd))
    subprocess.run(cmd)

if run_type == "1":
    # 2.获取模型运行完的结果
    all_file_abs = check_is_complete(model_dict, required_suffixes=["daily_metrics.csv", "pred.csv", "test.log"])
    target_dir = "A_result"

    # 3.清空目标目录
    clear_fold(target_dir)

    # 4.复制所有文件到目标目录
    for file_abs in all_file_abs:
        print(file_abs)
        shutil.copy2(file_abs, target_dir)
