import shutil
import subprocess
from utils.run_tools import check_is_complete, clear_fold

"""
----------------------------------------------------------------需要填写的运行参数---------------------------------------------------------------------------------
"""
# 运行模式
run_type = "1"  # 0:训练 1:测试
is_set_zero = True  # 是否将预测出来的负值置为0
get_pred_type = "all" # first:取第一个值 last:取最后一个值,all:取所有值

# 训练相关配置
lradj = "adjust_fuc"  # 学习率调整方式 adjust_fuc/OneCycleLR
learning_rate = 0.00001  # 学习率
patience = 10  # 训练耐心值
train_epochs = 50  # 训练轮数
batch_size = 64  # 批大小
  
# 模型相关配置
num_tokens = 30

# 数据集
farm1 = [
    # "dataset/cur_dataset/太阳坪风电场_48.0/train.csv",
    # "dataset/cur_dataset/太阳坪风电场_48.0/val.csv",
    # "dataset/cur_dataset/太阳坪风电场_48.0/test.csv", 
    # "dataset/cur_dataset/乐阳风电场_100.0/train.csv",
    # "dataset/cur_dataset/乐阳风电场_100.0/val.csv",
    # "dataset/cur_dataset/乐阳风电场_100.0/test.csv",
    # "dataset/cur_dataset/有名店风电场_80.0/train.csv",
    # "dataset/cur_dataset/有名店风电场_80.0/val.csv",
    # "dataset/cur_dataset/有名店风电场_80.0/test.csv",
    # "dataset/cur_dataset/黄石筠山风场_80.0/train.csv",
    # "dataset/cur_dataset/黄石筠山风场_80.0/val.csv",
    # "dataset/cur_dataset/黄石筠山风场_80.0/test.csv",
    # "dataset/cur_dataset/四眼坪风电场_二期__56.0/summer/train.csv",
    # "dataset/cur_dataset/四眼坪风电场_二期__56.0/summer/val.csv",
    # "dataset/cur_dataset/四眼坪风电场_二期__56.0/summer/test.csv",
    # "dataset/cur_dataset/四眼坪风电场_二期__56.0/train.csv",
    # "dataset/cur_dataset/四眼坪风电场_二期__56.0/val.csv",
    # "dataset/cur_dataset/四眼坪风电场_二期__56.0/test.csv",
    # "dataset/cur_dataset/麻城蔡家寨风场_50.0/train.csv",
    # "dataset/cur_dataset/麻城蔡家寨风场_50.0/val.csv",
    # "dataset/cur_dataset/麻城蔡家寨风场_50.0/test.csv",
    # "dataset/cur_dataset/荆门象河风场_100.0/train.csv",
    # "dataset/cur_dataset/荆门象河风场_100.0/val.csv",
    # "dataset/cur_dataset/荆门象河风场_100.0/test.csv",
    # "dataset/cur_dataset/利川一二期风场_96.0/train.csv",
    # "dataset/cur_dataset/利川一二期风场_96.0/val.csv",
    # "dataset/cur_dataset/利川一二期风场_96.0/test.csv",
    # "dataset/cur_dataset/wind_farm/farm6/train.csv",
    # "dataset/cur_dataset/wind_farm/farm6/val.csv",
    # "dataset/cur_dataset/wind_farm/farm6/test.csv",
    # "dataset/cur_dataset/wind_farm/farm4/train.csv",
    # "dataset/cur_dataset/wind_farm/farm4/val.csv",
    # "dataset/cur_dataset/wind_farm/farm4/test.csv",
    # "dataset/cur_dataset/wind_farm/farm1/train.csv",
    # "dataset/cur_dataset/wind_farm/farm1/val.csv",
    # "dataset/cur_dataset/wind_farm/farm1/test.csv",
    "dataset/cur_dataset/wind_farm/farm3/train.csv",
    "dataset/cur_dataset/wind_farm/farm3/val.csv",
    "dataset/cur_dataset/wind_farm/farm3/test.csv",
    # "dataset/cur_dataset/wind_farm/farm2/train.csv",
    # "dataset/cur_dataset/wind_farm/farm2/val.csv",
    # "dataset/cur_dataset/wind_farm/farm2/test.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Spring/train.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Spring/test.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Spring/test.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Summer/train.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Summer/test.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Summer/test.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Autumn/train.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Autumn/test.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Autumn/test.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Winter/train.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Winter/test.csv",
    # "dataset/cur_dataset/wind_farm/farm2/Winter/test.csv",
    # "dataset/cur_dataset/Location/train.csv",
    # "dataset/cur_dataset/Location/val.csv",
    # "dataset/cur_dataset/Location/test.csv",
]

data = farm1

# 配置需要执行的模型和权重
model_dict = {
    # "DLinear": "20260405_225336_DLinear",
    # "TimeXer": "20260405_185948_TimeXer",
    # "Transformer": "20260405_191000_Transformer",
    # "iTransformer": "20260405_192313_iTransformer",
    # "PatchTST": "20260405_193017_PatchTST",
    "iTransformer_xLSTM_VMD_Preprocessed": "20260403_210857_iTransformer_xLSTM_VMD_Preprocessed",
    # "TimesNet": "20260308_002426_TimesNet"
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
        "--pred_len", str(2),
        "--label_len", str(0),
        "--dropout", str(0.3), 
        "--freq", "15min",
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
    # 对于GAN模型，需要特殊处理路径
    # 因为GAN模型保存在 checkpoints/DLinear_Graph_GAN/ 下
    # 但运行时使用的 model_name 是 DLinear_Graph

    # 创建一个用于检查结果的字典
    check_dict = {}
    for model_name, weight_folder in model_dict.items():
        # 如果weight_folder包含"GAN"，说明是GAN模型
        if "GAN" in weight_folder:
            # 使用 DLinear_Graph_GAN 作为检查路径
            check_dict["DLinear_Graph_GAN"] = weight_folder
        else:
            check_dict[model_name] = weight_folder

    all_file_abs = check_is_complete(check_dict, required_suffixes=["pred.csv", "test.log"])
    target_dir = "A_result"

    # 3.清空目标目录
    clear_fold(target_dir)

    # 4.复制所有文件到目标目录
    for file_abs in all_file_abs:
        print(file_abs)
        shutil.copy2(file_abs, target_dir)
