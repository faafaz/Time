import shutil
import os
from pathlib import Path
import pandas as pd


def save_pred_csv(is_set_zero,get_pred_type, model_name, test_file_path, pred_config, pred_list, valid_indices=None):
    # 保存预测结果
    df_test = pd.read_csv(test_file_path)  # 读取原始数据


    if is_set_zero:
        pred_list = [max(0, x) for x in pred_list]

    if get_pred_type == 'last':
        full_pred = [None] * (pred_config[0] + pred_config[1] - 1) + pred_list  # 创建带None填充的完整预测列
    elif get_pred_type == 'first':
        full_pred = [None] * pred_config[0] + pred_list + (pred_config[1] - 1) * [None]  # 创建带None填充的完整预测列
    elif get_pred_type == 'all':
        # 取所有预测值的时候，移动步长就是pred_config[1]
        seq_len = pred_config[0]  # 输入序列长度
        pred_len = pred_config[1]  # 预测长度
        test_len = len(df_test)

        if valid_indices is not None:
            # 使用有效索引信息来填充预测结果
            full_pred = [None] * test_len

            # 确保有效索引和预测结果数量匹配
            if len(valid_indices) != len(pred_list):
                # 取较小的数量进行填充
                min_len = min(len(valid_indices), len(pred_list))
                for i in range(min_len):
                    if valid_indices[i] < test_len:
                        full_pred[valid_indices[i]] = pred_list[i]
            else:
                # 正常填充
                for i, pos in enumerate(valid_indices):
                    if pos < test_len:
                        full_pred[pos] = pred_list[i]
        else:
            # 如果没有有效索引信息，使用原来的逻辑
            test_step = pred_len  # 移动步长等于预测长度，无重叠
            full_pred = [None] * test_len

            # 计算窗口数量
            num_windows = (test_len - seq_len - pred_len) // test_step + 1

            # 按窗口填充预测值
            pred_idx = 0
            for window_idx in range(num_windows):
                # 当前窗口在测试集中的起始位置
                start_pos = window_idx * test_step
                # 预测值在测试集中的起始位置
                pred_start = start_pos + seq_len

                # 填充当前窗口的所有预测值
                for i in range(pred_len):
                    pos = pred_start + i
                    if pos < test_len and pred_idx < len(pred_list):
                        full_pred[pos] = pred_list[pred_idx]
                        pred_idx += 1

    print(f"测试集长度:{len(df_test)}|预测数组长度:{len(full_pred)}")
    df_test['pred'] = full_pred

    new_path = os.path.join(Path(test_file_path).parent, f"{model_name}_pred.csv")  # 生成新文件名
    df_test.to_csv(new_path, index=False)  # 保存文件
    return new_path


def export_time_station_tmpgrid_pred(pred_csv_path, time_col='Time', tmp_col='Power', pred_col='pred', pred_out_col='预测功率'):
    """
    从 <model>_pred.csv 中提取指定列并生成新CSV：
    列顺序为 [时间, Power, 预测功率]

    Args:
        pred_csv_path: 预测结果CSV路径
        time_col, tmp_col, pred_col: 列名映射
        pred_out_col: 导出时预测列的新名称
    Returns:
        输出CSV路径
    """
    # 读取CSV
    try:
        df = pd.read_csv(pred_csv_path)
    except Exception:
        df = pd.read_csv(pred_csv_path, encoding='gbk')

    # 校验列
    missing = [c for c in [time_col, tmp_col, pred_col] if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要列: {missing}，请检查输入文件: {pred_csv_path}")

    # 选择并重命名列
    out_df = df[[time_col, tmp_col, pred_col]].copy()
    if pred_out_col and pred_out_col != pred_col:
        out_df.rename(columns={pred_col: pred_out_col}, inplace=True)

    # 生成输出路径
    out_path = str(Path(pred_csv_path)).replace('.csv', '_subset.csv')
    out_df.to_csv(out_path, index=False)
    return out_path


def check_file_complete(file_path, file_names):
    files_abs = []
    missing_files = []
    # 检查文件完整性
    for file_name in file_names:
        file_path_abs = os.path.join(file_path, file_name)
        if not os.path.exists(file_path_abs):
            missing_files.append(file_path_abs)
        else:
            files_abs.append(file_path_abs)

    # 如果缺失必需文件，则清空数据文件夹并抛出异常
    if missing_files:
        raise FileNotFoundError(
            f"缺失必需文件: {', '.join(missing_files)}"
        )

    return files_abs


def check_is_complete(model_dict, required_suffixes=["station_metrics.csv", "pred.csv", "test.log"]):
    all_file_abs = []
    for model_name, weight_folder in model_dict.items():
        weight_path = os.path.join("./checkpoints", model_name, weight_folder)
        # 获取权重目录下的所有测试目录
        subdirs = []
        for item in Path(weight_path).iterdir():
            # 只保留目录类型的项
            if item.is_dir():
                subdirs.append(item)  # 这里的item就是包含了上级目录的
        # 按名称排序目录列表
        subdirs = sorted(subdirs)
        latest_run_path = subdirs[-1]

        # 检查哪些文件 每个权重下的结果文件不一样
        required_file_names = []
        for suffix in required_suffixes:
            required_file_name = f"{model_name}_{suffix}"
            # required_file_name = suffix
            required_file_names.append(required_file_name)

        files_abs = check_file_complete(latest_run_path, required_file_names)
        all_file_abs.extend(files_abs)

    return all_file_abs


# 清空结果文件夹
def clear_fold(target_dir):
    target_exists = os.path.exists(target_dir)
    if target_exists:
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)  # 重建空目录
    else:
        os.makedirs(target_dir)  # 重建空目录
