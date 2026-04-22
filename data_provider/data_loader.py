from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from data_provider.timefeatures import time_features


class Dataset_power_minute(Dataset):
    def __init__(self, model_name, data_path_list, flag, size=None, target='Power', scale=True, freq='60min',
                 date_name="Time",
                 get_pred_type="first", df_override=None):
        assert flag in ['train', 'test', 'val']
        self.model_name = model_name
        self.get_pred_type = get_pred_type
        self.flag = flag
        self.date_name = date_name

        self.data_path_list = data_path_list
        self.df_override = df_override

        self.seq_len = size[0]  # 输入序列长度
        self.pred_len = size[1]  # 预测长度
        self.target = target  # 目标标签
        self.scale = scale  # 是否数据标准化
        self.freq = freq  # 时序数据频率 15min
        self.__read_data__()  # 读取数据
        """
            tot_len表示有多少次预测，比如：
                power_608_test.csv中608条数据, 608 - 512(input多少条数据) - 96(预测多少条数据) + 1 = 1
        """

        # 根据flag设置不同的步长和总长度
        if (self.flag == 'test' or self.flag == 'val') and self.get_pred_type == "all":
            # 计算测试和验证时的总预测次数
            self.tot_len = (len(self.data_x) - self.seq_len - self.pred_len) // self.pred_len + 1
        else:
            # 表示有多少次预测，比如： power_608_test.csv中608条数据, 608 - 512(input多少条数据) - 96(预测多少条数据) + 1 = 1
            self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        # 为特征和目标分别创建scaler
        self.feature_scaler = StandardScaler()  # 用于输入特征
        self.target_scaler = StandardScaler()  # 用于目标变量

       
        if self.flag == 'train':
            df_raw = pd.read_csv(self.data_path_list[0])  # 训练数据
        elif self.flag == 'val':
            df_raw = pd.read_csv(self.data_path_list[1])  # 验证数据
        else:  # test
            df_raw = pd.read_csv(self.data_path_list[2])  # 测试集

        df_data = df_raw
        # 检查数据中是否包含nan值
        if df_data.isnull().values.any():
            raise ValueError(f"数据中包含nan值")

        # 多变量预测单变量
        cols_data = df_data.columns[1:]  # 获取数据列名（跳过第一列，通常是时间列）
        df_data = df_data[cols_data]

        # 从df_data中提取目标变量列
        if self.target in df_data.columns:
            df_target = df_data[[self.target]]  # 目标变量数据
        else:
            raise ValueError(f"目标变量 '{self.target}' 不在数据列中。可用列: {list(df_data.columns)}")

        if self.scale:
            # 输入特征进行标准化
            self.feature_scaler.fit(df_data.values)
            data = self.feature_scaler.transform(df_data.values)
            # print(f"前10行数据：{data[:10, :]}")

            # 目标变量单独进行标准化
            self.target_scaler.fit(df_target.values)
            target_data = self.target_scaler.transform(df_target.values)
        else:
            data = df_data.values
            target_data = df_data[[self.target]].values
            

        # 处理时间列
        df_stamp = df_raw[[self.date_name]]
        df_stamp[self.date_name] = pd.to_datetime(df_stamp[self.date_name])
        data_stamp = time_features(pd.to_datetime(df_stamp[self.date_name].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_stamp = data_stamp
        self.data_x = data
        self.data_y = target_data  # 目标变量（标准化后）
        # 检查数据中是否包含nan值
        # if np.isnan(self.data_x).any() or np.isnan(self.data_y).any():
        #     raise ValueError(f"数据中包含nan值")

    def __getitem__(self, index):
        # 根据flag计算实际的起始位置
        if (self.flag == 'test' or self.flag == 'val') and self.get_pred_type == "all":
            # 测试时每次移动pred_len步
            actual_index = index * self.pred_len
        else:
            # 训练和验证时每次移动1步
            actual_index = index

        input_begin = actual_index
        input_end = input_begin + self.seq_len  # 0 + 512(未来96个辐射量是他人预测好的)
        pred_begin = input_end  # 512
        pred_end = pred_begin + self.pred_len  # 512 + 96 = 608
        seq_x = self.data_x[input_begin:input_end]  # 0:512
        seq_y = self.data_y[pred_begin:pred_end]  # 512:608
        seq_x_mark = self.data_stamp[input_begin:input_end]  # 512:512+96
        seq_y_mark = self.data_stamp[pred_begin:pred_end]  # 512+96:608
        # # 检查数据中是否包含nan值
        # if np.isnan(seq_x).any() or np.isnan(seq_y).any():
        #     raise ValueError(f"数据中包含nan值，index={index}, actual_index={actual_index}")
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)

    def __len__(self):
        return self.tot_len


def get_data_loader(args, flag, shuffle_flag=True, drop_last=False):
    assert flag in ['train', 'val', 'test']
    # 测试集关闭shuffle
    if flag == 'test':
        shuffle_flag = False

    df_override = None


    data_set = Dataset_power_minute(
        model_name=args.model_name,
        flag=flag,
        data_path_list=args.data_path_list,
        size=[args.seq_len, args.pred_len],
        get_pred_type=args.get_pred_type,
        df_override=df_override
    )
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
