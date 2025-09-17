import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class WeatherDatasetGenerator:
    def __init__(self, config: Dict):
        """
        气象数据集生成器
        
        Args:
            config: 配置字典，包含以下字段：
                - base_path: 数据根目录路径
                - weather_features: 要提取的气象要素列表
                - train_years: 训练集年份范围 [start_year, end_year]
                - val_years: 验证集年份范围 [start_year, end_year] 
                - test_years: 测试集年份范围 [start_year, end_year]
                - output_dir: 输出目录
                - time_column: 时间列名
                - station_id_column: 站点标识列名
                - lag_hours: 滞后小时数列表
        """
        self.config = config
        self.base_path = Path(config['base_path'])
        self.weather_features = config['weather_features']
        self.train_years = config['train_years']
        self.val_years = config['val_years']
        self.test_years = config['test_years']
        self.output_dir = Path(config['output_dir'])
        self.time_column = config.get('time_column', '时间')
        self.station_id_column = config.get('station_id_column', '区站号(数字)')
        self.lag_hours = config.get('lag_hours', [1, 2, 3, 4, 5, 6])
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def convert_time_format(self, time_value):
        """
        将时间格式从 2016010100 转换为 2016-01-01 00:00:00
        
        Args:
            time_value: 原始时间值 (例如: 2016010100)
        
        Returns:
            str: 格式化后的时间字符串 (例如: "2016-01-01 00:00:00")
        """
        try:
            # 将时间值转换为字符串，确保是10位数字
            time_str = str(int(time_value)).zfill(10)
            
            # 提取年、月、日、时
            year = time_str[:4]
            month = time_str[4:6]
            day = time_str[6:8]
            hour = time_str[8:10]
            
            # 构造标准时间格式，分钟和秒补零
            formatted_time = f"{year}-{month}-{day} {hour}:00:00"
            
            # 验证时间格式是否正确
            pd.to_datetime(formatted_time)
            
            return formatted_time
            
        except Exception as e:
            self.logger.error(f"时间格式转换失败: {time_value}, 错误: {e}")
            return None
    
    def process_time_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理时间列，将原始格式转换为标准datetime格式
        
        Args:
            df: 包含时间列的DataFrame
        
        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        if self.time_column not in df.columns:
            self.logger.error(f"未找到时间列: {self.time_column}")
            return df
        
        df_processed = df.copy()
        
        # 转换时间格式
        self.logger.debug("开始转换时间格式...")
        df_processed[self.time_column] = df_processed[self.time_column].apply(self.convert_time_format)
        
        # 删除转换失败的行
        valid_time_mask = df_processed[self.time_column].notna()
        invalid_count = (~valid_time_mask).sum()
        
        if invalid_count > 0:
            self.logger.warning(f"删除 {invalid_count} 行无效时间数据")
            df_processed = df_processed[valid_time_mask]
        
        # 转换为datetime类型
        try:
            df_processed[self.time_column] = pd.to_datetime(df_processed[self.time_column])
            self.logger.debug(f"时间列转换完成，数据类型: {df_processed[self.time_column].dtype}")
        except Exception as e:
            self.logger.error(f"时间列转换为datetime失败: {e}")
        
        return df_processed
    
    def get_available_years(self) -> List[int]:
        """获取可用的年份列表"""
        years = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                years.append(int(item.name))
        return sorted(years)
    
    def get_csv_files_in_year(self, year: int) -> List[Path]:
        """获取指定年份目录下的所有CSV文件"""
        year_dir = self.base_path / str(year)
        if not year_dir.exists():
            self.logger.warning(f"年份目录不存在: {year_dir}")
            return []
        
        csv_files = list(year_dir.glob("*.csv"))
        return sorted(csv_files)
    
    def load_csv_with_encoding(self, file_path: Path) -> Optional[pd.DataFrame]:
        """尝试不同编码加载CSV文件"""
        encodings = ['gb2312', 'utf-8', 'gbk', 'utf-8-sig']  
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.debug(f"成功使用 {encoding} 编码加载文件: {file_path}")
                return df
            except Exception as e:
                continue
                
        self.logger.error(f"无法加载文件: {file_path}")
        return None
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据验证和清洗"""
        df_clean = df.copy()
        
        # 定义需要检查异常值的温度特征
        temperature_features = ['温度/气温', 'tmp_grid']
        
        for feature in self.weather_features:
            if feature in df_clean.columns:
                # 只对温度特征进行>100的异常值处理
                if feature in temperature_features:
                    outlier_mask = df_clean[feature] > 100
                    outlier_count = outlier_mask.sum()
                    if outlier_count > 0:
                        self.logger.info(f"发现 {outlier_count} 个超过100的异常值在列 {feature}")
                        df_clean.loc[outlier_mask, feature] = np.nan
                
                # 对所有特征进行缺失值处理
                nan_mask = df_clean[feature].isna()
                nan_count = nan_mask.sum()
                if nan_count > 0:
                    self.logger.info(f"发现 {nan_count} 个缺失值在列 {feature}")
                    # 使用前后均值填充（线性插值）
                    df_clean[feature] = df_clean[feature].interpolate(method='linear')
                    # 如果首尾仍有NaN，使用前向或后向填充
                    df_clean[feature] = df_clean[feature].fillna(method='ffill')
                    df_clean[feature] = df_clean[feature].fillna(method='bfill')
        
        return df_clean
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建滞后特征 - 只创建tmp_grid的滞后特征
        
        Args:
            df: 输入数据框，必须按时间排序
        
        Returns:
            pd.DataFrame: 包含滞后特征的数据框
        """
        df_with_lag = df.copy()
        
        # 确保按时间排序
        df_with_lag = df_with_lag.sort_values(by=self.time_column).reset_index(drop=True)
        
        # 只对tmp_grid特征创建滞后特征
        if 'tmp_grid' in df_with_lag.columns:
            for lag_hour in self.lag_hours:
                lag_column_name = f"tmp_grid_lag_{lag_hour}h"
                
                # 创建滞后特征 - shift(lag_hour) 表示向后移动lag_hour行
                df_with_lag[lag_column_name] = df_with_lag['tmp_grid'].shift(lag_hour)
                
                self.logger.debug(f"创建滞后特征: {lag_column_name}")
        
        # 记录滞后特征创建后的缺失值情况
        max_lag = max(self.lag_hours)
        self.logger.info(f"滞后特征创建完成，前 {max_lag} 行将包含NaN值")
        
        return df_with_lag
    
    def fill_cross_year_lag_features(self, current_df: pd.DataFrame, previous_df: pd.DataFrame) -> pd.DataFrame:
        """
        使用前一年的数据填充跨年的滞后特征 - 只填充tmp_grid的滞后特征
        
        Args:
            current_df: 当前年份的数据（包含NaN的滞后特征）
            previous_df: 前一年的数据
        
        Returns:
            pd.DataFrame: 填充后的数据框
        """
        if previous_df is None or previous_df.empty:
            self.logger.warning("没有前一年数据，无法填充跨年滞后特征")
            return current_df
        
        df_filled = current_df.copy()
        
        # 获取需要填充的行数（最大滞后小时数）
        max_lag = max(self.lag_hours)
        
        # 获取前一年最后的数据点
        previous_df_sorted = previous_df.sort_values(by=self.time_column)
        last_records = previous_df_sorted.tail(max_lag)
        
        if len(last_records) < max_lag:
            self.logger.warning(f"前一年数据不足，只有 {len(last_records)} 条记录，需要 {max_lag} 条")
        
        # 只对tmp_grid特征填充滞后特征
        if 'tmp_grid' in df_filled.columns:
            for lag_hour in self.lag_hours:
                lag_column_name = f"tmp_grid_lag_{lag_hour}h"
                
                if lag_column_name in df_filled.columns:
                    # 找到需要填充的位置（前lag_hour行的NaN值）
                    for i in range(min(lag_hour, len(df_filled))):
                        if pd.isna(df_filled.loc[i, lag_column_name]):
                            # 从前一年数据中获取对应的值
                            if len(last_records) > (lag_hour - 1 - i):
                                source_idx = len(last_records) - (lag_hour - i)
                                if source_idx >= 0 and 'tmp_grid' in last_records.columns:
                                    fill_value = last_records.iloc[source_idx]['tmp_grid']
                                    if not pd.isna(fill_value):
                                        df_filled.loc[i, lag_column_name] = fill_value
                                        self.logger.debug(f"填充 {lag_column_name}[{i}] = {fill_value}")
        
        return df_filled
    
    def process_station_data(self, station_files: Dict[int, Path]) -> pd.DataFrame:
        """处理单个站点多年的数据"""
        station_data_by_year = {}
        
        # 按年份顺序处理和清洗数据
        for year in sorted(station_files.keys()):
            file_path = station_files[year]
            df = self.load_csv_with_encoding(file_path)
            
            if df is None:
                self.logger.warning(f"跳过文件: {file_path}")
                continue
            
            # 验证必要的列是否存在
            missing_cols = []
            required_cols = [self.time_column, self.station_id_column] + self.weather_features
            for col in required_cols:
                if col not in df.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                self.logger.warning(f"文件 {file_path} 缺少列: {missing_cols}")
                continue
            
            # 选择需要的列
            selected_cols = [self.time_column, self.station_id_column] + self.weather_features
            df_selected = df[selected_cols].copy()
            
            # 处理时间列格式转换
            df_selected = self.process_time_column(df_selected)
            
            if df_selected.empty:
                self.logger.warning(f"时间处理后数据为空: {file_path}")
                continue
            
            # 添加年份信息
            df_selected['年份'] = year
            
            # 数据清洗
            df_clean = self.validate_and_clean_data(df_selected)
            
            # 按时间排序
            df_clean = df_clean.sort_values(by=self.time_column).reset_index(drop=True)
            
            station_data_by_year[year] = df_clean
            self.logger.debug(f"处理完成: {file_path}, 数据行数: {len(df_clean)}")
        
        if not station_data_by_year:
            return pd.DataFrame()
        
        # 为每年数据创建滞后特征
        station_data_with_lag = {}
        sorted_years = sorted(station_data_by_year.keys())
        
        for i, year in enumerate(sorted_years):
            self.logger.info(f"为年份 {year} 创建滞后特征...")
            
            # 创建滞后特征
            df_with_lag = self.create_lag_features(station_data_by_year[year])
            
            # 如果不是第一年，使用前一年数据填充跨年滞后特征
            if i > 0:
                previous_year = sorted_years[i-1]
                previous_df = station_data_by_year[previous_year]
                df_with_lag = self.fill_cross_year_lag_features(df_with_lag, previous_df)
            
            station_data_with_lag[year] = df_with_lag
        
        # 合并多年数据
        combined_data = pd.concat(station_data_with_lag.values(), ignore_index=True)
        
        # 按时间排序
        combined_data = combined_data.sort_values(by=self.time_column).reset_index(drop=True)
        
        return combined_data
    
    def generate_datasets(self):
        """生成训练集、验证集和测试集"""
        self.logger.info("开始生成数据集...")
        
        # 获取所有可用年份
        available_years = self.get_available_years()
        self.logger.info(f"可用年份: {available_years}")
        
        # 检查配置的年份是否可用，同时需要前一年的数据用于滞后特征
        all_required_years = set()
        all_required_years.update(range(self.train_years[0], self.train_years[1] + 1))
        all_required_years.update(range(self.val_years[0], self.val_years[1] + 1))
        all_required_years.update(range(self.test_years[0], self.test_years[1] + 1))
        
        # 添加前一年用于滞后特征
        min_year = min(all_required_years)
        all_required_years.add(min_year - 1)
        
        missing_years = all_required_years - set(available_years)
        if missing_years:
            self.logger.warning(f"缺少年份数据: {sorted(missing_years)}")
        
        # 获取所有站点的文件映射
        station_file_map = {}  # {station_filename: {year: file_path}}
        
        for year in available_years:
            csv_files = self.get_csv_files_in_year(year)
            for file_path in csv_files:
                station_name = file_path.name
                if station_name not in station_file_map:
                    station_file_map[station_name] = {}
                station_file_map[station_name][year] = file_path
        
        self.logger.info(f"发现 {len(station_file_map)} 个站点")
        
        # 初始化数据集
        train_data_list = []
        val_data_list = []
        test_data_list = []
        
        # 处理每个站点
        for station_idx, (station_name, year_files) in enumerate(station_file_map.items()):
            self.logger.info(f"处理站点 {station_idx + 1}/{len(station_file_map)}: {station_name}")
            
            # 处理站点的所有年份数据（包括滞后特征）
            station_data = self.process_station_data(year_files)
            
            if station_data.empty:
                self.logger.warning(f"站点 {station_name} 无有效数据")
                continue
            
            # 按年份分割数据
            train_mask = station_data['年份'].between(self.train_years[0], self.train_years[1])
            val_mask = station_data['年份'].between(self.val_years[0], self.val_years[1])
            test_mask = station_data['年份'].between(self.test_years[0], self.test_years[1])
            
            train_subset = station_data[train_mask]
            val_subset = station_data[val_mask]
            test_subset = station_data[test_mask]
            
            if not train_subset.empty:
                train_data_list.append(train_subset)
            if not val_subset.empty:
                val_data_list.append(val_subset)
            if not test_subset.empty:
                test_data_list.append(test_subset)
        
        # 合并所有站点数据
        self.logger.info("合并数据集...")
        
        datasets = {}
        if train_data_list:
            datasets['train'] = pd.concat(train_data_list, ignore_index=True)
            self.logger.info(f"训练集大小: {len(datasets['train'])}")
        
        if val_data_list:
            datasets['val'] = pd.concat(val_data_list, ignore_index=True)
            self.logger.info(f"验证集大小: {len(datasets['val'])}")
        
        if test_data_list:
            datasets['test'] = pd.concat(test_data_list, ignore_index=True)
            self.logger.info(f"测试集大小: {len(datasets['test'])}")
        
        # 移除区站号和年份列，保存数据集
        for dataset_name, dataset in datasets.items():
            # 复制数据集用于保存
            dataset_to_save = dataset.copy()
            
            # 移除区站号和年份列
            columns_to_remove = []
            if self.station_id_column in dataset_to_save.columns:
                columns_to_remove.append(self.station_id_column)
            if '年份' in dataset_to_save.columns:
                columns_to_remove.append('年份')
            
            if columns_to_remove:
                dataset_to_save = dataset_to_save.drop(columns=columns_to_remove)
                self.logger.info(f"从 {dataset_name} 数据集中移除列: {columns_to_remove}")
            
            # 将datetime列转换为字符串格式保存
            if self.time_column in dataset_to_save.columns:
                dataset_to_save[self.time_column] = dataset_to_save[self.time_column].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            output_file = self.output_dir / f"{dataset_name}_dataset.csv"
            dataset_to_save.to_csv(output_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"保存 {dataset_name} 数据集到: {output_file}")
            
            # 生成数据集统计信息
            self.generate_dataset_stats(dataset_to_save, dataset_name)
        
        self.logger.info("数据集生成完成!")
        return datasets
    
    def generate_dataset_stats(self, dataset: pd.DataFrame, dataset_name: str):
        """生成数据集统计信息"""
        stats_file = self.output_dir / f"{dataset_name}_stats.txt"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"{dataset_name.upper()} 数据集统计信息\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总行数: {len(dataset)}\n")
            f.write(f"总列数: {len(dataset.columns)}\n\n")
            
            f.write("列信息:\n")
            for col in dataset.columns:
                f.write(f"  - {col} ({dataset[col].dtype})\n")
            f.write("\n")
            
            # 时间连续性统计
            if self.time_column in dataset.columns:
                time_col = pd.to_datetime(dataset[self.time_column])
                f.write(f"时间范围: {time_col.min()} 到 {time_col.max()}\n")
                f.write(f"总时间跨度: {time_col.max() - time_col.min()}\n")
                
                # 检查时间间隔
                time_diffs = time_col.diff().dropna()
                expected_interval = pd.Timedelta(hours=1)
                interval_counts = time_diffs.value_counts().sort_index()
                
                f.write(f"时间间隔统计:\n")
                for interval, count in interval_counts.head(5).items():
                    f.write(f"  - {interval}: {count} 次\n")
                
                # 检查是否完全连续（每小时一个点）
                non_hourly_intervals = time_diffs[time_diffs != expected_interval]
                if len(non_hourly_intervals) == 0:
                    f.write("时间序列完全连续（每小时一个点）\n")
                else:
                    f.write(f"发现 {len(non_hourly_intervals)} 个非标准时间间隔\n")
                
                f.write("\n")
            
            # 原始气象要素统计
            f.write("原始气象要素统计:\n")
            for feature in self.weather_features:
                if feature in dataset.columns:
                    series = dataset[feature]
                    f.write(f"\n{feature}:\n")
                    f.write(f"  - 数据类型: {series.dtype}\n")
                    f.write(f"  - 最小值: {series.min():.2f}\n")
                    f.write(f"  - 最大值: {series.max():.2f}\n")
                    f.write(f"  - 平均值: {series.mean():.2f}\n")
                    f.write(f"  - 标准差: {series.std():.2f}\n")
                    f.write(f"  - 缺失值: {series.isna().sum()}\n")
                    
                    # 对温度特征显示异常值处理信息
                    if feature in ['温度/气温', 'tmp_grid']:
                        f.write(f"  - 注意: 该特征已进行>100异常值处理\n")
            
            # 滞后特征统计 - 只统计tmp_grid的滞后特征
            f.write("\n滞后特征统计:\n")
            lag_features = [col for col in dataset.columns if col.startswith('tmp_grid_lag_')]
            f.write(f"滞后特征数量: {len(lag_features)}\n")
            
            if lag_features:
                f.write(f"滞后小时数: {sorted(self.lag_hours)}\n")
                
                for lag_feature in lag_features:
                    series = dataset[lag_feature]
                    f.write(f"\n{lag_feature}:\n")
                    f.write(f"  - 缺失值: {series.isna().sum()}\n")
                    f.write(f"  - 平均值: {series.mean():.2f}\n")
                    f.write(f"  - 标准差: {series.std():.2f}\n")


def main():
    """主函数 - 配置和运行数据集生成器"""
    
    # 配置参数
    config = {
        'base_path': r'C:\Users\Administrator\Desktop\SCG_ML\Data\Proc',  # 数据根目录，包含年份子目录
        'weather_features': [
            '温度/气温',           # 温度
            "tmp_grid",      # 格点温度
            "经度",
            "纬度",
            "测站高度",
            '相对湿度',           # 湿度
            '过去1小时降水量',     # 降水量
            '10分钟平均风速'      # 风速
        ],
        'train_years': [2015, 2017],    # 训练集年份范围
        'val_years': [2018, 2018],      # 验证集年份
        'test_years': [2019, 2019],     # 测试集年份
        'output_dir': './datasets',      # 输出目录
        'time_column': '时间',           # 时间列名
        'station_id_column': '区站号(数字)',  # 站点ID列名
        'lag_hours': [1, 2, 3, 4, 5, 6]  # 滞后小时数列表
    }
    
    # 创建生成器并运行
    generator = WeatherDatasetGenerator(config)
    datasets = generator.generate_datasets()
    
    print("\n数据集生成完成!")
    print(f"输出目录: {config['output_dir']}")
    print("生成的文件:")
    output_dir = Path(config['output_dir'])
    for file in output_dir.glob("*.csv"):
        print(f"  - {file.name}")
    for file in output_dir.glob("*.txt"):
        print(f"  - {file.name}")
    print(f"  - processing.log")


if __name__ == "__main__":
    main()