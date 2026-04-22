import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_csv_file(file_path):
    """
    检查CSV文件的时间连续性和异常值
    
    参数:
    file_path (str): CSV文件路径
    
    返回:
    dict: 包含检查结果的字典
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        results = {
            'file_loaded': True,
            'time_column_continuous': True,
            'time_gaps': [],
            'columns_with_issues': {},
            'summary': ''
        }
        
        # 检查是否有时间列
        if df.empty:
            results['summary'] = '文件为空'
            return results
            
        time_col = df.columns[0]
        results['time_column_name'] = time_col
        
        # 转换时间列为datetime格式
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            results['summary'] = f'时间列格式错误: {str(e)}'
            return results
        
        # 检查时间连续性 (1小时间隔)
        time_diffs = df[time_col].diff().dropna()
        expected_interval = timedelta(hours=1)
        
        gaps = time_diffs[time_diffs != expected_interval]
        if not gaps.empty:
            results['time_column_continuous'] = False
            for idx, gap in gaps.items():
                prev_time = df[time_col].iloc[idx-1]
                expected_time = prev_time + expected_interval
                actual_time = df[time_col].iloc[idx]
                results['time_gaps'].append({
                    'position': idx,
                    'expected_time': expected_time,
                    'actual_time': actual_time,
                    'gap_size': gap
                })
        
        # 检查各列的异常值
        for col in df.columns[1:]:  # 跳过时间列
            col_issues = {
                'nan_count': 0,
                'non_numeric_count': 0,
                'large_value_count': 0,
                'issues_found': False
            }
            
            # 检查NaN值
            nan_mask = df[col].isna()
            col_issues['nan_count'] = nan_mask.sum()
            
            # 检查非数值型数据
            non_numeric_mask = ~df[col].apply(lambda x: isinstance(x, (int, float, np.number)))
            col_issues['non_numeric_count'] = non_numeric_mask.sum()
            
            # 检查大数值 (>= 99999)
            try:
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                large_value_mask = numeric_values >= 99999
                col_issues['large_value_count'] = large_value_mask.sum()
            except:
                col_issues['large_value_count'] = len(df[col])
            
            # 检查是否有任何问题
            if (col_issues['nan_count'] > 0 or 
                col_issues['non_numeric_count'] > 0 or 
                col_issues['large_value_count'] > 0):
                col_issues['issues_found'] = True
            
            results['columns_with_issues'][col] = col_issues
        
        # 生成摘要信息
        summary_parts = []
        if not results['time_column_continuous']:
            summary_parts.append(f"时间列不连续，发现 {len(results['time_gaps'])} 处间隔错误")
        
        issue_cols = [col for col, issues in results['columns_with_issues'].items() 
                     if issues['issues_found']]
        if issue_cols:
            summary_parts.append(f"{len(issue_cols)} 个特征列存在异常值")
        else:
            summary_parts.append("所有特征列未发现异常值")
        
        results['summary'] = "; ".join(summary_parts)
        return results
        
    except Exception as e:
        return {
            'file_loaded': False,
            'error': str(e),
            'summary': f'处理文件时出错: {str(e)}'
        }

def print_detailed_report(results):
    """打印详细的检查报告"""
    if not results['file_loaded']:
        print(f"错误: {results['error']}")
        return
    
    print("=" * 50)
    print("CSV文件检查报告")
    print("=" * 50)
    print(f"时间列名称: {results.get('time_column_name', '未知')}")
    print(f"时间连续性: {'是' if results['time_column_continuous'] else '否'}")
    
    if not results['time_column_continuous']:
        print("\n时间间隔问题详情:")
        for gap in results['time_gaps']:
            print(f"  位置 {gap['position']}: ")
            print(f"    期望时间: {gap['expected_time']}")
            print(f"    实际时间: {gap['actual_time']}")
            print(f"    间隔大小: {gap['gap_size']}")
    
    print("\n特征列异常值检查:")
    for col, issues in results['columns_with_issues'].items():
        if issues['issues_found']:
            print(f"  {col}:")
            if issues['nan_count'] > 0:
                print(f"    NaN值数量: {issues['nan_count']}")
            if issues['non_numeric_count'] > 0:
                print(f"    非数值数据数量: {issues['non_numeric_count']}")
            if issues['large_value_count'] > 0:
                print(f"    大数值(>=99999)数量: {issues['large_value_count']}")
        else:
            print(f"  {col}: 无异常") 
    
    print("\n摘要:", results['summary'])
    print("=" * 50)

# 使用示例
if __name__ == "__main__":
    # 替换为您的CSV文件路径
    csv_file_path = r"dataset\cur_dataset\wind_farm\farm2\train.csv"
    
    # 检查文件
    results = check_csv_file(csv_file_path)
    
    # 打印详细报告
    print_detailed_report(results)