# import pandas as pd
#
#
# def compute_stats(data, scale_start, scale_end):
#     # 根据规模范围过滤实例
#     instances_range = [f'Instance_{i}.txt' for i in range(scale_start, scale_end + 1)]
#     filtered_data = data[data['File'].isin(instances_range)]
#
#     # 按MetaHeuristic分组，计算平均值和标准差
#     stats = filtered_data.groupby('MetaHeuristic')['Cost'].agg(['mean', 'std']).reset_index()
#     stats.rename(columns={'mean': 'Average Cost', 'std': 'Standard Deviation'}, inplace=True)
#     return stats
#
#
# # 读取Excel文件
# file_path = r'C:\Users\10133\Desktop\meta_heuristics_results.xlsx'  # 修改为你的文件路径
# data = pd.read_excel(file_path)
#
# # 定义不同客户规模的范围
# scale_ranges = {
#     '10_customers': (1, 20),
#     '20_customers': (21, 40),
#     '30_customers': (41, 60),
#     '40_customers': (61, 80),
#     '50_customers': (81, 100)
# }
#
# # 计算每个规模的统计结果
# results = {scale: compute_stats(data, start, end) for scale, (start, end) in scale_ranges.items()}
#
# # 输出结果
# for scale, df in results.items():
#     print(f"\nStatistics for {scale}:")
#     print(df)


# import pandas as pd
#
#
# def compute_delta_mean(data, scale_start, scale_end):
#     # 根据规模范围过滤实例
#     instances_range = [f'Instance_{i}.txt' for i in range(scale_start, scale_end + 1)]
#     filtered_data = data[data['File'].isin(instances_range)]
#
#     # 清理数据，去除百分号和空格
#     filtered_data.loc[:, 'Delta_Obj (%)'] = filtered_data['Delta_Obj (%)'].str.replace('%', '').str.strip()
#
#     # 将字符串转换为浮点数
#     filtered_data.loc[:, 'Delta_Obj (%)'] = pd.to_numeric(filtered_data['Delta_Obj (%)'], errors='coerce') / 100.0
#
#     # 按MetaHeuristic分组，计算delta的平均值
#     delta_stats = filtered_data.groupby('Method')['Delta_Obj (%)'].mean().reset_index()
#     delta_stats.rename(columns={'Delta_Obj (%)': 'Average Delta'}, inplace=True)
#     return delta_stats
#
#
# # 读取Excel文件
# file_path = r'C:\Users\10133\Desktop\evrp\改\T7delta_obj_comparison.xlsx'  # 修改为你的文件路径
# data = pd.read_excel(file_path)
#
# # 确保'delta'列存在
# if 'Delta_Obj (%)' not in data.columns:
#     print("数据集中没有'Delta_Obj (%)'列，请确认文件格式是否正确。")
# else:
#     # 定义不同客户规模的范围
#     scale_ranges = {
#         '10_customers': (1, 20),
#         '20_customers': (21, 40),
#         '30_customers': (41, 60),
#         '40_customers': (61, 80),
#         '50_customers': (81, 100)
#     }
#
#     # 计算每个规模的delta平均值
#     delta_results = {scale: compute_delta_mean(data, start, end) for scale, (start, end) in scale_ranges.items()}
#
#     # 输出结果
#     for scale, df in delta_results.items():
#         print(f"\nAverage Delta for {scale}:")
#         print(df)


#求规模所有实例最大最小平均值
import pandas as pd


def compute_mean_stats(data, scale_start, scale_end):
    # 根据规模范围过滤实例
    instances_range = [f'Instance_{i}.txt' for i in range(scale_start, scale_end + 1)]
    filtered_data = data[data['File'].isin(instances_range)]

    # 按算法分组，计算均值的平均值、最大值和最小值
    mean_stats = filtered_data.groupby('MetaHeuristic')['mean'].agg(['mean', 'max', 'min']).reset_index()
    mean_stats.rename(columns={'mean': 'Average of Means', 'max': 'Max of Means', 'min': 'Min of Means'}, inplace=True)
    return mean_stats


# 读取Excel文件
file_path = '保存algorithm_results.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)
data['File'] = data['File'].fillna(method='ffill')

# 定义不同客户规模的范围
scale_ranges = {
    '10_customers': (1, 20),
    '20_customers': (21, 40),
    '30_customers': (41, 60),
    '40_customers': (61, 80),
    '50_customers': (81, 100)
}

# 计算每个规模的统计结果
mean_results = {scale: compute_mean_stats(data, start, end) for scale, (start, end) in scale_ranges.items()}

# 输出结果
for scale, df in mean_results.items():
    print(f"\nStatistics of Means for {scale}:")
    print(df)
