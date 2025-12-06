import pandas as pd

# 读取 Excel 文件
file_path = r'C:\Users\10133\Desktop\evrp\改1\改1multi_period_instance_summary1-100.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 定义要比较的元启发式方法和 ALNS 方法
# methods_to_compare = ['SimulatedAnnealing1', 'FCFS', 'VariableNeighbourhoodSearch']
methods_to_compare = ['FCFS','SimulatedAnnealing1', 'VariableNeighbourhoodSearch', 'Adaptive']
alns_method = 'DRL-ALNS'

# 创建一个新的 DataFrame 存储每个实例文件的 \(\Delta_{Obj}\) 值
delta_obj_df = pd.DataFrame(columns=['File', 'Method', 'Delta_Obj (%)'])

# 对每个实例文件分别进行计算
for file in df['File'].unique():
    for method in methods_to_compare:
        # 获取基准方法 Z 和 ALNS 方法的平均目标值
        obj_z = df[(df['File'] == file) & (df['MetaHeuristic'] == method)]['mean'].values[0]
        obj_alns = df[(df['File'] == file) & (df['MetaHeuristic'] == alns_method)]['mean'].values[0]

        # 计算 Delta_Obj 并转换为百分比形式
        delta_obj = ((obj_alns - obj_z) / obj_z) * 100
        delta_obj = round(delta_obj, 1)  # 保留小数点后一位

        # 将百分比结果格式化为带有百分号的字符串
        delta_obj_str = f"{delta_obj}\%"

        # 将结果添加到新的 DataFrame 中
        delta_obj_df = delta_obj_df._append({'File': file, 'Method': method, 'Delta_Obj (%)': delta_obj_str},
                                           ignore_index=True)

# 显示计算结果
print(delta_obj_df)

# 保存结果到 Excel 文件
delta_obj_df.to_excel('1-100.xlsx', index=False)
