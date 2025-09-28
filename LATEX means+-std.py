import pandas as pd

# 加载 Excel 文件并读取数据
file_path = 'T7rounded_output.xlsx'  # 请替换为你的 Excel 文件路径
df = pd.read_excel(file_path, sheet_name='Sheet1')


# 将文件名映射为 TxxIxx 格式的实例名称
def map_instance_name(row):
    instance_number = int(row.split('_')[1].split('.')[0])
    if 1 <= instance_number <= 20:
        return f"T10I{instance_number}"
    elif 21 <= instance_number <= 40:
        return f"T20I{instance_number - 20}"
    elif 41 <= instance_number <= 60:
        return f"T30I{instance_number - 40}"
    elif 61 <= instance_number <= 80:
        return f"T40I{instance_number - 60}"
    elif 81 <= instance_number <= 100:
        return f"T50I{instance_number - 80}"
    else:
        return None


# 映射实例名称
df['Instance'] = df['File'].apply(lambda x: map_instance_name(x) if 'Instance' in str(x) else None)
df_cleaned = df.dropna(subset=['Instance', 'MetaHeuristic'])

df['File'] = df['File'].fillna(method='ffill')

# 然后重新映射实例名称
df['Instance'] = df['File'].apply(lambda x: map_instance_name(x) if 'Instance' in str(x) else None)

# 透视表格以便于数据提取
pivot_df = df.pivot_table(index='Instance', columns='MetaHeuristic', values=['mean', 'std'], aggfunc='first', dropna=False)



# 生成 LaTeX 表格的函数
def generate_latex_table(pivot_df):
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|l|l|l|l|}\n\\hline\n"
    latex_table += "Instance & FCFS (mean) & SA (mean ± std) & VNS (mean ± std) & ALNS (mean ± std) \\\\ \\hline\n"

    for instance in pivot_df.index:
        alns_mean = pivot_df.loc[instance, ('mean', 'Adaptive')] if ('mean', 'Adaptive') in pivot_df.columns else '-'
        alns_std = pivot_df.loc[instance, ('std', 'Adaptive')] if ('std', 'Adaptive') in pivot_df.columns else '-'

        fcfs_mean = pivot_df.loc[instance, ('mean', 'FCFS')] if ('mean', 'FCFS') in pivot_df.columns else '-'

        sa_mean = pivot_df.loc[instance, ('mean', 'SimulatedAnnealing1')] if ('mean',
                                                                              'SimulatedAnnealing1') in pivot_df.columns else '-'
        sa_std = pivot_df.loc[instance, ('std', 'SimulatedAnnealing1')] if ('std',
                                                                            'SimulatedAnnealing1') in pivot_df.columns else '-'

        vns_mean = pivot_df.loc[instance, ('mean', 'VariableNeighbourhoodSearch')] if ('mean',
                                                                                       'VariableNeighbourhoodSearch') in pivot_df.columns else '-'
        vns_std = pivot_df.loc[instance, ('std', 'VariableNeighbourhoodSearch')] if ('std',
                                                                                     'VariableNeighbourhoodSearch') in pivot_df.columns else '-'



        latex_table += f"{instance} & {fcfs_mean} & {sa_mean} $\pm$ {sa_std} & {vns_mean} $\pm$ {vns_std} & {alns_mean} $\pm$ {alns_std} \n"

    latex_table += "\\end{tabular}\n\\caption{Comparison of Algorithm Performances}\n\\end{table}"

    return latex_table


# 生成 LaTeX 表格
latex_table_code = generate_latex_table(pivot_df)

print(latex_table_code)
