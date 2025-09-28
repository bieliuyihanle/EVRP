import pandas as pd

# 加载新的 Excel 数据
file_path = 'T7delta_obj_comparison.xlsx'  # 请替换为你的实际文件路径
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 使用 ffill() 方法填充空的文件名
df['File'] = df['File'].fillna(method='ffill')


# 映射实例名称
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


df['Instance'] = df['File'].apply(lambda x: map_instance_name(x) if 'Instance' in str(x) else None)

# 透视数据表格以便提取 Delta_Obj
pivot_df = df.pivot_table(index='Instance', columns='Method', values='Delta_Obj (%)', aggfunc='first')


# 生成 LaTeX 表格的函数
def generate_latex_table(pivot_df):
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|l|l|l|}\n\\hline\n"
    latex_table += "Instance & FCFS (Delta Obj) & SA (Delta Obj) & VNS (Delta Obj) \\\\ \\hline\n"

    for instance in pivot_df.index:
        fcfs_delta = pivot_df.loc[instance, 'FCFS'] if 'FCFS' in pivot_df.columns else '-'
        sa_delta = pivot_df.loc[instance, 'SimulatedAnnealing1'] if 'SimulatedAnnealing1' in pivot_df.columns else '-'
        vns_delta = pivot_df.loc[
            instance, 'VariableNeighbourhoodSearch'] if 'VariableNeighbourhoodSearch' in pivot_df.columns else '-'

        latex_table += f"{instance} & {fcfs_delta} & {sa_delta} & {vns_delta} \\\\ \n"

    latex_table += "\\end{tabular}\n\\caption{Comparison of Delta Obj for Different Methods}\n\\end{table}"

    return latex_table


# 生成 LaTeX 表格
latex_table_code = generate_latex_table(pivot_df)

# 输出生成的 LaTeX 表格代码
print(latex_table_code)
