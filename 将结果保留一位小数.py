
import pandas as pd

# 读取Excel文件
file_path =r'C:\Users\10133\Desktop\evrp\改\敏感性分析\CT\CT25.xlsx' # 替换为你的实际文件路径
df = pd.read_excel(file_path)

# 对指定的数值列进行保留小数点后一位
df['mean'] = df['mean'].round(1)
df['std'] = df['std'].round(1)
df['min'] = df['min'].round(1)

# 将处理后的数据保存为新的Excel文件
output_file_path = '1CT25.xlsx'  # 替换为你希望保存的文件路径
df.to_excel(output_file_path, index=False)

print("处理完成，已保存为新的文件:", output_file_path)
