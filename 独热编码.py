import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# 读取Excel文件
excel_file_path = '汇总.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(excel_file_path)

# 删除“初诊诊断”列中每行的数字，只保留中文字符
df['初诊诊断'] = df['初诊诊断'].apply(lambda x: re.sub(r'\d+', '', x))

# 使用LabelEncoder进行编码，将每种中文情况用一个数字表示
label_encoder = LabelEncoder()
df['初诊诊断_编码'] = label_encoder.fit_transform(df['初诊诊断'])

# 输出处理后的数据
print(df[['初诊诊断', '初诊诊断_编码']])

# 保存处理后的数据为新的Excel文件
df.to_excel('处理后的汇总.xlsx', index=False)

# 保存处理后的数据为CSV文件
df.to_csv('处理后的汇总.csv', index=False)
