import chardet
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 加载训练好的模型
model = joblib.load('随机森林--test.pickle')

# 读取数据
with open('D:\桌面\项目\随机森林--牙根断裂风险\测试--随机森林.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

df = pd.read_csv('D:\桌面\项目\随机森林--牙根断裂风险\测试--随机森林.csv', encoding=encoding)
data = df[['初诊预后', '剩余牙体组织', '牙体损伤', '疼痛', '冠修复', '性别', '松动', '根管再治疗']]

# 对ever_married特征进行编码
data['性别'] = LabelEncoder().fit_transform(data['性别'])

# 标准化和降维
data = StandardScaler().fit_transform(data)
data = PCA().fit_transform(data)
print(data)

# 对整个数据集进行预测
result1 = model.predict(data)
print(result1)

# 对单挑数据进行预测
print(data[0].reshape(1, -1))
result2 = model.predict(data[0].reshape(1, -1)).tolist()
print(result2)

print('-' * 100)

# 再次读取数据
with open('D:\桌面\项目\随机森林--牙根断裂风险\验证--随机森林.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

df = pd.read_csv('D:\桌面\项目\随机森林--牙根断裂风险\验证--随机森林.csv', encoding=encoding)
data = df[['初诊预后', '剩余牙体组织', '牙体损伤', '疼痛', '冠修复', '性别', '松动', '根管再治疗']]


data['性别'] = LabelEncoder().fit_transform(data['性别'])

# 标准化和降维
data = StandardScaler().fit_transform(data)
print(data)


# 使用训练好的模型进行批量预测
results2 = model.predict(data).tolist()

# 打印预测结果
print(results2)
