import warnings
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib
import chardet

warnings.filterwarnings('ignore')

# 读取文件并检测编码
with open(r'D:\桌面\项目\随机森林--牙根断裂风险\90汇总.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

df = pd.read_csv(r'D:\桌面\项目\随机森林--牙根断裂风险\90汇总.csv', encoding=encoding)


# 选择所有为字符串类型的列
string_columns = df.select_dtypes(include=['object']).columns.tolist()
# print(string_columns)

for i in string_columns:
    uniqe = df[i].unique()
    numique = df[i].nunique()
    print(i, uniqe, numique)
    df[i] = LabelEncoder().fit_transform(df[i])  # 将字符串类型转化为数字类型


# 准备数据
data = df
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=24)
print(X_test, y_test)

# 对训练数据进行欠采样   对比可知欠采样更优
rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_resample(X_train, y_train)

# 使用 SMOTE 进行上采样
# smote = SMOTE(random_state=1337)
# X_train, y_train = smote.fit_resample(X_train, y_train)



# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''
# MinMaxScaler
x_train = MinMaxScaler().fit_transform(X_train)
x_test = MinMaxScaler().fit_transform(X_test)
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
'''

'''
# PCA降维
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
'''


# 模型训练与调优
gau = RandomForestRegressor()
param_grid = {
    'n_estimators': [150, 175, 200, 225,240, 250, 260, 275, 300, 325, 350],
    'max_depth': [ 9, 10, 11, 12, 13, 14, 15, 16],
    'min_samples_split': [1, 2, 5],
    'min_samples_leaf': [1, 2, 3]
}

# 进行网格搜索
grid_search = GridSearchCV(gau, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
print("最优参数组合: ", best_params)

# 使用最佳参数训练模型
rf_optimized = RandomForestRegressor(**best_params)
rf_optimized.fit(X_train, y_train)

# 保存模型
joblib.dump(rf_optimized, '随机森林--test.pickle')

# 加载模型并进行预测
test_model = joblib.load('随机森林--test.pickle')
y_pred = test_model.predict(X_test)

# 打印结果
print("测试集预测结果:", y_pred)
print("实际值:", y_test.values)

# 计算并输出性能指标
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)
