import warnings
import pandas as pd
from imblearn.over_sampling import SMOTE
from skimage.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize, MinMaxScaler, StandardScaler, OneHotEncoder

# 使用 SMOTE 类对数据进行自动重采
# 网格搜索
# 基于贝叶斯优化的交叉验证搜索方法
from skopt import BayesSearchCV
# Real用于定义实数范围的搜索空间，Categorical用于定义类别变量的搜索空间。
from skopt.space import Real, Categorical
# 模型
import joblib
# 警告信息
import warnings
import chardet

warnings.filterwarnings('ignore')


with open('D:\桌面\项目\随机森林--牙根断裂风险\测试--随机森林.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

df = pd.read_csv('D:\桌面\项目\随机森林--牙根断裂风险\测试--随机森林.csv', encoding=encoding)

# 定义年龄划分的函数，用于根据年龄将数据集分为青年、中年和老年三个类别：
def age_classification(age):
    if 0 < age < 18:
        return '未成年'
    elif 18 <= age < 40:
        return '青年'
    elif 40 <= age < 65:
        return '中年'
    else:
        return '老年'


# 将年龄数据作为参数传递给age_classification函数, 将年龄分类结果作为新的一列添加到DataFrame中
df['age_class'] = df['年龄'].apply(age_classification)
# print(df.head(5))

# 选择所有为字符串类型的列
string_columns = df.select_dtypes(include=['object']).columns.tolist()
# print(string_columns)

for i in string_columns:
    uniqe = df[i].unique()
    numique = df[i].nunique()
    print(i, uniqe, numique)
    df[i] = LabelEncoder().fit_transform(df[i])  # 将字符串类型转化为数字类型
    # df[i] = OneHotEncoder().fit_transform(df[i].values.reshape(-1, 1)).toarray()   # 转化为独热编码


# 特征选择
print(df)
print(df.shape)
df = df[df.columns[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]]


# 相关系数
feature_select = df.corr()['风险'].rename('相关性系数')
print(feature_select)
feature_select = feature_select.drop('风险').map(lambda x: abs(x)).sort_values(ascending=False)

# 选择的特征
feature = feature_select[feature_select > 0.0].index.tolist()
feature.append('风险')
print('选择的特征为', feature)

# 数据
data = df[feature]
print(data)

# 查看stroke的样本是否均衡
print("查看stroke的样本是否均衡\n",data['风险'].value_counts())

# 先输出x，后输出y，中间是换行
X = data.iloc[:, :-1]
y = data.iloc[: , -1]
print(X, y, sep='\n')

# 随机拆分样本集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# 对训练数据进行欠采样   对比可知欠采样更优
# rus = RandomUnderSampler(random_state=0)
# X_train, y_train = rus.fit_resample(X_train, y_train)

# 使用 SMOTE 类对数据进行自动重采样
smote = SMOTE(random_state=1337)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 标准化  对比可知标准化更优
# StandardScaler
x_train = StandardScaler().fit_transform(X_train)  # 在这里heart_disease原本的取值就是0、1
x_test = StandardScaler().fit_transform(X_test)
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
print("x_train.shape:", x_train.shape, "\nx_test.shape:", x_test.shape)

'''
# MinMaxScaler
x_train = MinMaxScaler().fit_transform(X_train)
x_test = MinMaxScaler().fit_transform(X_test)
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
'''

# PCA降维
x_train = PCA().fit_transform(x_train)
x_test = PCA().fit_transform(x_test)
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)

# 划分训练集和测试集
'''
train_score = []
test_score = []
train_cross_score = []
test_cross_score = []

model_name = ['Random Forest']
model = RandomForestRegressor()

# 训练
model.fit(x_train, y_train)

# 预测
predict_y = model.predict(x_test)
print(model_name[0], model.score(x_test, y_test))

# 评分
train_score.append(model.score(x_train, y_train))
test_score.append(model.score(x_test, y_test))

# 交叉验证评分
train_cross_score.append(cross_val_score(model, x_train, y_train, cv=10).mean())
test_cross_score.append(cross_val_score(model, x_test, y_test, cv=10).mean())

result = pd.DataFrame({'训练集评分': train_score, '测试集评分': test_score, '交叉验证训练集': train_cross_score,
                       '交叉验证测试集': test_cross_score}, index=model_name).round(2)
print(result)
'''
# 模型调优
gau = RandomForestRegressor()
gau.fit(x_train, y_train)


# 定义超参数空间
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [8, 10],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 3]
}

# 进行网格搜索
grid_search = GridSearchCV(gau, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

# 获取网格搜索得到的最优超参数组合
best_params = grid_search.best_params_
print(best_params)


# follow the above, the first best_params: {'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 50}
# best_params = {'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 50}

# 使用最优超参数组合创建新的RandomForestRegressor实例
rf_optimized = RandomForestRegressor(**best_params)

# 构建随机森林回归模型
rf = RandomForestRegressor(n_estimators=50, random_state=42)  # 设置决策树的数量为50

# 训练模型
rf.fit(X_train, y_train)
# 输出最优超参数组合和相应的评分

print("最优参数组合: ", best_params)


# 获取最优的模型实例
Model = grid_search.best_estimator_

# 保存最优模型
joblib.dump(Model, '随机森林--test.pickle')

# 模型的使用
test_model = joblib.load('随机森林--test.pickle')

# 预测的结果
result = test_model.predict(x_test)
print(x_test)
print(result, y_test, sep='\n')


# 预测结果
y_pred = rf.predict(X_test)
y_test = pd.DataFrame(y_test.values, columns=y_test.columns)
