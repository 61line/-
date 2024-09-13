import warnings
import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import chardet
import numpy as np

warnings.filterwarnings('ignore')

# 读取文件并检测编码
with open(r'D:\桌面\项目\随机森林--牙根断裂风险\90测试-随机森林.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

df = pd.read_csv(r'D:\桌面\项目\随机森林--牙根断裂风险\90测试-随机森林.csv', encoding=encoding)

# 年龄分类函数
def age_classification(age):
    if 0 < age < 18:
        return '未成年'
    elif 18 <= age < 40:
        return '青年'
    elif 40 <= age < 65:
        return '中年'
    else:
        return '老年'

df['age_class'] = df['年龄'].apply(age_classification)

# 转换字符串类型为数值
string_columns = df.select_dtypes(include=['object']).columns.tolist()
for i in string_columns:
    df[i] = LabelEncoder().fit_transform(df[i])

# 特征选择（使用所有特征）
X = df.drop(columns=['风险'])
y = df['风险']

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# 使用 SMOTE 进行上采样
smote = SMOTE(random_state=1337)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 模型训练与调优 (XGBoost)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [1, 2, 3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.1, 0.15, 0.2, 0.25],
    'subsample': [0.8, 0.9, 1.0, 1.1, 1.2]
}

# 进行网格搜索
grid_search = GridSearchCV(xgb, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
print("最优参数组合: ", best_params)

# 使用最佳参数训练模型
xgb_optimized = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')
xgb_optimized.fit(X_train, y_train)

# 保存模型
joblib.dump(xgb_optimized, 'xgboost_model.pickle')

# 加载模型并进行预测
test_model = joblib.load('xgboost_model.pickle')
y_pred = test_model.predict(X_test)

# 打印结果
print("测试集预测结果:", y_pred)
print("实际值:", y_test.values)

# 计算并输出性能指标
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
epsilon = 1e-10  # 小常量，防止除以零
loss_rate = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)
print("损失率:", loss_rate, "%")
