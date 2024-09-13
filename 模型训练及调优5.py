import warnings
import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import chardet
import numpy as np

warnings.filterwarnings('ignore')

# 读取文件并检测编码
with open(r'D:\桌面\项目\随机森林--牙根断裂风险\处理后的汇总.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

df = pd.read_csv(r'D:\桌面\项目\随机森林--牙根断裂风险\处理后的汇总.csv', encoding=encoding)

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

# Drop the original '初诊诊断' column
df.drop(columns=['初诊诊断'], inplace=True)

# 转换字符串类型为数值
string_columns = df.select_dtypes(include=['object']).columns.tolist()
for i in string_columns:
    df[i] = LabelEncoder().fit_transform(df[i])

# 特征选择（使用所有特征）
X = df.drop(columns=['风险'])
y = df['风险']

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, stratify=y)

# 使用 SMOTE 进行上采样
smote = SMOTE(random_state=1337)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 模型训练与调优 (XGBoost + RandomForest 集成)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
rf = RandomForestClassifier()

# XGBoost参数搜索
xgb_param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [1, 2, 3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
    'subsample': [0.8, 0.9, 1.0, 1.1, 1.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 1.5, 2]
}

# RandomForest参数搜索
rf_param_grid = {
    'n_estimators': [150, 175, 200, 225,240, 250, 260, 275, 300],
    'max_depth': [7, 8, 9, 10, 11, 12, 13, 14],
    'min_samples_split': [1, 2, 3, 5],
    'min_samples_leaf': [1, 2, 3]
}

# 进行网格搜索
xgb_grid_search = GridSearchCV(xgb, param_grid=xgb_param_grid, cv=5, scoring='accuracy')
rf_grid_search = GridSearchCV(rf, param_grid=rf_param_grid, cv=5, scoring='accuracy')

# 分别训练两个模型
xgb_grid_search.fit(X_train, y_train)
rf_grid_search.fit(X_train, y_train)

# 获取最佳参数
xgb_best_params = xgb_grid_search.best_params_
rf_best_params = rf_grid_search.best_params_

print("XGBoost 最优参数组合: ", xgb_best_params)
print("RandomForest 最优参数组合: ", rf_best_params)

# 使用最佳参数训练模型
xgb_optimized = XGBClassifier(**xgb_best_params, use_label_encoder=False, eval_metric='mlogloss')
rf_optimized = RandomForestClassifier(**rf_best_params)

# 创建投票分类器（集成XGBoost和RandomForest）
voting_clf = VotingClassifier(estimators=[('xgb', xgb_optimized), ('rf', rf_optimized)], voting='soft')
voting_clf.fit(X_train, y_train)

# 保存模型
joblib.dump(voting_clf, 'ensemble_model.pickle')

# 加载模型并进行预测
test_model = joblib.load('ensemble_model.pickle')
y_pred = test_model.predict(X_test)

# 打印结果
print("测试集预测结果:", y_pred)
print("实际值:", y_test.values)

# 计算并输出性能指标
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
