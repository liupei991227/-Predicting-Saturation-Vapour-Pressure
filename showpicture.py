# /Users/liupei/Desktop/24fall period 2/introduction to ML/final report/iml24-term-project/


# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# 数据加载与预处理
root_path = '/Users/liupei/Desktop/24fall period 2/introduction to ML/final report/iml24-term-project/'
train_data = pd.read_csv(root_path + "train.csv")
test_data = pd.read_csv(root_path + "test.csv")

train_data['parentspecies'] = train_data['parentspecies'].fillna('unknown')
test_data['parentspecies'] = test_data['parentspecies'].fillna('unknown')

train_data = pd.get_dummies(train_data, columns=['parentspecies'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['parentspecies'], drop_first=True)

missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

X = train_data.drop(columns=['ID', 'log_pSat_Pa'])
y = train_data['log_pSat_Pa']
X_test = test_data.drop(columns=['ID'])

X_test = X_test[X.columns]

X['NumOfC_NumHBondDonors'] = X['NumOfC'] * X['NumHBondDonors']
X_test['NumOfC_NumHBondDonors'] = X_test['NumOfC'] * X_test['NumHBondDonors']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
xgb_model = XGBRegressor(
    random_state=42,
    n_estimators=650,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

# XGBoost 模型训练
xgb_model.fit(X_train, y_train)
y_val_pred = xgb_model.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred)
print(f"XGBoost Validation R²: {val_r2:.4f}")

# 初始化其他模型
models = {
    'Lasso': Lasso(alpha=0.01, random_state=42, max_iter=10000),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=600, learning_rate=0.05, max_depth=10, random_state=42)
}

r2_scores = {'XGBoost': val_r2}

# 训练与评估其他模型
for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    r2_scores[name] = r2
    print(f"{name} Validation R\u00b2: {r2:.4f}")

# 图示化对比（折线图）
plt.figure(figsize=(10, 6))
plt.plot(list(r2_scores.keys()), list(r2_scores.values()), marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
plt.ylim(0.6, 0.8)
plt.xlabel('Models')
plt.ylabel('R\u00b2 Score')
plt.title('R\u00b2 Score Comparison Across Models')
for i, (model, score) in enumerate(r2_scores.items()):
    plt.text(i, score + 0.005, f"{score:.4f}", ha='center', fontsize=10, color='black')
plt.grid(alpha=0.5)
plt.show()

# 计算每个模型的 MAE
mae_scores = {}
for name, model in models.items():
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    mae_scores[name] = mae
    print(f"{name} Validation MAE: {mae:.4f}")

# 添加 XGBoost 的 MAE
y_val_pred_xgb = xgb_model.predict(X_val)
mae_scores['XGBoost'] = mean_absolute_error(y_val, y_val_pred_xgb)
print(f"XGBoost Validation MAE: {mae_scores['XGBoost']:.4f}")

# 图示化对比（柱状图）
plt.figure(figsize=(10, 6))
plt.bar(mae_scores.keys(), mae_scores.values(), color=['blue', 'orange', 'green', 'purple'], alpha=0.7)
plt.ylim(1.1, 1.3)  # 根据具体数据调整范围
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE Comparison Across Models')

# 添加数值标注
for i, (model, score) in enumerate(mae_scores.items()):
    plt.text(i, score + 0.005, f"{score:.4f}", ha='center', fontsize=10, color='black')

plt.grid(axis='y', alpha=0.5)
plt.show()

# 计算每个模型的 RMSE
rmse_scores = {}
for name, model in models.items():
    y_val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)  # 设置 squared=False 计算 RMSE
    rmse_scores[name] = rmse
    print(f"{name} Validation RMSE: {rmse:.4f}")

# 添加 XGBoost 的 RMSE
y_val_pred_xgb = xgb_model.predict(X_val)
rmse_scores['XGBoost'] = mean_squared_error(y_val, y_val_pred_xgb, squared=False)
print(f"XGBoost Validation RMSE: {rmse_scores['XGBoost']:.4f}")

# 图示化对比（柱状图）
plt.figure(figsize=(10, 6))
plt.bar(rmse_scores.keys(), rmse_scores.values(), color=['blue', 'orange', 'green', 'purple'], alpha=0.7)
plt.ylim(min(rmse_scores.values()) * 0.9, max(rmse_scores.values()) * 1.1)  # 自动调整范围
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('RMSE Comparison Across Models')

# 添加数值标注
for i, (model, score) in enumerate(rmse_scores.items()):
    plt.text(i, score + 0.005, f"{score:.4f}", ha='center', fontsize=10, color='black')

plt.grid(axis='y', alpha=0.5)
plt.show()


# 在测试集上预测并保存（以 LightGBM 为例）
final_model = models['LightGBM']
y_test_pred = final_model.predict(X_test)
submission = pd.DataFrame({
    "ID": test_data['ID'],
    "log_pSat_Pa": y_test_pred
})
submission_path = root_path + "predictions_lgbm.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")
