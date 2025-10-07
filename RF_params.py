import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. 读取数据
train_data = pd.read_csv("/Users/liupei/Desktop/24fall period 2/introduction to ML/final report/iml24-term-project/train.csv")
test_data = pd.read_csv("/Users/liupei/Desktop/24fall period 2/introduction to ML/final report/iml24-term-project/test.csv")

# 分离特征和目标变量
X = train_data.drop(columns=["ID", "log_pSat_Pa"])
y = train_data["log_pSat_Pa"]

# 记录测试集 ID
test_ids = test_data["ID"]
X_test = test_data.drop(columns=["ID"])

# 2. 数据预处理
# 确定数值特征和分类特征
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = ["parentspecies"]

# 创建预处理流水线
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 3. 构建随机森林模型流水线
rf_model = RandomForestRegressor(random_state=42)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", rf_model),
    ]
)

# 4. 拆分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 超参数调优（可选）
param_grid = {
    "regressor__n_estimators": [200, 400, 600, 800],  # 增加树的数量
    "regressor__max_depth": [10, 20, 30, None],       # 允许更大的深度
    "regressor__min_samples_split": [2, 5, 10, 15],   # 控制分裂节点的最小样本数
    "regressor__min_samples_leaf": [1, 2, 4],         # 控制叶节点的最小样本数
    "regressor__max_features": ["sqrt", "log2", None],# 控制每棵树选择的最大特征数
    "regressor__max_samples": [0.8, 0.9, 1.0],  
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, scoring="r2", verbose=2, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 打印最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳模型预测
best_model = grid_search.best_estimator_

# 6. 评估模型性能
y_val_pred = best_model.predict(X_val)
r2 = r2_score(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)

print(f"验证集 R²: {r2:.4f}")
print(f"验证集 MAE: {mae:.4f}")

# 7. 测试集预测
y_test_pred = best_model.predict(X_test)

# 生成提交文件
submission = pd.DataFrame({"ID": test_ids, "TARGET": y_test_pred})
submission.to_csv("/Users/liupei/Desktop/24fall period 2/introduction to ML/final report/iml24-term-project/submission_RF.csv", index=False)
print("提交文件已保存为 submission.csv")

