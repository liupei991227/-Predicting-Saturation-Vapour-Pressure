import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder

# 1. 读取数据
train_data = pd.read_csv("/Users/liupei/Desktop/24fall period 2/introduction to ML/final report/iml24-term-project/train.csv")
test_data = pd.read_csv("/Users/liupei/Desktop/24fall period 2/introduction to ML/final report/iml24-term-project/test.csv")

# 分离特征和目标变量
X = train_data.drop(columns=["ID", "log_pSat_Pa"])
y = train_data["log_pSat_Pa"]

# 记录测试集 ID
test_ids = test_data["ID"]
X_test = test_data.drop(columns=["ID"])

# 确定数值特征和分类特征
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = ["parentspecies"]

# 2. 分类特征目标编码
encoder = TargetEncoder()
X[categorical_features] = encoder.fit_transform(X[categorical_features], y)
X_test[categorical_features] = encoder.transform(X_test[categorical_features])

# 3. 数据预处理
# 数值特征标准化
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ],
    remainder="passthrough"  # 保留目标编码后的分类特征
)

# 4. 特征选择（保留最相关的 10 个特征）
feature_selector = SelectKBest(score_func=f_regression, k=10)

# 5. 创建模型
lgb_model = LGBMRegressor(
    n_estimators=300,       # 树的数量
    learning_rate=0.05,     # 学习率
    max_depth=15,           # 最大深度
    num_leaves=31,          # 叶节点数量
    random_state=42         # 随机种子
)

# 6. 构建流水线
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),           # 数据预处理
        ("feature_selection", feature_selector),  # 特征选择
        ("regressor", lgb_model),                 # 模型
    ]
)

# 7. 数据集划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 DataFrame，保持列名一致
X_train = pd.DataFrame(X_train, columns=X.columns)
X_val = pd.DataFrame(X_val, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# 8. 模型训练
pipeline.fit(X_train, y_train)

# 9. 验证集评估
y_val_pred = pipeline.predict(X_val)
r2 = r2_score(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)

print(f"验证集 R²: {r2:.4f}")
print(f"验证集 MAE: {mae:.4f}")

# 10. 测试集预测
y_test_pred = pipeline.predict(X_test)

# 11. 生成提交文件
submission = pd.DataFrame({"ID": test_ids, "TARGET": y_test_pred})
submission.to_csv("/Users/liupei/Desktop/24fall period 2/introduction to ML/final report/iml24-term-project/submission_optimized.csv", index=False)
print("提交文件已保存为 submission_optimized.csv")

