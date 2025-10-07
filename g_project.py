# 导入必要的库
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

# 1. 读取数据

root_path = '/Users/gaohao/Desktop/'
train_data = pd.read_csv(root_path + "train.csv")
test_data = pd.read_csv(root_path + "test.csv")


# 数据预处理
# 填充分类变量的缺失值
train_data['parentspecies'] = train_data['parentspecies'].fillna('unknown')
test_data['parentspecies'] = test_data['parentspecies'].fillna('unknown')

# 独热编码分类变量
train_data = pd.get_dummies(train_data, columns=['parentspecies'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['parentspecies'], drop_first=True)

# 确保测试集与训练集特征一致
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

# 分离特征和目标变量
X = train_data.drop(columns=['ID', 'log_pSat_Pa'])
y = train_data['log_pSat_Pa']
X_test = test_data.drop(columns=['ID'])

# 确保列顺序一致
X_test = X_test[X.columns]

# 创建交互特征
X['NumOfC_NumHBondDonors'] = X['NumOfC'] * X['NumHBondDonors']
X_test['NumOfC_NumHBondDonors'] = X_test['NumOfC'] * X_test['NumHBondDonors']

# 分割训练和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化XGBoost模型（降低复杂度）
xgb_model = XGBRegressor(
    random_state=42,
    n_estimators=650,        # 减少迭代次数
    max_depth=5,             # 5 > 4
    learning_rate=0.05,       # 小学习率通常能更稳定地提高性能
    subsample=0.8,           # 调低到 0.7-0.8 以引入更多随机性，减少过拟合。
    colsample_bytree=0.8,    # 控制每棵树训练时使用的特征比例.调低到 0.7-0.8，适当引入随机性，避免模型过拟合
    n_jobs=-1                # 并行计算
)

# 模型训练
xgb_model.fit(X_train, y_train)

# 验证集预测
y_val_pred = xgb_model.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred)
print(f"Validation R²: {val_r2:.4f}")

# 在测试集上预测
y_test_pred = xgb_model.predict(X_test)

# 生成提交文件
submission = pd.DataFrame({
    "ID": test_data['ID'],
    "log_pSat_Pa": y_test_pred
})

# 保存预测结果
submission_path = root_path + "predictions.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")
