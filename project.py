# project

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from sklearn.cluster import KMeans

# Load the files
path1 = r'E:\project\train.csv'
path2 = r'E:\project\test.csv'
train_data = pd.read_csv(path1)
test_data = pd.read_csv(path2)

# Check for missing values
train_miss = train_data.isnull().sum()
test_miss = test_data.isnull().sum()

train_miss_percentage = (train_miss / len(train_data)) * 100
test_miss_percentage = (test_miss / len(test_data)) * 100

# Combine missing values and proportions into a summary dataframe
train_miss_table = pd.DataFrame({
    'Missing Count': train_miss,
    'Percentage(%)': train_miss_percentage
})

test_miss_table = pd.DataFrame({
    'Missing Count': test_miss,
    'Percentage(%)': test_miss_percentage
})

print("Train Missing Summary:")
print(train_miss_table)
print("\nTest Missing Summary:")
print(test_miss_table)

def encode_non_numeric_with_dummies(data):
    """
    Encodes non-numeric columns into dummy/one-hot variables.
    Args:
        data (pd.DataFrame): DataFrame with potential non-numeric columns.
    Returns:
        pd.DataFrame: DataFrame with dummy variables.
    """
    # Convert non-numeric columns to dummy variables
    data_encoded = pd.get_dummies(data, drop_first=True)
    return data_encoded

def kmeans_impute(data, n_clusters=5, random_state=42):
    """
    Fill missing values using KMeans clustering.
    Args:
        data (pd.DataFrame): Data with missing values.
        n_clusters (int): Number of clusters for KMeans.
        random_state (int): Random state for reproducibility.
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    data_filled = data.copy()
    # Encode non-numeric columns using dummy variables
    data_encoded = encode_non_numeric_with_dummies(data_filled)
    numeric_cols = data_encoded.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        # Temporarily drop rows where the target column is NaN
        temp_data = data_encoded.dropna(subset=[col])

        # Select features for clustering
        feature_cols = temp_data.columns.difference([col])
        features = temp_data[feature_cols]

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(features_scaled)
        temp_data["cluster"] = clusters

        # Calculate the mean value of the target column per cluster
        cluster_means = temp_data.groupby("cluster")[col].mean()

        # Impute missing values
        missing_features = data_encoded.loc[data_encoded[col].isnull(), feature_cols]
        if not missing_features.empty:
            missing_scaled = scaler.transform(missing_features)
            missing_clusters = kmeans.predict(missing_scaled)
            data_encoded.loc[data_encoded[col].isnull(), col] = [
                cluster_means.get(cluster, data_encoded[col].mean())
                for cluster in missing_clusters
            ]
    # Replace original data with imputed numeric data
    for col in data_filled.columns:
        if col in data_encoded:
            data_filled[col] = data_encoded[col]
    return data_filled

# Apply K-means imputation to training and test datasets
train_data_filled = kmeans_impute(train_data)
test_data_filled = kmeans_impute(test_data)

# Feature selection
y = train_data_filled['log_pSat_Pa']
X = train_data_filled.drop(columns=['log_pSat_Pa', 'parentspecies'])
X = sm.add_constant(X)

def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_col]])).fit()
            new_pval[new_col] = model.pvalues[new_col]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add {best_feature} with p-value {best_pval}')
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f'Remove {worst_feature} with p-value {worst_pval}')
        if not changed:
            break
    return included

selected_features = stepwise_selection(X, y)
selected_features = [feature for feature in selected_features if feature != 'const']

# Filter selected features
X = train_data_filled[selected_features]
X_test = test_data_filled[selected_features]

# Ensure columns match
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = []

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Pipeline
rf_model = RandomForestRegressor(random_state=42)
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", rf_model),
    ]
)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    "regressor__n_estimators": [100, 300, 500, 700, 1000],  # 增加树的数量
    "regressor__max_depth": [10, 20, 30, 50, None],         # 允许更大的深度
    "regressor__min_samples_split": [2, 5, 10],             # 控制分裂节点的最小样本数
    "regressor__min_samples_leaf": [1, 2, 4, 8],            # 控制叶节点的最小样本数
    "regressor__max_features": ["sqrt", "log2", None],      # 控制每棵树选择的最大特征数
    "regressor__max_samples": [0.6, 0.8, 1.0],              # 控制每棵树的采样比例
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="r2", verbose=2, n_jobs=-1, error_score='raise'
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
print(f"验证集 R²: {r2_score(y_val, y_val_pred):.4f}")
print(f"验证集 MAE: {mean_absolute_error(y_val, y_val_pred):.4f}")

# Test predictions
if "ID" not in test_data_filled.columns:
    raise KeyError("The test dataset does not contain an 'ID' column.")
y_test_pred = best_model.predict(X_test)
submission = pd.DataFrame({"ID": test_data_filled["ID"], "TARGET": y_test_pred})
submission.to_csv(r"E:\project\submission_RF.csv", index=False)
print("提交文件已保存为 submission_RF.csv")
