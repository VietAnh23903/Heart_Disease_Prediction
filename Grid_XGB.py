from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from Data_Preprocessing import X_train, X_val, y_train, y_val, X_test_final
from Evaluate import evaluate
import pandas as pd

# Xây dựng đặc trưng
def create_features(df):
    df = df.copy()

    # Rủi ro theo độ tuổi
    df['age_risk'] = pd.cut(
        df['Age'],
        bins=[0, 40, 55, 65, 120],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Tương tác giữa huyết áp và cholesterol
    df['bp_chol_interaction'] = df['BP'] * df['Cholesterol'] / 10000

    # Tỉ lệ dự trữ nhịp tim tối đa đạt được
    df['hr_reserve_ratio'] = df['Max HR'] / (220 - df['Age'] + 1e-6)

    # Mức độ căng thẳng tim mạch
    df['stress_score'] = df['ST depression'] / (df['hr_reserve_ratio'] + 1e-6)

    # Thiếu máu cơ tim thầm lặng (đau ngực không triệu chứng điển hình nhưng có đau khi gắng sức)
    df['silent_ischemia_flag'] = (
        (df['Chest pain type'] == 4) &
        (df['Exercise angina'] == 1)
    ).astype(int)

    return df

X_train_xgb = create_features(X_train)
X_val_xgb = create_features(X_val)
X_test_xgb = create_features(X_test_final)

print("\n=== CÁC FEATURE SAU FEATURE ENGINEERING ===")
print(X_train_xgb.head(10))

# Tính tỷ lệ mất cân bằng
#ratio = (y_train == 0).sum() / (y_train == 1).sum()

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8],
    # 'scale_pos_weight': [ratio],
    'gamma': [0],
    'min_child_weight': [1, 3]
}

model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1,
    # scale_pos_weight=ratio,
    tree_method='hist'
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=2
)

grid_search.fit(X_train_xgb, y_train)

best_model = grid_search.best_estimator_

print("\nBest Params:", grid_search.best_params_)

# Đánh giá
evaluate(best_model, X_val_xgb, y_val, "GRID_XGB")

# Dự đoán
final_predictions = best_model.predict(X_test_xgb)

mapping = {0: 'Absence', 1: 'Presence'}
predicted_labels = [mapping[p] for p in final_predictions]

output = pd.DataFrame({
    'id': range(len(predicted_labels)),
    'Predicted_Heart_Disease': predicted_labels
})

print("\n[DỰ ĐOÁN TEST]")
print(output.head())

output.to_csv("xgb_predictions.csv", index=False)

import joblib

# Lưu mô hình tốt nhất
joblib.dump(best_model, 'best_xgb_model.pkl')
print('Đã lưu mô hình vào best_xgb_model.pkl')