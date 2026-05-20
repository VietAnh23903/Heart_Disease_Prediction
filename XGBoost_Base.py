from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import joblib

# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
df = pd.read_csv("train.csv")

# Chuyển nhãn về dạng số
df["Heart Disease"] = df["Heart Disease"].map({
    "Absence": 0,
    "Presence": 1
})

# ==========================================
# 2. TÁCH FEATURES VÀ LABEL
# ==========================================
X = df.drop(columns=["id", "Heart Disease"])
y = df["Heart Disease"]

# ==========================================
# 3. CHIA TRAIN / TEST
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ==========================================
# 4. XGBOOST BASELINE ĐƠN GIẢN (CỐ Ý ĐỂ HIỆU SUẤT THẤP HƠN)
# ==========================================
baseline_model = XGBClassifier(
    n_estimators=10,        # rất ít cây
    max_depth=1,            # cây nông
    learning_rate=0.3,      # tốc độ học cao
    subsample=0.3,
    colsample_bytree=0.3,
    eval_metric='logloss',
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

# ==========================================
# 5. HUẤN LUYỆN
# ==========================================
baseline_model.fit(X_train, y_train)

# ==========================================
# 6. DỰ ĐOÁN
# ==========================================
y_pred = baseline_model.predict(X_test)
y_proba = baseline_model.predict_proba(X_test)[:, 1]

# ==========================================
# 7. ĐÁNH GIÁ
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)




print("\n===== XGBOOST BASELINE =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1_score : {f1_score:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=2))
