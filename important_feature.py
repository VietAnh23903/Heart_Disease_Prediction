# ==========================================
# KIỂM TRA ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance
from Grid_XGB import X_train_xgb, best_model

# 1. Lấy độ quan trọng của các feature từ mô hình tốt nhất
feature_importance = pd.DataFrame({
    'Feature': X_train_xgb.columns,
    'Importance': best_model.feature_importances_
})

# 2. Sắp xếp giảm dần theo mức độ quan trọng
feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

# 3. In ra bảng kết quả
print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance)

# 4. Lưu ra file CSV
feature_importance.to_csv("feature_importance.csv", index=False)

# 5. Vẽ biểu đồ cột
plt.figure(figsize=(10, 8))
plt.barh(
    feature_importance['Feature'],
    feature_importance['Importance']
)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance - XGBoost")
plt.gca().invert_yaxis()   # Feature quan trọng nhất ở trên cùng
plt.tight_layout()
plt.show()

# # ==========================================
# # CÁCH KHÁC: Dùng hàm plot_importance của XGBoost
# # ==========================================
# plt.figure(figsize=(10, 8))
# plot_importance(
#     best_model,
#     importance_type='gain',   # gain = mức đóng góp vào giảm loss
#     max_num_features=20
# )
# plt.title("Top Feature Importance (Gain)")
# plt.tight_layout()
# plt.show()