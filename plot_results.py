import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

# Metrics
metrics = ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1 Score']


xgboost = [0.8882, 0.9550, 0.8815, 0.8673, 0.8744]
rf = [0.8793, 0.9487, 0.8788, 0.8478, 0.8630]
dt = [0.8752, 0.9437, 0.8675, 0.8519, 0.8596]
knn = [0.8682, 0.9231, 0.8579, 0.8463, 0.8521]

# Vị trí cột
x = np.arange(len(metrics))
width = 0.2

# Figure
plt.figure(figsize=(10, 6))

# Vẽ cột
plt.bar(x - 1.5*width, xgboost, width, label='XGBoost')
plt.bar(x - 0.5*width, rf, width, label='Random Forest')
plt.bar(x + 0.5*width, dt, width, label='Decision Tree')
plt.bar(x + 1.5*width, knn, width, label='KNN')

# Label
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('So sánh kết quả của mô hình')
plt.xticks(x, metrics)

# Hiển thị legend + grid
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()