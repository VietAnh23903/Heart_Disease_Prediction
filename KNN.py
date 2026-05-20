from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from Data_Preprocessing import X_train, X_val, y_train, y_val
from Evaluate import evaluate

# Lấy 10.000 mẫu từ tập train
X_train_sample = X_train.sample(n=10000, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

# Lấy 10.000 mẫu từ tập validation
X_val_sample = X_val.sample(n=10000, random_state=42)
y_val_sample = y_val.loc[X_val_sample.index]

# Scale dữ liệu
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_sample)
X_val_scaled = scaler.transform(X_val_sample)

# Model
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_scaled, y_train_sample)

# Dự đoán
evaluate(model, X_val_scaled, y_val_sample, "KNN")
