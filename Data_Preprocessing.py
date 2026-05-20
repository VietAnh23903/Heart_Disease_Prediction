import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Mã hóa biến mục tiêu
train_df['Heart Disease'] = train_df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

# Xử lý dữ liệu ngoại lai
def handle_outliers_iqr(df):
    df_clean = df.copy()
    
    # xử lý các cột số
    exclude_cols = [
        'id',
        'Heart Disease',
        'Sex',
        'Chest pain type',
        'FBS over 120',
        'EKG results',
        'Exercise angina',
        'Slope of ST',
        'Number of vessels fluro',
        'Thallium'
    ]
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        # Sử dụng hệ số thận trọng = 1.5
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Giá trị trung vị để thay thế
        median_val = df_clean[col].median()

        # Thay thế ngoại lai bằng Median
        df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = median_val

    return df_clean

train_df = handle_outliers_iqr(train_df)
test_df = handle_outliers_iqr(test_df)


# Loại bỏ cột ID và chuẩn bị tập đặc trưng (X), nhãn (y)
X = train_df.drop(columns=['id', 'Heart Disease'])
y = train_df['Heart Disease']
X_test_final = test_df.drop(columns=['id'])


# Chia tập dữ liệu để kiểm tra
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# print("\nShape:")
# print("X_train:", X_train.shape)
# print("X_val  :", X_val.shape)