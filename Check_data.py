import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Hàm kiểm tra thông tin tổng quan của dataset
def check_data(df, title="Dataset"):
    print(f"=== {title} ===")
    print(f"Số lượng dòng, cột: {df.shape}")
    print("\nKiểu dữ liệu của từng thuộc tính:")
    print(df.dtypes)
    print("\nSố lượng giá trị khuyết:")
    print(df.isnull().sum())
    print("\nThống kê mô tả cơ bản:")
    print(df.describe().round(2))
    print("-" * 30)

check_data(train_df, "Kết quả kiểm tra bộ dữ liệu")