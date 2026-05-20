# ỨNG DỤNG XGBOOST TRONG DỰ ĐOÁN NGUY CƠ MẮC BỆNH TIM TỪ DỮ LIỆU CHỈ SỐ SỨC KHỎE

## Giới thiệu
Dự án này ứng dụng các kỹ thuật Học máy để dự đoán nguy cơ mắc bệnh tim dựa trên các thông tin lâm sàng của bệnh nhân. Nghiên cứu tiến hành so sánh nhiều mô hình phân loại như KNN, Decision Tree, Random Forest và XGBoost. Mô hình cuối cùng được lựa chọn là Grid_XGB, kết hợp giữa feature engineering, tối ưu siêu tham số, Stratified K-Fold Cross Validation, Early Stopping và SHAP để giải thích kết quả dự đoán.

## Mục tiêu
- Tiền xử lý và làm sạch bộ dữ liệu Heart Disease.
- So sánh hiệu suất của nhiều mô hình học máy.
- Tối ưu hóa mô hình XGBoost để nâng cao hiệu quả dự đoán.
- Đánh giá mô hình bằng các chỉ số Accuracy, Precision, Recall, F1-score và ROC-AUC.
- Giải thích kết quả dự đoán.

## Bộ dữ liệu Heart Disease Dataset
Bộ dữ liệu train.csv gồm 630.000 bản ghi với hai lớp mục tiêu:
- Absence: Không mắc bệnh tim
- Presence: Mắc bệnh tim

Phân bố lớp:
- Absence: 55,17%
- Presence: 44,83%

## Cách sử dụng:
- Chạy kiểm tra dữ liệu: Check_data
- Tiền xử lí dữ liệu: Data_Preprocessing
- Chạy các mô hình học máy: KNN, DecisionTree, Grid_XGB
- Biểu đồ so sánh kết quả chỉ số hiệu năng các mô hình: plot_results
- Mức độ quan trọng của các đặc trưng: important_feature
- Kết quả dự đoán trên file test: xgb_predictions
- Demo dự đoán bệnh tim demo.py: streamlit run demo.py

