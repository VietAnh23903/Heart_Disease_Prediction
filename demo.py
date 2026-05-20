from Grid_XGB import create_features
import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load('best_xgb_model.pkl')

# ==============================
# GIAO DIỆN
# ==============================
st.set_page_config(
    page_title='Heart Disease Prediction',
    page_icon='❤️',
    layout='centered'
)

st.title('❤️ Dự Đoán Nguy Cơ Mắc Bệnh Tim')
st.write('Nhập các chỉ số sức khỏe để dự đoán xác suất mắc bệnh tim.')

# ==============================
# NHẬP DỮ LIỆU
# ==============================
Age = st.number_input('Age (Tuổi)', 1, 120, 50)
Sex = st.selectbox('Sex (Giới tính)', [0, 1], format_func=lambda x: 'Nữ (0)' if x == 0 else 'Nam (1)')
Chest_pain_type = st.selectbox(
    'Chest Pain Type',
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: '1 - Đau thắt ngực điển hình',
        2: '2 - Đau thắt ngực không điển hình',
        3: '3 - Không phải đau thắt ngực',
        4: '4 - Không có triệu chứng'
    }[x]
)

BP = st.number_input(
    'Blood Pressure (BP) - Huyết áp lúc nghỉ (mmHg)',
    min_value=50,
    max_value=250,
    value=120,
    help='Huyết áp tâm thu khi nghỉ ngơi. Giá trị bình thường thường khoảng 90-140 mmHg.'
)

Cholesterol = st.number_input(
    'Cholesterol (mg/dL)',
    min_value=50,
    max_value=700,
    value=200,
    help='Nồng độ cholesterol toàn phần trong máu. Bình thường thường dưới 200 mg/dL.'
)

FBS_over_120 = st.selectbox(
    'FBS over 120',
    [0, 1],
    format_func=lambda x:
        '0 - Đường huyết lúc đói ≤ 120 mg/dL'
        if x == 0 else
        '1 - Đường huyết lúc đói > 120 mg/dL',
    help='Fasting Blood Sugar: chỉ số đường huyết lúc đói.'
)

EKG_results = st.selectbox(
    'EKG Results',
    [0, 1, 2],
    format_func=lambda x: {
        0: '0 - Kết quả điện tâm đồ bình thường',
        1: '1 - Bất thường sóng ST-T',
        2: '2 - Phì đại thất trái'
    }[x],
    help='Kết quả điện tâm đồ lúc nghỉ.'
)

Max_HR = st.number_input(
    'Maximum Heart Rate (Max HR)',
    min_value=60,
    max_value=250,
    value=150,
    help='Nhịp tim tối đa đạt được trong quá trình gắng sức.'
)

Exercise_angina = st.selectbox(
    'Exercise Angina',
    [0, 1],
    format_func=lambda x:
        '0 - Không xuất hiện đau thắt ngực khi vận động'
        if x == 0 else
        '1 - Có xuất hiện đau thắt ngực khi vận động',
    help='Đau thắt ngực xuất hiện khi vận động.'
)

ST_depression = st.number_input(
    'ST Depression',
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help='Mức độ chênh xuống của đoạn ST trên điện tâm đồ khi gắng sức. Giá trị càng cao có thể cho thấy nguy cơ thiếu máu cơ tim.'
)

Slope_of_ST = st.selectbox(
    'Slope of ST',
    [1, 2, 3],
    format_func=lambda x: {
        1: '1 - Đoạn ST dốc lên',
        2: '2 - Đoạn ST bằng phẳng',
        3: '3 - Đoạn ST dốc xuống'
    }[x],
    help='Độ dốc của đoạn ST trong điện tâm đồ khi gắng sức.'
)

Number_of_vessels_fluro = st.selectbox(
    'Number of Vessels Fluro',
    [0, 1, 2, 3],
    format_func=lambda x:
        f'{x} - Số lượng mạch máu lớn được nhuộm màu qua soi huỳnh quang',
    help='Số mạch máu lớn được quan sát thấy trong kỹ thuật chụp mạch vành.'
)

Thallium = st.selectbox(
    'Thallium',
    [3, 6, 7],
    format_func=lambda x: {
        3: '3 - Bình thường',
        6: '6 - Khiếm khuyết cố định (Fixed Defect)',
        7: '7 - Khiếm khuyết có thể hồi phục (Reversible Defect)'
    }[x],
    help='Kết quả xét nghiệm xạ hình tưới máu cơ tim bằng Thallium.'
)

# ==============================
# DỰ ĐOÁN
# ==============================
if st.button('🔍 Dự đoán'):

    # Tạo DataFrame đầu vào
    input_df = pd.DataFrame([{
        'Age': Age,
        'Sex': Sex,
        'Chest pain type': Chest_pain_type,
        'BP': BP,
        'Cholesterol': Cholesterol,
        'FBS over 120': FBS_over_120,
        'EKG results': EKG_results,
        'Max HR': Max_HR,
        'Exercise angina': Exercise_angina,
        'ST depression': ST_depression,
        'Slope of ST': Slope_of_ST,
        'Number of vessels fluro': Number_of_vessels_fluro,
        'Thallium': Thallium
    }])

    # Feature Engineering
    input_processed = create_features(input_df)

    # Dự đoán xác suất
    proba = model.predict_proba(input_processed)[0]
    prob_absence = proba[0] * 100
    prob_presence = proba[1] * 100

    # Dự đoán nhãn
    prediction = model.predict(input_processed)[0]

    # Hiển thị kết quả
    st.subheader('📊 Kết quả dự đoán')

    if prediction == 1:
        st.error(f'⚠️ Nguy cơ mắc bệnh tim: {prob_presence:.2f}%')
        st.write('**Kết luận:** Có khả năng mắc bệnh tim.')
    else:
        st.success(f'✅ Xác suất không mắc bệnh tim: {prob_absence:.2f}%')
        st.write(f'**Nguy cơ mắc bệnh tim:** {prob_presence:.2f}%')
        st.write('**Kết luận:** Khả năng không mắc bệnh tim.')

    # Progress bar
    st.progress(min(int(prob_presence), 100))

    # Biểu đồ
    chart_df = pd.DataFrame({
        'Tình trạng': ['Không mắc bệnh', 'Mắc bệnh'],
        'Xác suất (%)': [prob_absence, prob_presence]
    })
    st.bar_chart(chart_df.set_index('Tình trạng'))

    # # Bảng dữ liệu nhập
    # with st.expander('📋 Dữ liệu đã nhập'):
    #     st.dataframe(input_df)
# Bảng dữ liệu đã nhập + các đặc trưng mới được tạo

    with st.expander('📋 Dữ liệu đã nhập và các đặc trưng khác'):
        # Tạo dữ liệu sau feature engineering
        input_processed = create_features(input_df)

        # Hiển thị toàn bộ dữ liệu (bao gồm cả feature gốc và feature mới)
        st.dataframe(input_processed)
