import streamlit as st
import joblib
import numpy as np
from keras.saving import load_model as load_keras_model
import pandas as pd
import numpy as np

st.markdown("# Dự báo nồng độ bụi PM<sub>2.5</sub> tại thành phố Hồ Chí Minh", unsafe_allow_html=True) #🌫️
st.image('pm2-5-icon.jpg', width=250)
st.markdown("Nồng độ PM<sub>2.5</sub> được dự báo dựa trên các thông số khí tượng", unsafe_allow_html=True)

predict_case = st.selectbox("Lựa chọn trường hợp dự đoán", 
                         ["Mô phỏng", "Sớm 1 ngày", "Sớm 3 ngày", "Sớm 5 ngày", "Sớm 7 ngày"])


model_file = {'Mô phỏng': 'S_ANN_6.keras', 
           'Sớm 1 ngày': 'P1_ANN_4.keras',
           'Sớm 3 ngày': 'P3_ANN_5.keras', 
           'Sớm 5 ngày': 'P5_ANN_3.keras', 
           'Sớm 7 ngày': 'P7_ANN_2.keras'
           }
model_info = {}

################################################################
# Create input columns
col1, col2 = st.columns(2)
if predict_case == "Mô phỏng":
    with col1:
        humidity = st.number_input("Độ ẩm (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
        temperature = st.number_input("Nhiệt độ (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5)
        wind_speed = st.number_input("Tốc độ gió (m/s)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    with col2:
        rainfall = st.number_input("Lượng mưa (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
        evaporation = st.number_input("Độ bốc hơi (mm)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        sunshine = st.number_input("Số giờ nắng (giờ)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
    model_info = (model_file.get(predict_case), 
                  [humidity, temperature, wind_speed, rainfall, evaporation, sunshine])

elif predict_case == "Sớm 1 ngày":
    with col1:
        humidity = st.number_input("Độ ẩm (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
        temperature = st.number_input("Nhiệt độ (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5)
    with col2:
        wind_speed = st.number_input("Tốc độ gió (m/s)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        evaporation = st.number_input("Độ bốc hơi (mm)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    model_info = (model_file.get(predict_case), [humidity, temperature, wind_speed, evaporation])

elif predict_case == "Sớm 3 ngày":
    with col1:
        humidity = st.number_input("Độ ẩm (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
        temperature = st.number_input("Nhiệt độ (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5)
        wind_speed = st.number_input("Tốc độ gió (m/s)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    with col2:
        rainfall = st.number_input("Lượng mưa (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
        evaporation = st.number_input("Độ bốc hơi (mm)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    model_info = (model_file.get(predict_case), [humidity, temperature, wind_speed, rainfall, evaporation])

elif predict_case == "Sớm 5 ngày":
    with col1:
        humidity = st.number_input("Độ ẩm (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
        temperature = st.number_input("Nhiệt độ (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5)
    with col2:
        evaporation = st.number_input("Độ bốc hơi (mm)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    model_info = (model_file.get(predict_case), [humidity, temperature, evaporation])

elif predict_case == "Sớm 7 ngày":
    with col1:
        humidity = st.number_input("Độ ẩm (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
        temperature = st.number_input("Nhiệt độ (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5)
    model_info = (model_file.get(predict_case), [humidity, temperature])
################################################################


def load_model(model_name):
    # Load the model and scalers
    if model_name.endswith('.keras'):
        model = load_keras_model(f'models/{model_name}')
    else:
        model = joblib.load(f'models/{model_name}')
        
    x_scaler_file = f'models/{model_name.split(".")[0]}_scalerX.pkl'
    y_scaler_file = f'models/{model_name.split(".")[0]}_scalery.pkl'
    scalerX = joblib.load(x_scaler_file)  
    scalery = joblib.load(y_scaler_file)  
    return model, scalerX, scalery

def make_prediction(model_info):
    # Get model, scaler and input features
    model_name = model_info[0]
    input_features = np.array([model_info[1]])
    model, scalerX, scalery = load_model(model_name)
    
    # Make prediction
    scaled_input = scalerX.transform(input_features) # Scale the input features
    scaled_prediction = model.predict(scaled_input) # Make prediction
    prediction = scalery.inverse_transform(scaled_prediction.reshape(-1, 1)) # Inverse transform the prediction
    return prediction[0][0]


def calculate_aqi_from_pm25(pm25_pred):
    """
    Tính AQI từ giá trị PM2.5 dự đoán 
    Dựa theo công thức trong quyết định 1459/QĐ-TCMT năm 2019 của Tổng cục Môi trường
    """
    # Bảng giá trị I và BP
    aqi_ref = [
        (0, 50, 0, 25),
        (50, 100, 25, 50),
        (100, 150, 50, 80),
        (150, 200, 80, 150),
        (200, 300, 150, 250),
        (300, 400, 250, 350),
        (400, 500, 350, 500)
    ]
    
    if pm25_pred >= 500: 
        AQI_PM25 = ((500 - 400) / (pm25_pred - 350)) * (pm25_pred - 350) + 400
        return round(AQI_PM25)
    
    for I_low, I_high, BP_low, BP_high in aqi_ref:
        if BP_low <= pm25_pred <= BP_high:
            AQI_PM25 = ((I_high - I_low) / (BP_high - BP_low)) * (pm25_pred - BP_low) + I_low
            return round(AQI_PM25)
    return None  # trường hợp pm25_pred < 0

def interpret_aqi(aqi_value):
    """Diễn giải AQI dựa trên bảng ảnh hưởng sức khỏe và hoạt động khuyến nghị."""
    if 0 <= aqi_value <= 50:
        return {
            "mức độ": "Tốt",
            "ảnh hưởng sức khỏe": "Chất lượng không khí tốt, không ảnh hưởng tới sức khỏe.",
            "khuyến nghị (bình thường)": "Tự do thực hiện các hoạt động ngoài trời.",
            "khuyến nghị (nhạy cảm)": "Tự do thực hiện các hoạt động ngoài trời."
        }
    elif 51 <= aqi_value <= 100:
        return {
            "mức độ": "Trung bình",
            "ảnh hưởng sức khỏe": """Chất lượng không khí ở mức chấp nhận được. 
                                     Tuy nhiên, đối với những người nhạy cảm (người già, trẻ em, người mắc các bệnh hô hấp, tim mạch,...) có thể chịu những tác động nhất định tới sức khỏe.""",
            "khuyến nghị (bình thường)": "Tự do thực hiện các hoạt động ngoài trời.",
            "khuyến nghị (nhạy cảm)": "Nên theo dõi triệu chứng như ho hoặc khó thở, nhưng vẫn có thể hoạt động bên ngoài."
        }
    elif 101 <= aqi_value <= 150:
        return {
            "mức độ": "Kém",
            "ảnh hưởng sức khỏe": "Những người nhạy cảm gặp phải các vấn đề về sức khỏe, những người bình thường ít ảnh hưởng.",
            "khuyến nghị (bình thường)": """Những người thấy có triệu chứng đau mắt, ho hoặc đau họng,... nên cân nhắc giảm các hoạt động ngoài trời.
                                            Đối với học sinh, có thể hoạt động bên ngoài, nhưng nên giảm bớt việc tập thể dục kéo dài.""",
            "khuyến nghị (nhạy cảm)": """Nên giảm các hoạt động mạnh và giảm thời gian hoạt động ngoài trời.
                                         Những người mắc bệnh hen suyễn có thể cần sử dụng thuốc thường xuyên hơn."""
        }
    elif 151 <= aqi_value <= 200:
        return {
            "mức độ": "Xấu",
            "ảnh hưởng sức khỏe": "Những người bình thường bắt đầu có các ảnh hưởng tới sức khỏe, nhóm người nhạy cảm có thể gặp những vấn đề sức khỏe nghiêm trọng hơn.",
            "khuyến nghị (bình thường)": "Mọi người nên giảm các hoạt động mạnh khi ở ngoài trời, tránh tập thể dục kéo dài và nghỉ ngơi nhiều hơn trong nhà.",
            "khuyến nghị (nhạy cảm)": "Nên ở trong nhà và giảm hoạt động mạnh. Nếu cần thiết phải ra ngoài, hãy đeo khẩu trang đạt tiêu chuẩn."
        }
    elif 201 <= aqi_value <= 300:
        return {
            "mức độ": "Rất xấu",
            "ảnh hưởng sức khỏe": "Cảnh báo hưởng tới sức khỏe: mọi người bị ảnh hưởng tới sức khỏe nghiêm trọng hơn.",
            "khuyến nghị (bình thường)": """Mọi người hạn chế tối đa các hoạt động ngoài trời và chuyển tất cả các hoạt động vào trong nhà.
                                            Nếu cần thiết phải ra ngoài, hãy đeo khẩu trang đạt tiêu chuẩn.""",
            "khuyến nghị (nhạy cảm)": "Nên ở trong nhà và giảm hoạt động mạnh."
        }
    elif 301 <= aqi_value <= 500:
        return {
            "mức độ": "Nguy hại",
            "ảnh hưởng sức khỏe": "Cảnh báo khẩn cấp về sức khỏe: Toàn bộ dân số bị ảnh hưởng tới sức khỏe tới mức nghiêm trọng.",
            "khuyến nghị (bình thường)": "Mọi người nên ở trong nhà, đóng cửa ra vào và cửa sổ. Nếu cần thiết phải ra ngoài, hãy đeo khẩu trang đạt tiêu chuẩn.",
            "khuyến nghị (nhạy cảm)": "Mọi người nên ở trong nhà, đóng cửa ra vào và cửa sổ. Nếu cần thiết phải ra ngoài, hãy đeo khẩu trang đạt tiêu chuẩn."
        }
    else:
        return {"mức độ": "Không xác định", "ảnh hưởng sức khỏe": "Ngoài phạm vi đo lường."}

def get_aqi_color(level):
    colors = {
        "Tốt": "#00e400",         # Xanh lá
        "Trung bình": "#ffff00",   # Vàng
        "Kém": "#ff7e00",          # Cam
        "Xấu": "#ff0000",          # Đỏ
        "Rất xấu": "#8f3f97",      # Tím
        "Nguy hại": "#7e0023"      # Nâu
    }
    return colors.get(level, "#ffffff")  # Mặc định là trắng nếu không tìm thấy


################################################################
# Make prediction
st.markdown("""
    <style>
    div.stButton > button {streamlit 
        border: 2px solid #666869;
        border-radius: 10px;
        font-weight: bold;
        font-size: 200px;
        padding: 12px 24px;
        background-color: #e3e4e6;
        margin: auto;
        display: block;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #b9c4c7;  
        color: white;  /* White text on hover */
    }
    </style>
""", unsafe_allow_html=True)

# Centered button
col1, col2, col3 = st.columns(3)
with col2:
    predict_button = st.button("**Kết quả**")

if not predict_button:
    st.divider()
    st.markdown("**Chú thích**")
    st.markdown(
            """
            1. Ứng dụng là một phần của luận án của NCS Nguyễn Phúc Hiếu với đề tài: \n
            *“Nghiên cứu ứng dụng phương pháp học máy và học sâu dự báo nồng độ bụi PM<sub>2.5</sub> - Trường hợp nghiên cứu cho thành phố Hồ Chí Minh”* \n
            2. Nguồn cung cấp dữ liệu: Trạm đo Lãnh sự quán Hoa Kỳ tại TP.HCM và Trạm khí tượng Tân Sơn Hòa
            3. Phương pháp tính AQI: Theo Quyết định 1459/QĐ-TCMT năm 2019 về hướng dẫn kỹ thuật tính toán và công bố chỉ số chất lượng không khí Việt Nam (VN_AQI) do Tổng cục trưởng Tổng cục Môi trường ban hành.
            4. Thông tin liên hệ: NCS Nguyễn Phúc Hiếu - phuchieu50@gmail.com
            """, 
            unsafe_allow_html=True
        )

elif predict_button: 
    pm25_pred = make_prediction(model_info)
    # st.divider()
    # st.subheader("Kết quả dự đoán")
    if predict_case != "Mô phỏng":
        st.markdown(f"**Nồng độ PM<sub>2.5</sub> (dự báo {predict_case.lower()})**", unsafe_allow_html=True)
    else:
        st.markdown(f"**Nồng độ PM<sub>2.5</sub> ({predict_case.lower()})**", unsafe_allow_html=True)
    st.success(f"{pm25_pred:.2f} μg/m³")
    
    # Tính AQI từ PM2.5 dự đoán
    aqi_value = calculate_aqi_from_pm25(pm25_pred)
    st.markdown("**Giá trị AQI tương ứng**")
    st.success(f"{aqi_value}")
    
    # Diễn giải kết quả AQI
    result = interpret_aqi(aqi_value)
    for key, value in result.items():
        if key == "mức độ":
            aqi_level = value 
            aqi_color = get_aqi_color(aqi_level)
            st.markdown("**Chất lượng không khí**")
            st.markdown(
            f"""
            <div style="background-color: {aqi_color}; padding: 10px; border-radius: 5px; text-align: center; color: black; font-weight: bold;">
                {aqi_level}
            </div><br>
            
            """,
            unsafe_allow_html=True
        )
            
        if key == "ảnh hưởng sức khỏe":
            st.markdown(f"👥 **Mức độ ảnh hưởng đến sức khỏe con người**")
            st.markdown(
            f"""
            <div style="background-color: #e8f3fc; padding: 10px; border-radius: 5px; color: black; border-left: 5px;">
                {value}
            </div><br>
            """,
            unsafe_allow_html=True
            )

        if key == "khuyến nghị (bình thường)":
            st.markdown(f"🏋️ **Khuyến nghị hoạt động cho người bình thường**")
            st.markdown(
            f"""
            <div style="background-color: #e8f3fc; padding: 10px; border-radius: 5px; color: black; border-left: 5px;">
                {value}
            </div><br>
            """,
            unsafe_allow_html=True
            )
        if key == "khuyến nghị (nhạy cảm)":
            st.markdown(f"👉 **Khuyến nghị hoạt động cho người nhạy cảm**")
            st.markdown(
            f"""
            <div style="background-color: #e8f3fc; padding: 10px; border-radius: 5px; color: black; border-left: 5px;">
                {value}
            </div><br>
            """,
            unsafe_allow_html=True
            )
    
    ###########################
    # st.subheader("Bảng tham chiếu chất lượng không khí")
    st.markdown("**Bảng tham chiếu chất lượng không khí**")
    # Dữ liệu bảng AQI
    aqi_data = {
        "Khoảng giá trị AQI": ["0 - 50", "51 - 100", "101 - 150", "151 - 200", "201 - 300", "301 - 500"],
        "Chất lượng không khí": ["Tốt", "Trung bình", "Kém", "Xấu", "Rất xấu", "Nguy hại"]
    }
    rgb_colors = ["0,228,0", "255,255,0", "255,126,0", "255,0,0", "143,63,151", "126,0,35"]

    aqi_df = pd.DataFrame(aqi_data)

    # chuyển đổi RGB thành CSS
    def apply_style(val, color):
        return f"background-color: rgb({color}); color: black; font-weight: bold; text-align: center;"

    # Áp dụng màu sắc cho từng ô
    styled_df = aqi_df.style.apply(lambda row: [apply_style(row[col], rgb_colors[row.name]) for col in aqi_df.columns], axis=1)
    # Hiển thị bảng
    st.dataframe(styled_df, hide_index=True, width=400)
    st.divider()
    st.markdown("**Chú thích**")
    st.markdown(
        """
        1. Ứng dụng là một phần của luận án của NCS Nguyễn Phúc Hiếu với đề tài: \n
        *“Nghiên cứu ứng dụng phương pháp học máy và học sâu dự báo nồng độ bụi PM<sub>2.5</sub> - Trường hợp nghiên cứu cho thành phố Hồ Chí Minh”* \n
        2. Nguồn cung cấp dữ liệu: Trạm đo Lãnh sự quán Hoa Kỳ tại TP.HCM và Trạm khí tượng Tân Sơn Hòa
        3. Phương pháp tính AQI: Theo Quyết định 1459/QĐ-TCMT năm 2019 về hướng dẫn kỹ thuật tính toán và công bố chỉ số chất lượng không khí Việt Nam (VN_AQI) do Tổng cục trưởng Tổng cục Môi trường ban hành.
        4. Thông tin liên hệ: NCS Nguyễn Phúc Hiếu - phuchieu50@gmail.com
        """,
        unsafe_allow_html=True
    )
################################################################





