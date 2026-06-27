import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
import datetime
from dotenv import load_dotenv
from PIL import Image


# Load environment variables
load_dotenv()

sys.path.append('src')
from chatbot import get_response

def get_language_response(user_input, drought_level, language):
    if language == 'हिंदी ':
        try:
            from hindi_support import get_hindi_response
            from chatbot import CROP_ADVICE
            return get_hindi_response(
                user_input, drought_level, CROP_ADVICE)
        except:
            return get_response(user_input, drought_level)
    elif language == 'मराठी ':
        try:
            from marathi_support import get_marathi_response
            return get_marathi_response(user_input, drought_level)
        except:
            return get_response(user_input, drought_level)
    else:
        return get_response(user_input, drought_level)
# Try importing torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

# Auto-generate data and models if not present
if not os.path.exists('data/rainfall_data.csv'):
    os.system('python create_dataset.py')
if not os.path.exists('models/ml_model.pkl'):
    os.system('python src/ml_model.py')
if TORCH_AVAILABLE and not os.path.exists('models/dl_model.pth'):
    os.system('python src/dl_model.py')

# Page Config
st.set_page_config(
    page_title="AI Drought Predictor & Farmer Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2d5a27, #4a9e3f);
        padding: 20px; border-radius: 12px;
        text-align: center; color: white !important;
        margin-bottom: 20px;
    }
    .main-header h1, .main-header p {
        color: white !important;
    }
    .chat-user {
        background:#1565c0;
        padding:10px 15px;
        border-radius:18px 18px 4px 18px;
        margin:8px 0;
        color: white !important;
    }
    .chat-bot {
        background:#2e7d32;
        padding:10px 15px;
        border-radius:18px 18px 18px 4px;
        margin:8px 0;
        color: white !important;
    }
    .chat-user b, .chat-user * { color: white !important; }
    .chat-bot b, .chat-bot * { color: white !important; }
    .weather-card {
        background: linear-gradient(135deg, #1565c0, #42a5f5);
        padding: 20px; border-radius: 12px;
        color: white; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Get API Key
def get_api_key():
    api_key = os.getenv("WEATHER_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["WEATHER_API_KEY"]
        except:
            api_key = ""
    return api_key

# Load ML Model
@st.cache_resource
def load_ml_model():
    with open('models/ml_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

# DL Model Architecture
if TORCH_AVAILABLE:
    class DroughtNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(DroughtNet, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, num_classes)
            )
        def forward(self, x):
            return self.network(x)

# Load DL Model
@st.cache_resource
def load_dl_model():
    if not TORCH_AVAILABLE:
        return None, None, None
    try:
        with open('models/dl_config.json', 'r') as f:
            config = json.load(f)
        model = DroughtNet(config['input_size'], config['num_classes'])
        model.load_state_dict(torch.load('models/dl_model.pth',
                                          map_location='cpu'))
        model.eval()
        with open('models/dl_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/dl_label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, scaler, le
    except:
        return None, None, None

# Drought style helper
def drought_style(level):
    return {
        'Severe':   ('drought-severe',   '🔴', '#d32f2f'),
        'Moderate': ('drought-moderate', '🟠', '#f57c00'),
        'Mild':     ('drought-mild',     '🟡', '#fbc02d'),
        'Normal':   ('drought-normal',   '🟢', '#388e3c'),
    }.get(level, ('drought-normal', '🟢', '#388e3c'))

# Sidebar
with st.sidebar:
    st.title("🌾 Navigation")
    page = st.radio("Go to", [
        "🏠 Home",
        "🔮 Drought Predictor",
        "🌧️ Rainfall Forecast",
	"🌿 Disease Detection",
	"🧮 Farm Calculator",
	"💰 Crop Price",
        "🤖 Farmer Chatbot",
        "📊 Data Insights",
	"🛰️ Satellite Analysis",
	"👩‍💻 About"
    ])
    st.markdown("---")
    st.markdown("**About This Project**")
    st.markdown("""
- 🌲 ML: Random Forest
- 🧠 DL: PyTorch Neural Net
- 💬 NLP: spaCy Chatbot
- 📊 Data: 15 Indian States
- 📅 Years: 2000–2023
    """)
    st.markdown("---")
    st.markdown("Built with ❤️ for Indian Farmers")

# ══════════════════════════════════════════════
# PAGE 1 - HOME
# ══════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header'>
        <h1>🌾 AI Drought Predictor & Farmer Assistant</h1>
        <p>Helping Indian farmers survive drought with Artificial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌧️ States Covered", "15")
    with col2:
        st.metric("📅 Years of Data", "24 (2000-2023)")
    with col3:
        st.metric("🌲 ML Accuracy", "95%+")
    with col4:
        st.metric("🧠 DL Accuracy", "88%+")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 What This App Does")
        st.markdown("""
- Predicts drought severity using ML + DL models
- Recommends crops suitable for current drought level
- AI Chatbot answers farmer queries in plain English
- Data Insights show rainfall trends across India
        """)
    with col2:
        st.subheader("🚀 How to Use")
        st.markdown("""
1. Go to **Drought Predictor** → enter your city name
2. Fetch live weather → get instant drought prediction
3. Visit **Farmer Chatbot** → ask any farming question
4. Explore **Data Insights** for rainfall trends
        """)

    st.markdown("---")
    st.subheader("🗺️ States Covered")
    states_data = {
        'State': ['Maharashtra','Rajasthan','Kerala','Punjab','Gujarat',
                  'Karnataka','Tamil Nadu','MP','UP','Bihar',
                  'West Bengal','Andhra Pradesh','Haryana','Odisha','Jharkhand'],
        'Avg Rainfall (mm)': [877,265,2005,650,489,785,855,750,810,900,
                               1485,735,568,1285,1040],
        'Risk Level': ['Moderate','Severe','Normal','Mild','Moderate',
                       'Mild','Mild','Moderate','Mild','Mild',
                       'Normal','Mild','Mild','Normal','Mild']
    }
    df_states = pd.DataFrame(states_data)
    fig = px.bar(df_states, x='State', y='Avg Rainfall (mm)',
                 color='Risk Level',
                 color_discrete_map={
                     'Normal':'#388e3c','Mild':'#fbc02d',
                     'Moderate':'#f57c00','Severe':'#d32f2f'
                 },
                 title='Average Annual Rainfall by State')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 2 - DROUGHT PREDICTOR
# ══════════════════════════════════════════════
elif page == "🔮 Drought Predictor":
    st.title("🔮 Drought Level Predictor")
    st.markdown("Enter your city name to get live weather or adjust manually!")

    ml_model, ml_scaler, ml_le = load_ml_model()
    dl_model, dl_scaler, dl_le = load_dl_model()

    # Weather API section
    st.subheader("🌤️ Fetch Live Weather Data")
    API_KEY = get_api_key()

    col_city, col_btn = st.columns([4, 1])
    with col_city:
        city_input = st.text_input(
            "🏙️ Enter your city name:",
            placeholder="e.g. Pune, Mumbai, Delhi, Jaipur"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("🌤️ Fetch Weather")

    # Initialize session state
    if 'annual_rain' not in st.session_state:
        st.session_state.annual_rain   = 500
        st.session_state.avg_temp      = 30
        st.session_state.avg_humidity  = 50
        st.session_state.soil_moisture = 30
        st.session_state.deficit       = 25

    if fetch_btn and city_input:
        from src.weather import (get_weather, get_rainfall_estimate,
                                  estimate_soil_moisture,
                                  estimate_rainfall_deficit)
        if not API_KEY:
            st.error("❌ API Key not found! Add WEATHER_API_KEY to .env file")
        else:
            with st.spinner(f"Fetching weather for {city_input}..."):
                weather, error = get_weather(city_input, API_KEY)

            if error:
                st.error(f"❌ {error}")
            else:
                rain_est    = get_rainfall_estimate(
                                  weather['humidity'],
                                  weather['description'])
                soil_est    = estimate_soil_moisture(
                                  weather['humidity'],
                                  weather['description'])
                deficit_est = estimate_rainfall_deficit(rain_est)

                st.session_state.annual_rain   = int(rain_est * 12)
                st.session_state.avg_temp      = int(weather['temperature'])
                st.session_state.avg_humidity  = int(weather['humidity'])
                st.session_state.soil_moisture = int(soil_est)
                st.session_state.deficit       = int(deficit_est)

                # Weather card
                st.markdown(f"""
                <div class='weather-card'>
                    <h3 style='color:white; margin:0'>
                        🌤️ {weather['city']} — Live Weather
                    </h3>
                    <p style='color:white; margin:5px 0'>
                        {weather['description']}
                    </p>
                    <div style='display:flex; gap:30px; margin-top:10px;
                                flex-wrap:wrap'>
                        <span style='color:white'>
                            🌡️ <b>{weather['temperature']}°C</b>
                        </span>
                        <span style='color:white'>
                            💧 <b>{weather['humidity']}%</b>
                        </span>
                        <span style='color:white'>
                            💨 <b>{weather['wind_speed']} km/h</b>
                        </span>
                        <span style='color:white'>
                            👁️ <b>{weather['visibility']} km</b>
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ Live weather loaded! Sliders updated automatically.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📥 Climate Data")
        annual_rain   = st.slider("🌧️ Annual Rainfall (mm)", 0, 3000,
                                   int(st.session_state.annual_rain))
        avg_temp      = st.slider("🌡️ Average Temperature (C)", 15, 45,
                                   min(45, max(15, int(
                                       st.session_state.avg_temp))))
        avg_humidity  = st.slider("💧 Average Humidity (%)", 10, 100,
                                   int(st.session_state.avg_humidity))
        soil_moisture = st.slider("🪱 Soil Moisture (%)", 5, 90,
                                   int(st.session_state.soil_moisture))
        deficit       = st.slider("📉 Rainfall Deficit (%)", 0, 80,
                                   int(st.session_state.deficit))

    with col2:
        st.subheader("📊 Deficit Gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=deficit,
            title={'text': "Rainfall Deficit (%)"},
            gauge={
                'axis': {'range': [0, 80]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 10],  'color': '#c8e6c9'},
                    {'range': [10, 25], 'color': '#fff9c4'},
                    {'range': [25, 40], 'color': '#ffe0b2'},
                    {'range': [40, 80], 'color': '#ffcdd2'},
                ]
            }
        ))
        gauge.update_layout(height=280)
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("---")
    if st.button("🔮 Predict Drought Level"):
        input_data = np.array([[annual_rain, avg_temp,
                                avg_humidity, soil_moisture, deficit]])

        # ML Prediction
        ml_input = ml_scaler.transform(input_data)
        ml_pred  = ml_model.predict(ml_input)[0]
        ml_proba = ml_model.predict_proba(ml_input)[0]
        ml_label = ml_le.inverse_transform([ml_pred])[0]
        ml_conf  = max(ml_proba) * 100

        # DL Prediction
        if TORCH_AVAILABLE and dl_model is not None:
            dl_input = torch.FloatTensor(dl_scaler.transform(input_data))
            with torch.no_grad():
                dl_out  = dl_model(dl_input)
                dl_prob = torch.softmax(dl_out, dim=1).numpy()[0]
                dl_pred = np.argmax(dl_prob)
            dl_label = dl_le.inverse_transform([dl_pred])[0]
            dl_conf  = max(dl_prob) * 100
        else:
            dl_label = ml_label
            dl_conf  = ml_conf

        # Show Results
        col1, col2 = st.columns(2)
        _, icon1, color1 = drought_style(ml_label)
        _, icon2, color2 = drought_style(dl_label)

        with col1:
            st.markdown(f"""
            <div style='border-left:6px solid {color1};
                        background:{color1}22;
                        padding:20px; border-radius:10px;'>
                <h3 style='color:{color1}; margin:0'>
                    {icon1} ML Model (Random Forest)
                </h3>
                <h2 style='color:{color1}; margin:10px 0'>
                    Drought Level: {ml_label}
                </h2>
                <p style='color:{color1}; margin:0; font-size:18px'>
                    Confidence: <b>{ml_conf:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='border-left:6px solid {color2};
                        background:{color2}22;
                        padding:20px; border-radius:10px;'>
                <h3 style='color:{color2}; margin:0'>
                    {icon2} DL Model (Neural Network)
                </h3>
                <h2 style='color:{color2}; margin:10px 0'>
                    Drought Level: {dl_label}
                </h2>
                <p style='color:{color2}; margin:0; font-size:18px'>
                    Confidence: <b>{dl_conf:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

# PDF Download button
        st.markdown("---")
        st.subheader("📄 Download Report")
        from src.report_generator import generate_report
        import io

        weather_data_for_report = {
            'temperature': avg_temp,
            'humidity':    avg_humidity,
            'wind_speed':  0,
            'description': 'Manual Input'
        }

        crops_list = CROP_ADVICE.get(
            ml_label.lower(), {}).get('crops', [])

        pdf_bytes = generate_report(
            city=st.session_state.get('city_name', 'Your Region'),
            weather_data=weather_data_for_report,
            ml_label=ml_label,
            ml_conf=ml_conf,
            dl_label=dl_label,
            dl_conf=dl_conf,
            crops=crops_list,
            deficit=deficit
        )

        st.download_button(
            label="📥 Download PDF Report",
            data=bytes(pdf_bytes),
            file_name=f"drought_report_{datetime.date.today()}.pdf",
            mime="application/pdf"
        )

        # Crop Recommendations
        st.markdown("---")
        st.subheader("🌱 Recommended Crops")
        from chatbot import CROP_ADVICE
        level_key = ml_label.lower()
        if level_key in CROP_ADVICE:
            info  = CROP_ADVICE[level_key]
            crops = info['crops']
            cols  = st.columns(len(crops))
            for i, crop in enumerate(crops):
                cols[i].success(f"✅ {crop}")
            st.info(f"💡 {info['advice'][0]}")
# WhatsApp Alert
        st.markdown("---")
        st.subheader("📱 Send WhatsApp Alert")
        col1, col2 = st.columns([3, 1])
        with col1:
            phone = st.text_input(
                "📱 WhatsApp Number (10 digits):",
                placeholder="9876543210"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            send_wa = st.button("📲 Send Alert")

        if send_wa and phone:
            if len(phone) == 10 and phone.isdigit():
                from src.whatsapp_alerts import (
                    send_whatsapp_alert)
                crops_for_wa = CROP_ADVICE.get(
                    ml_label.lower(), {}).get('crops', [])
                success, msg = send_whatsapp_alert(
                    phone, ml_label,
                    st.session_state.get('city_name',
                                         'Your Region'),
                    crops_for_wa
                )
                if success:
                    st.success(
                        "✅ WhatsApp alert sent successfully!")
                else:
                    st.error(f"❌ Failed: {msg}")
                    st.info("💡 Setup Twilio account at "
                            "twilio.com for WhatsApp alerts")
            else:
                st.error("❌ Enter valid 10 digit number!")

# ══════════════════════════════════════════════
# PAGE 3 - DISEASE DETECTION (CNN)
# ══════════════════════════════════════════════
elif page == "🌿 Disease Detection":
    st.title("🌿 Crop Disease Detection")
    st.markdown("Upload a leaf photo to detect diseases using AI!")

    st.info("🧠 Powered by CNN (Convolutional Neural Network) "
            "— Computer Vision AI")

    # Language for disease page
    lang = st.selectbox(
        "🌐 Language / भाषा:",
        ['English', 'हिंदी', 'मराठी'],
        key='disease_lang'
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of the crop leaf"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf",
                     use_column_width=True)

        # Sample images tip
        st.markdown("**💡 Tips for best results:**")
        st.markdown("""
- Use clear, well-lit photos
- Focus on the affected leaf
- Avoid blurry images
- Single leaf works best
        """)

    with col2:
        st.subheader("🔍 Detection Results")

        if uploaded_file:
            with st.spinner("🧠 CNN analyzing your crop..."):
                from src.disease_detection import (
                    predict_disease,
                    DISEASE_INFO,
                    get_disease_advice_marathi,
                    get_disease_advice_hindi
                )
                from PIL import Image as PILImage
                import time
                time.sleep(1)  # Simulate processing

                predicted, confidence, all_probs = predict_disease(image)
                info = DISEASE_INFO[predicted]

            # Result card
            st.markdown(f"""
            <div style='border-left: 6px solid {info["color"]};
                        background: {info["color"]}22;
                        padding: 20px; border-radius: 10px;
                        margin: 10px 0'>
                <h3 style='color: {info["color"]}; margin:0'>
                    🔬 Detected: {predicted}
                </h3>
                <p style='color: {info["color"]}; margin: 5px 0'>
                    Severity: <b>{info["severity"]}</b>
                </p>
                <p style='color: {info["color"]}; margin: 5px 0'>
                    Confidence: <b>{confidence:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Description
            st.markdown("**📋 Description:**")
            st.write(info['description'])

            # Treatment
            st.markdown("**💊 Treatment:**")
            st.error(f"🚨 {info['treatment']}")

            # Language specific advice
            if lang == 'मराठी':
                st.markdown("**🌾 मराठी सल्ला:**")
                st.info(get_disease_advice_marathi(predicted))
            elif lang == 'हिंदी':
                st.markdown("**🇮🇳 हिंदी सलाह:**")
                st.info(get_disease_advice_hindi(predicted))

            # Prevention tips
            st.markdown("**🛡️ Prevention Tips:**")
            for tip in info['prevention']:
                st.markdown(f"• {tip}")

            # All probabilities chart
            st.markdown("---")
            st.subheader("📊 Disease Probability Chart")
            import plotly.express as px
            prob_df = pd.DataFrame({
                'Disease': list(all_probs.keys()),
                'Probability': list(all_probs.values())
            })
            fig = px.bar(
                prob_df.sort_values('Probability'),
                x='Probability', y='Disease',
                orientation='h',
                color='Probability',
                color_continuous_scale='RdYlGn_r',
                title='Disease Detection Probabilities'
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center; padding:50px;
                        border: 2px dashed #4a9e3f;
                        border-radius:12px; color:#4a9e3f'>
                <h3>📸 Upload a leaf image</h3>
                <p>to detect diseases</p>
            </div>
            """, unsafe_allow_html=True)

            # Show disease reference
            st.markdown("---")
            st.subheader("📚 Disease Reference Guide")
            from src.disease_detection import DISEASE_INFO
            for disease, info in DISEASE_INFO.items():
                with st.expander(f"{disease} — {info['severity']} Severity"):
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Treatment:** {info['treatment']}")

# ══════════════════════════════════════════════
# PAGE 4 - FARM CALCULATOR
# ══════════════════════════════════════════════
elif page == "🧮 Farm Calculator":
    st.title("🧮 Smart Farm Calculator")
    st.markdown("Calculate water needs, fertilizer & costs!")

    tab1, tab2, tab3 = st.tabs([
        "💧 Water Calculator",
        "🌱 Fertilizer Calculator",
        "💰 Cost Estimator"
    ])

    with tab1:
        st.subheader("💧 Water Requirement Calculator")
        col1, col2 = st.columns(2)
        with col1:
            crop_w    = st.selectbox("Select Crop:", [
                'Rice', 'Wheat', 'Sugarcane', 'Cotton',
                'Maize', 'Bajra', 'Jowar', 'Moong',
                'Groundnut', 'Soybean'
            ])
            area_w    = st.number_input(
                "Farm Area (acres):", 1, 100, 5)
            soil_type = st.selectbox("Soil Type:", [
                'Sandy', 'Loamy', 'Clay', 'Black Cotton'
            ])
            season    = st.selectbox("Season:", [
                'Kharif (Monsoon)',
                'Rabi (Winter)',
                'Zaid (Summer)'
            ])

        with col2:
            # Water requirements per acre per season (liters)
            water_req = {
                'Rice':      {'Kharif (Monsoon)': 1200000,
                              'Rabi (Winter)':    1400000,
                              'Zaid (Summer)':    1600000},
                'Wheat':     {'Kharif (Monsoon)': 400000,
                              'Rabi (Winter)':    450000,
                              'Zaid (Summer)':    500000},
                'Sugarcane': {'Kharif (Monsoon)': 2000000,
                              'Rabi (Winter)':    2200000,
                              'Zaid (Summer)':    2500000},
                'Cotton':    {'Kharif (Monsoon)': 700000,
                              'Rabi (Winter)':    800000,
                              'Zaid (Summer)':    900000},
                'Maize':     {'Kharif (Monsoon)': 500000,
                              'Rabi (Winter)':    550000,
                              'Zaid (Summer)':    600000},
                'Bajra':     {'Kharif (Monsoon)': 300000,
                              'Rabi (Winter)':    350000,
                              'Zaid (Summer)':    400000},
                'Jowar':     {'Kharif (Monsoon)': 350000,
                              'Rabi (Winter)':    400000,
                              'Zaid (Summer)':    450000},
                'Moong':     {'Kharif (Monsoon)': 250000,
                              'Rabi (Winter)':    280000,
                              'Zaid (Summer)':    300000},
                'Groundnut': {'Kharif (Monsoon)': 500000,
                              'Rabi (Winter)':    550000,
                              'Zaid (Summer)':    600000},
                'Soybean':   {'Kharif (Monsoon)': 450000,
                              'Rabi (Winter)':    500000,
                              'Zaid (Summer)':    550000},
            }

            soil_factor = {
                'Sandy': 1.3, 'Loamy': 1.0,
                'Clay':  0.8, 'Black Cotton': 0.85
            }

            base_water  = water_req.get(
                crop_w, {}).get(season, 500000)
            total_water = (base_water * area_w *
                           soil_factor.get(soil_type, 1.0))

            st.metric("💧 Total Water Needed",
                      f"{total_water/1000:,.0f} KL")
            st.metric("📅 Per Day (120 days)",
                      f"{total_water/120/1000:,.1f} KL/day")
            st.metric("🚿 Drip System Saves",
                      f"{total_water*0.45/1000:,.0f} KL")

            if total_water > 1000000 * area_w:
                st.warning("⚠️ High water crop in drought!")
            else:
                st.success("✅ Suitable water requirement")

    with tab2:
        st.subheader("🌱 Fertilizer Calculator")
        col1, col2 = st.columns(2)
        with col1:
            crop_f  = st.selectbox("Select Crop:", [
                'Rice', 'Wheat', 'Maize', 'Cotton',
                'Sugarcane', 'Bajra', 'Soybean'
            ], key='fert_crop')
            area_f  = st.number_input(
                "Farm Area (acres):", 1, 100, 5,
                key='fert_area')
            soil_ph = st.slider("Soil pH:", 4.0, 9.0, 6.5)

        with col2:
            # NPK requirements per acre (kg)
            npk_req = {
                'Rice':      {'N': 80, 'P': 40, 'K': 40},
                'Wheat':     {'N': 60, 'P': 30, 'K': 30},
                'Maize':     {'N': 75, 'P': 35, 'K': 25},
                'Cotton':    {'N': 60, 'P': 30, 'K': 30},
                'Sugarcane': {'N': 150,'P': 60, 'K': 60},
                'Bajra':     {'N': 40, 'P': 20, 'K': 10},
                'Soybean':   {'N': 20, 'P': 40, 'K': 30},
            }

            req   = npk_req.get(crop_f,
                        {'N': 50, 'P': 25, 'K': 25})
            n_req = req['N'] * area_f
            p_req = req['P'] * area_f
            k_req = req['K'] * area_f

            st.metric("🟢 Nitrogen (N)", f"{n_req} kg")
            st.metric("🔴 Phosphorus (P)", f"{p_req} kg")
            st.metric("🔵 Potassium (K)", f"{k_req} kg")

            # Urea equivalent
            urea = n_req / 0.46
            dap  = p_req / 0.46
            mop  = k_req / 0.60

            st.markdown("**Fertilizer Bags Needed:**")
            st.info(f"Urea: {urea/50:.1f} bags (50kg)\n"
                    f"DAP: {dap/50:.1f} bags (50kg)\n"
                    f"MOP: {mop/50:.1f} bags (50kg)")

    with tab3:
        st.subheader("💰 Season Cost Estimator")
        col1, col2 = st.columns(2)
        with col1:
            crop_c = st.selectbox("Select Crop:", [
                'Rice', 'Wheat', 'Cotton', 'Sugarcane',
                'Maize', 'Bajra', 'Soybean', 'Groundnut'
            ], key='cost_crop')
            area_c = st.number_input(
                "Farm Area (acres):", 1, 100, 5,
                key='cost_area')
            irr_type = st.selectbox("Irrigation Type:", [
                'Flood', 'Drip', 'Sprinkler', 'Rainfed'
            ])

        with col2:
            # Cost components per acre (Rs)
            costs = {
                'Rice':      {'seed': 1500, 'fert': 3000,
                              'pest': 2000, 'labour': 8000},
                'Wheat':     {'seed': 1200, 'fert': 2500,
                              'pest': 1500, 'labour': 6000},
                'Cotton':    {'seed': 2000, 'fert': 3500,
                              'pest': 3000, 'labour': 8000},
                'Sugarcane': {'seed': 5000, 'fert': 5000,
                              'pest': 2000, 'labour': 12000},
                'Maize':     {'seed': 1500, 'fert': 2500,
                              'pest': 1500, 'labour': 5000},
                'Bajra':     {'seed': 800,  'fert': 1500,
                              'pest': 800,  'labour': 3500},
                'Soybean':   {'seed': 2000, 'fert': 2000,
                              'pest': 1500, 'labour': 4000},
                'Groundnut': {'seed': 3000, 'fert': 2500,
                              'pest': 1500, 'labour': 5000},
            }

            irr_cost = {
                'Flood': 4000, 'Drip': 2000,
                'Sprinkler': 2500, 'Rainfed': 500
            }

            c = costs.get(crop_c,
                    {'seed': 1500, 'fert': 2500,
                     'pest': 1500, 'labour': 5000})
            total_per_acre = (c['seed'] + c['fert'] +
                              c['pest'] + c['labour'] +
                              irr_cost.get(irr_type, 2000))
            total_cost = total_per_acre * area_c

            st.metric("🌱 Seed Cost",
                      f"Rs {c['seed']*area_c:,}")
            st.metric("🌿 Fertilizer Cost",
                      f"Rs {c['fert']*area_c:,}")
            st.metric("🐛 Pesticide Cost",
                      f"Rs {c['pest']*area_c:,}")
            st.metric("👷 Labour Cost",
                      f"Rs {c['labour']*area_c:,}")
            st.metric("💧 Irrigation Cost",
                      f"Rs {irr_cost.get(irr_type,2000)*area_c:,}")
            st.markdown("---")
            st.metric("💰 TOTAL COST",
                      f"Rs {total_cost:,}",
                      delta=f"Rs {total_per_acre:,}/acre")

# ══════════════════════════════════════════════
# PAGE 5 - CROP PRICE PREDICTOR
# ══════════════════════════════════════════════
elif page == "💰 Crop Price":
    st.title("💰 Crop Price & Profit Predictor")
    st.markdown("Find the most profitable crop for your farm!")

    from src.price_predictor import (predict_crop_price,
                                      get_best_crop_by_profit,
                                      CROP_BASE_PRICES)

    col1, col2, col3 = st.columns(3)
    with col1:
        drought = st.selectbox(
            "🌵 Drought Level:",
            ['normal', 'mild', 'moderate', 'severe'],
            index=1
        )
    with col2:
        acres = st.number_input(
            "🌾 Farm Size (acres):", 1, 100, 5)
    with col3:
        selected_crop = st.selectbox(
            "🌱 Select Crop:",
            list(CROP_BASE_PRICES.keys())
        )

    if st.button("💰 Calculate Profit"):
        result = predict_crop_price(
            selected_crop, drought, acres)

        if result:
            col1, col2, col3 = st.columns(3)
            profit_color = (
                "green" if result['profit'] > 0 else "red")

            col1.metric(
                "💵 Predicted Price",
                f"₹{result['predicted_price']:,.0f}/qtl",
                f"MSP: ₹{result['msp']:,}"
            )
            col2.metric(
                "📦 Expected Yield",
                f"{result['predicted_yield']} qtl"
            )
            col3.metric(
                "💰 Expected Profit",
                f"₹{result['profit']:,.0f}",
                f"ROI: {result['roi']}%"
            )

            # Revenue breakdown
            st.markdown("---")
            st.subheader("📊 Revenue Breakdown")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Revenue",
                        f"₹{result['total_revenue']:,.0f}")
            col2.metric("Total Cost",
                        f"₹{result['total_cost']:,.0f}")
            col3.metric("Net Profit",
                        f"₹{result['profit']:,.0f}")

        # Best crops comparison
        st.markdown("---")
        st.subheader("🏆 Most Profitable Crops Comparison")

        all_crops = list(CROP_BASE_PRICES.keys())
        best_crops = get_best_crop_by_profit(
            all_crops, drought, acres)

        import plotly.express as px
        df_crops = pd.DataFrame(best_crops[:10])
        fig = px.bar(
            df_crops,
            x='crop',
            y='profit',
            color='roi',
            title=f'Profit Comparison ({drought.title()} Drought, {acres} acres)',
            color_continuous_scale='RdYlGn',
            labels={'profit': 'Expected Profit (Rs)',
                    'crop': 'Crop', 'roi': 'ROI %'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 6 - RAINFALL FORECAST (LSTM)
# ══════════════════════════════════════════════
elif page == "🌧️ Rainfall Forecast":
    st.title("🌧️ AI Rainfall Forecaster")
    st.markdown("Predict next 6 months rainfall using Deep Learning LSTM!")

    st.info("🧠 Powered by PyTorch LSTM Neural Network — "
            "trained on 24 years of Indian rainfall data")

    col1, col2 = st.columns(2)
    with col1:
        df_states   = pd.read_csv('data/rainfall_data.csv')
        state_list  = ['All India'] + sorted(
                           df_states['State'].unique().tolist())
        selected    = st.selectbox("🗺️ Select State:", state_list)
        months_fwd  = st.slider("📅 Forecast Months:", 3, 12, 6)

    with col2:
        st.markdown("### 📋 What LSTM Predicts:")
        st.markdown("""
- Monthly rainfall for next 6 months
- Drought risk outlook
- Crop planning advice
- Water management tips
        """)

    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("🧠 LSTM model training & forecasting..."):
            from src.lstm_forecast import (forecast_rainfall,
                                            get_forecast_summary)
            state = None if selected == 'All India' else selected
            predictions, historical = forecast_rainfall(
                                          state, months_fwd)
            summary = get_forecast_summary(predictions)

        st.markdown("---")
        st.subheader("📊 Rainfall Forecast Results")

        # Outlook card
        st.markdown(f"""
        <div style='background: #1e3a1e; border-radius:12px;
                    padding:20px; margin:10px 0;
                    border-left: 6px solid #4a9e3f;'>
            <h3 style='color:white; margin:0'>
                {summary['outlook']}
            </h3>
            <p style='color:#90ee90; margin:10px 0'>
                {summary['advice']}
            </p>
            <p style='color:white; margin:0'>
                Total forecast: <b>{summary['total']} mm</b> |
                Monthly avg: <b>{summary['avg']} mm</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Forecast chart
        import plotly.graph_objects as go
        fig = go.Figure()

        # Historical
        hist_months = ['M-12','M-11','M-10','M-9','M-8',
                       'M-7','M-6','M-5','M-4','M-3',
                       'M-2','M-1']
        fig.add_trace(go.Scatter(
            x=hist_months,
            y=historical,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2196F3', width=2),
            marker=dict(size=6)
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=summary['month_names'],
            y=summary['predictions'],
            mode='lines+markers',
            name='LSTM Forecast',
            line=dict(color='#ff9800', width=3,
                      dash='dash'),
            marker=dict(size=8, symbol='star')
        ))

        fig.update_layout(
            title=f'Rainfall Forecast — {selected}',
            xaxis_title='Month',
            yaxis_title='Rainfall (mm)',
            legend=dict(x=0, y=1),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Monthly breakdown
        st.subheader("📅 Monthly Breakdown")
        cols = st.columns(min(6, months_fwd))
        for i in range(min(6, months_fwd)):
            rain = summary['predictions'][i]
            month = summary['month_names'][i]
            if rain < 30:
                color = "🔴"
            elif rain < 70:
                color = "🟡"
            else:
                color = "🟢"
            cols[i].metric(f"{color} {month}",
                           f"{rain:.0f}mm")

        # Crop advice based on forecast
        st.markdown("---")
        st.subheader("🌱 Crop Planning Based on Forecast")
        avg = summary['avg']
        if avg < 30:
            crops = "Bajra, Moth Bean, Amaranth"
            tip   = "Only plant severe drought resistant crops!"
        elif avg < 60:
            crops = "Bajra, Jowar, Moong, Cluster Bean"
            tip   = "Focus on drought tolerant crops."
        elif avg < 100:
            crops = "Maize, Soybean, Groundnut, Sunflower"
            tip   = "Moderate water crops recommended."
        else:
            crops = "Rice, Wheat, Sugarcane, Cotton"
            tip   = "Good rainfall — plant water-intensive crops!"

        st.success(f"✅ Recommended crops: **{crops}**")
        st.info(f"💡 {tip}")

# ══════════════════════════════════════════════
# PAGE 7 - FARMER CHATBOT
# ══════════════════════════════════════════════
elif page == "🤖 Farmer Chatbot":
    st.title("🤖 AI Farmer Chatbot")
    st.markdown("Ask me anything about farming, drought, crops, irrigation!")

    col1, col2 = st.columns(2)
    with col1:
        drought_level = st.selectbox(
            "🌵 Current Drought Level:",
            ['normal', 'mild', 'moderate', 'severe'],
            index=2
        )
    with col2:
        language = st.selectbox(
            "🌐 Select Language / भाषा निवडा:",
            ['English', 'हिंदी ', 'मराठी '],
            index=0
        )

    # Language info
    if language == 'हिंदी ':
        st.info("हिंदी मोड चालू है। हिंदी में पूछें!")
    elif language == 'मराठी ':
        st.info("मराठी मोड सुरू आहे। मराठीत विचारा!")
    else:
        st.info("English mode active. Ask in English!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            'role': 'bot',
            'text': "Namaste Kisan! 🌾 I am your AI Farming Assistant. Ask me about crops, irrigation, government schemes, or drought management!"
        })

    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(
                f"<div class='chat-user'>👨‍🌾 <b>You:</b> {msg['text']}</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='chat-bot'>🤖 <b>Assistant:</b> {msg['text']}</div>",
                unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Type your question:",
            placeholder="e.g. What crops should I grow in drought?",
            key="chat_input"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send = st.button("Send 📨")

    # Quick questions based on language
    st.markdown("**Quick Questions:**")
    qcols = st.columns(4)

    if language == 'हिंदी ':
        quick_qs = [
            "कौन सी फसल उगाएं?",
            "पानी कैसे बचाएं?",
            "सरकारी योजना?",
            "मिट्टी की देखभाल?"
        ]
    elif language == 'मराठी ':
        quick_qs = [
            "कोणते पीक घ्यावे?",
            "पाणी कसे वाचवावे?",
            "सरकारी योजना?",
            "माती व्यवस्थापन?"
        ]
    else:
        quick_qs = [
            "What crops to grow?",
            "How to save water?",
            "Government schemes?",
            "Soil management tips?"
        ]

    for i, q in enumerate(quick_qs):
        if qcols[i].button(q):
            response = get_language_response(
                q, drought_level, language)
            st.session_state.messages.append(
                {'role': 'user', 'text': q})
            st.session_state.messages.append(
                {'role': 'bot', 'text': response})
            st.rerun()

    if send and user_input:
        response = get_language_response(
            user_input, drought_level, language)
        st.session_state.messages.append(
            {'role': 'user', 'text': user_input})
        st.session_state.messages.append(
            {'role': 'bot', 'text': response})
        st.rerun()

# ══════════════════════════════════════════════
# PAGE 8 - DATA INSIGHTS
# ══════════════════════════════════════════════
elif page == "📊 Data Insights":
    st.title("📊 India Rainfall & Drought Insights")

    df = pd.read_csv('data/rainfall_data.csv')
    df['Drought_Level'] = df['Drought_Level'].replace({'Severe': 'Moderate'})

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(
            df, names='Drought_Level',
            title='Drought Level Distribution (2000-2023)',
            color_discrete_map={
                'Normal':'#388e3c',
                'Mild':'#fbc02d',
                'Moderate':'#f57c00'
            })
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        trend = df.groupby('Year')['Annual_Rainfall_mm'].mean().reset_index()
        fig2  = px.line(
            trend, x='Year', y='Annual_Rainfall_mm',
            title='India Average Rainfall Trend (2000-2023)',
            markers=True)
        fig2.update_traces(line_color='#2196F3', line_width=2)
        st.plotly_chart(fig2, use_container_width=True)

    state_avg = df.groupby('State')['Annual_Rainfall_mm'].mean().reset_index()
    fig3 = px.bar(
        state_avg.sort_values('Annual_Rainfall_mm'),
        x='Annual_Rainfall_mm', y='State',
        orientation='h',
        title='Average Annual Rainfall by State (mm)',
        color='Annual_Rainfall_mm',
        color_continuous_scale='Blues')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🔍 State-wise Drought History")
    selected_state = st.selectbox("Select State:",
                                   sorted(df['State'].unique()))
    state_df = df[df['State'] == selected_state]

    fig4 = px.bar(
        state_df, x='Year', y='Annual_Rainfall_mm',
        color='Drought_Level',
        color_discrete_map={
            'Normal':'#388e3c',
            'Mild':'#fbc02d',
            'Moderate':'#f57c00'
        },
        title=f'{selected_state} — Yearly Rainfall & Drought Level')
    st.plotly_chart(fig4, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Rainfall",
                f"{state_df['Annual_Rainfall_mm'].mean():.0f} mm")
    col2.metric("Drought Years",
                f"{(state_df['Drought_Level'] != 'Normal').sum()}")
    col3.metric("Worst Deficit",
                f"{state_df['Rainfall_Deficit_percent'].max():.1f}%")

# ══════════════════════════════════════════════
# PAGE 9 - SATELLITE ANALYSIS
# ══════════════════════════════════════════════
elif page == "🛰️ Satellite Analysis":
    st.title("🛰️ Satellite Drought Zone Analysis")
    st.markdown("Analyze satellite images to detect drought zones!")

    st.info("🧠 Uses Computer Vision to analyze "
            "vegetation health from satellite imagery")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Upload Satellite Image")
        sat_image = st.file_uploader(
            "Upload satellite/aerial image",
            type=['jpg', 'jpeg', 'png'],
            key='satellite'
        )
        st.markdown("""
        **What to upload:**
        - Google Earth screenshot
        - Drone aerial photo
        - Any top-down farm image
        - Satellite imagery
        """)

    with col2:
        st.subheader("🔍 Analysis Results")
        if sat_image:
            image = Image.open(sat_image)
            img_array = np.array(
                image.resize((256, 256)))

            with st.spinner("🛰️ Analyzing satellite image..."):
                import time
                time.sleep(2)

                # NDVI-like analysis using RGB
                if len(img_array.shape) == 3:
                    r = img_array[:,:,0].astype(float)
                    g = img_array[:,:,1].astype(float)
                    b = img_array[:,:,2].astype(float)

                    # Vegetation index
                    veg_index  = g / (r + g + b + 1)
                    # Dry index
                    dry_index  = r / (g + b + 1)
                    # Water index
                    water_index = b / (r + g + 1)

                    healthy_pct = float(
                        (veg_index > 0.38).mean() * 100)
                    dry_pct     = float(
                        (dry_index > 0.45).mean() * 100)
                    water_pct   = float(
                        (water_index > 0.40).mean() * 100)
                    stressed_pct = max(0, 100 - healthy_pct
                                       - dry_pct - water_pct)
                else:
                    healthy_pct  = 30.0
                    dry_pct      = 45.0
                    water_pct    = 5.0
                    stressed_pct = 20.0

            # Show original image
            st.image(image, caption="Uploaded Image",
                     use_column_width=True)

            # Zone analysis
            st.markdown("### 🗺️ Zone Analysis")
            col_a, col_b = st.columns(2)
            col_a.metric("🟢 Healthy Vegetation",
                         f"{healthy_pct:.1f}%")
            col_b.metric("🔴 Dry/Drought Zone",
                         f"{dry_pct:.1f}%")
            col_a.metric("💧 Water Bodies",
                         f"{water_pct:.1f}%")
            col_b.metric("🟡 Stressed Vegetation",
                         f"{stressed_pct:.1f}%")

            # Pie chart
            import plotly.express as px
            fig = px.pie(
                values=[healthy_pct, dry_pct,
                        water_pct, stressed_pct],
                names=['Healthy', 'Dry Zone',
                       'Water', 'Stressed'],
                color_discrete_map={
                    'Healthy':  '#388e3c',
                    'Dry Zone': '#d32f2f',
                    'Water':    '#1565c0',
                    'Stressed': '#f57c00'
                },
                title='Land Zone Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Overall assessment
            if dry_pct > 50:
                st.error("🔴 CRITICAL: More than 50% area "
                         "is drought affected!")
            elif dry_pct > 30:
                st.warning("🟠 WARNING: Significant drought "
                           "zones detected!")
            else:
                st.success("🟢 GOOD: Vegetation looks "
                           "relatively healthy!")

        else:
            st.markdown("""
            <div style='text-align:center; padding:50px;
                        border: 2px dashed #4a9e3f;
                        border-radius:12px'>
                <h3>🛰️ Upload a satellite image</h3>
                <p>to analyze drought zones</p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 10 - ABOUT
# ══════════════════════════════════════════════
elif page == "👩‍💻 About":
    st.title("👩‍💻 About This Project")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #2d5a27, #4a9e3f);
                    padding: 30px; border-radius: 12px;
                    text-align: center; color: white;'>
            <h2 style='color:white'>Aditi Kamble</h2>
            <p style='color:white'>AI/ML Developer</p>
            <p style='color:white'>Maharashtra, India 🇮🇳</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🔗 Connect With Me")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Aditi--Kamble-black?logo=github)](https://github.com/Aditi-Kamble)")
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-profile)")
        st.markdown("[![Live App](https://img.shields.io/badge/Live-App-green?logo=streamlit)](https://aditi-kamble-drought-predictor.streamlit.app)")

    with col2:
        st.markdown("### 🌾 Project Overview")
        st.markdown("""
        This project was built to help Indian farmers navigate drought
        conditions using Artificial Intelligence. With India facing
        below-normal rainfall, this tool provides:

        - Real-time drought severity prediction
        - Crop recommendations based on water availability
        - AI-powered farmer advisory chatbot
        - Rainfall forecasting for next 6 months
        - Crop disease detection from leaf images
        """)

        st.markdown("### 🛠️ Tech Stack")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            **AI/ML**
            - Scikit-learn
            - PyTorch
            - CNN
            - LSTM
            """)
        with col_b:
            st.markdown("""
            **NLP**
            - Google Gemini
            - Hindi Support
            - Marathi Support
            - Intent Detection
            """)
        with col_c:
            st.markdown("""
            **Web/API**
            - Streamlit
            - Plotly
            - OpenWeather API
            - GitHub
            """)

    st.markdown("---")
    st.markdown("### 📊 Project Stats")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌲 ML Accuracy", "95%+")
    col2.metric("🧠 DL Accuracy", "88%+")
    col3.metric("🗺️ States", "15")
    col4.metric("📅 Years Data", "24")
    col5.metric("🌐 Languages", "3")

    st.markdown("---")
    st.markdown("### 🎯 Project Modules")
    modules = {
        "🔮 Drought Predictor": "ML + DL models predict drought severity from climate data",
        "🌧️ Rainfall Forecast": "LSTM neural network forecasts next 6 months rainfall",
        "🌿 Disease Detection": "CNN analyzes leaf images to detect crop diseases",
        "🤖 Farmer Chatbot":   "Gemini AI answers farming questions in 3 languages",
        "📊 Data Insights":    "Interactive charts showing 24 years of rainfall trends",
    }
    for module, desc in modules.items():
        st.markdown(f"**{module}** — {desc}")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:20px;
                background:#1e3a1e; border-radius:12px;'>
        <p style='color:#90ee90; font-size:18px'>
            Built with ❤️ for Indian Farmers 🌾
        </p>
        <p style='color:white'>
            © 2026 Aditi Kamble | AI Drought Predictor
        </p>
    </div>
    """, unsafe_allow_html=True)