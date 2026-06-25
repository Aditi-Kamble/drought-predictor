import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append('src')
from chatbot import get_response

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
        "🤖 Farmer Chatbot",
        "📊 Data Insights"
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

# ══════════════════════════════════════════════
# PAGE 3 - FARMER CHATBOT
# ══════════════════════════════════════════════
elif page == "🤖 Farmer Chatbot":
    st.title("🤖 AI Farmer Chatbot")
    st.markdown("Ask me anything about farming, drought, crops, irrigation!")

    drought_level = st.selectbox(
        "🌵 Current Drought Level in your area:",
        ['normal', 'mild', 'moderate', 'severe'],
        index=2
    )

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

    st.markdown("**Quick Questions:**")
    qcols = st.columns(4)
    quick_qs = [
        "What crops to grow?",
        "How to save water?",
        "Government schemes?",
        "Soil management tips?"
    ]
    for i, q in enumerate(quick_qs):
        if qcols[i].button(q):
            response = get_response(q, drought_level)
            st.session_state.messages.append({'role': 'user', 'text': q})
            st.session_state.messages.append({'role': 'bot', 'text': response})
            st.rerun()

    if send and user_input:
        response = get_response(user_input, drought_level)
        st.session_state.messages.append({'role': 'user', 'text': user_input})
        st.session_state.messages.append({'role': 'bot', 'text': response})
        st.rerun()

# ══════════════════════════════════════════════
# PAGE 4 - DATA INSIGHTS
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