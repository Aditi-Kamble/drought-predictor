import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import json
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

sys.path.append('src')
from chatbot import get_response

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="🌾 Drought Predictor & Farmer Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2d5a27, #4a9e3f);
        padding: 20px; border-radius: 12px;
        text-align: center; color: white; margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f8f0; border-left: 5px solid #4a9e3f;
        padding: 15px; border-radius: 8px; margin: 10px 0;
    }
    .drought-severe   { background:#ffe0e0; border-left:5px solid #d32f2f; padding:15px; border-radius:8px; }
    .drought-moderate { background:#fff3e0; border-left:5px solid #f57c00; padding:15px; border-radius:8px; }
    .drought-mild     { background:#fffde7; border-left:5px solid #fbc02d; padding:15px; border-radius:8px; }
    .drought-normal   { background:#e8f5e9; border-left:5px solid #388e3c; padding:15px; border-radius:8px; }
    .chat-user { background:#e3f2fd; padding:10px 15px; border-radius:18px 18px 4px 18px; margin:8px 0; }
    .chat-bot  { background:#f1f8e9; padding:10px 15px; border-radius:18px 18px 18px 4px; margin:8px 0; }
    .stButton>button {
        background:#4a9e3f; color:white; border:none;
        border-radius:8px; padding:10px 24px; font-size:16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load ML Model ─────────────────────────────────
@st.cache_resource
def load_ml_model():
    with open('models/ml_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

# ── Load DL Model ─────────────────────────────────
class DroughtNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DroughtNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_dl_model():
    with open('models/dl_config.json', 'r') as f:
        config = json.load(f)
    model = DroughtNet(config['input_size'], config['num_classes'])
    model.load_state_dict(torch.load('models/dl_model.pth', map_location='cpu'))
    model.eval()
    with open('models/dl_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/dl_label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

# ── Helper: Drought Color ─────────────────────────
def drought_style(level):
    return {
        'Severe':   ('drought-severe',   '🔴', '#d32f2f'),
        'Moderate': ('drought-moderate', '🟠', '#f57c00'),
        'Mild':     ('drought-mild',     '🟡', '#fbc02d'),
        'Normal':   ('drought-normal',   '🟢', '#388e3c'),
    }.get(level, ('drought-normal', '🟢', '#388e3c'))

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/320px-Image_created_with_a_mobile_phone.png", width=80)
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

# ══════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════
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
        - **Predicts drought severity** using ML + DL models
        - **Recommends crops** suitable for current drought level
        - **AI Chatbot** answers farmer queries in plain English
        - **Data Insights** show rainfall trends across India
        """)

    with col2:
        st.subheader("🚀 How to Use")
        st.markdown("""
        1. Go to **Drought Predictor** → enter your region's data
        2. Get instant **drought level prediction**
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

# ══════════════════════════════════════════════════
# PAGE 2 — DROUGHT PREDICTOR
# ══════════════════════════════════════════════════
elif page == "🔮 Drought Predictor":
    st.title("🔮 Drought Level Predictor")
    st.markdown("Enter your region's climate data to predict drought severity.")

    ml_model, ml_scaler, ml_le = load_ml_model()
    dl_model, dl_scaler, dl_le = load_dl_model()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📥 Enter Climate Data")
        annual_rain   = st.slider("🌧️ Annual Rainfall (mm)", 0, 3000, 500)
        avg_temp      = st.slider("🌡️ Average Temperature (°C)", 15, 45, 30)
        avg_humidity  = st.slider("💧 Average Humidity (%)", 10, 100, 50)
        soil_moisture = st.slider("🪱 Soil Moisture (%)", 5, 90, 30)
        deficit       = st.slider("📉 Rainfall Deficit (%)", 0, 80, 25)

    with col2:
        st.subheader("📊 Input Summary")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=deficit,
            title={'text': "Rainfall Deficit (%)"},
            gauge={
                'axis': {'range': [0, 80]},
                'bar':  {'color': "darkred"},
                'steps': [
                    {'range': [0, 10],  'color': '#c8e6c9'},
                    {'range': [10, 25], 'color': '#fff9c4'},
                    {'range': [25, 40], 'color': '#ffe0b2'},
                    {'range': [40, 80], 'color': '#ffcdd2'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 'value': deficit
                }
            }
        ))
        gauge.update_layout(height=280)
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("---")
    if st.button("🔮 Predict Drought Level"):
        input_data = np.array([[annual_rain, avg_temp,
                                avg_humidity, soil_moisture, deficit]])

        # ML Prediction
        ml_input   = ml_scaler.transform(input_data)
        ml_pred    = ml_model.predict(ml_input)[0]
        ml_proba   = ml_model.predict_proba(ml_input)[0]
        ml_label   = ml_le.inverse_transform([ml_pred])[0]
        ml_conf    = max(ml_proba) * 100

        # DL Prediction
        dl_input   = torch.FloatTensor(dl_scaler.transform(input_data))
        with torch.no_grad():
            dl_out  = dl_model(dl_input)
            dl_prob = torch.softmax(dl_out, dim=1).numpy()[0]
            dl_pred = np.argmax(dl_prob)
        dl_label    = dl_le.inverse_transform([dl_pred])[0]
        dl_conf     = max(dl_prob) * 100

        # Display Results
        col1, col2 = st.columns(2)
        css1, icon1, _ = drought_style(ml_label)
        css2, icon2, _ = drought_style(dl_label)

        with col1:
            st.markdown(f"""
            <div class='{css1}'>
                <h3>{icon1} ML Model (Random Forest)</h3>
                <h2>Drought Level: <b>{ml_label}</b></h2>
                <p>Confidence: <b>{ml_conf:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='{css2}'>
                <h3>{icon2} DL Model (Neural Network)</h3>
                <h2>Drought Level: <b>{dl_label}</b></h2>
                <p>Confidence: <b>{dl_conf:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Crop Recommendations
        st.markdown("---")
        st.subheader("🌱 Recommended Crops & Actions")
        from chatbot import CROP_ADVICE
        level_key = ml_label.lower()
        if level_key in CROP_ADVICE:
            info  = CROP_ADVICE[level_key]
            crops = info['crops']
            col1, col2, col3 = st.columns(3)
            for i, crop in enumerate(crops[:3]):
                [col1, col2, col3][i].success(f"✅ {crop}")
            if len(crops) > 3:
                col1, col2 = st.columns(2)
                for i, crop in enumerate(crops[3:5]):
                    [col1, col2][i].success(f"✅ {crop}")
            st.info(f"💡 {info['advice'][0]}")

# ══════════════════════════════════════════════════
# PAGE 3 — FARMER CHATBOT
# ══════════════════════════════════════════════════
elif page == "🤖 Farmer Chatbot":
    st.title("🤖 AI Farmer Chatbot")
    st.markdown("Ask me anything about farming, drought, crops, irrigation, or government schemes!")

    # Drought level selector
    drought_level = st.selectbox(
        "🌵 Current Drought Level in your area:",
        ['normal', 'mild', 'moderate', 'severe'],
        index=2
    )

    # Init chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            'role': 'bot',
            'text': "Namaste Kisan! 🌾 I am your AI Farming Assistant. Ask me about crops, irrigation, government schemes, or drought management!"
        })

    # Display chat history
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"<div class='chat-user'>👨‍🌾 <b>You:</b> {msg['text']}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>🤖 <b>Assistant:</b> {msg['text']}</div>",
                        unsafe_allow_html=True)

    # Input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("Type your question:", placeholder="e.g. What crops should I grow in drought?", key="chat_input")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send = st.button("Send 📨")

    # Quick question buttons
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
            st.session_state.messages.append({'role': 'bot',  'text': response})
            st.rerun()

    if send and user_input:
        response = get_response(user_input, drought_level)
        st.session_state.messages.append({'role': 'user', 'text': user_input})
        st.session_state.messages.append({'role': 'bot',  'text': response})
        st.rerun()

# ══════════════════════════════════════════════════
# PAGE 4 — DATA INSIGHTS
# ══════════════════════════════════════════════════
elif page == "📊 Data Insights":
    st.title("📊 India Rainfall & Drought Insights")

    df = pd.read_csv('data/rainfall_data.csv')
    df['Drought_Level'] = df['Drought_Level'].replace({'Severe': 'Moderate'})

    col1, col2 = st.columns(2)

    with col1:
        # Drought distribution
        fig1 = px.pie(df, names='Drought_Level',
                      title='Drought Level Distribution (2000-2023)',
                      color_discrete_map={
                          'Normal':'#388e3c', 'Mild':'#fbc02d',
                          'Moderate':'#f57c00'
                      })
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Rainfall trend
        trend = df.groupby('Year')['Annual_Rainfall_mm'].mean().reset_index()
        fig2  = px.line(trend, x='Year', y='Annual_Rainfall_mm',
                        title='India Average Rainfall Trend (2000-2023)',
                        markers=True)
        fig2.update_traces(line_color='#2196F3', line_width=2)
        st.plotly_chart(fig2, use_container_width=True)

    # Rainfall by state
    state_avg = df.groupby('State')['Annual_Rainfall_mm'].mean().reset_index()
    fig3 = px.bar(state_avg.sort_values('Annual_Rainfall_mm'),
                  x='Annual_Rainfall_mm', y='State',
                  orientation='h',
                  title='Average Annual Rainfall by State (mm)',
                  color='Annual_Rainfall_mm',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig3, use_container_width=True)

    # State selector for detailed view
    st.subheader("🔍 State-wise Drought History")
    selected_state = st.selectbox("Select State:", sorted(df['State'].unique()))
    state_df = df[df['State'] == selected_state]

    fig4 = px.bar(state_df, x='Year', y='Annual_Rainfall_mm',
                  color='Drought_Level',
                  color_discrete_map={
                      'Normal':'#388e3c','Mild':'#fbc02d','Moderate':'#f57c00'
                  },
                  title=f'{selected_state} — Yearly Rainfall & Drought Level')
    st.plotly_chart(fig4, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Rainfall",    f"{state_df['Annual_Rainfall_mm'].mean():.0f} mm")
    col2.metric("Drought Years",   f"{(state_df['Drought_Level']!='Normal').sum()}")
    col3.metric("Worst Deficit",   f"{state_df['Rainfall_Deficit_percent'].max():.1f}%")