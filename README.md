# 🌾 AI Drought Predictor & Farmer Assistant

End-to-end AI system for Indian Agriculture built 
using Machine Learning, Deep Learning, NLP and 
Computer Vision.

## 🔗 Links
- **Live App:** https://aditi-kamble-drought-predictor.streamlit.app
- **GitHub:** https://github.com/Aditi-Kamble/drought-predictor

## 🛠️ Tech Stack
- **ML:** Scikit-learn (Random Forest, 99%+ CV accuracy)
- **DL:** PyTorch (Neural Network, LSTM, CNN)
- **NLP:** Groq AI (LLaMA3) + Gemini fallback
- **Languages:** English, Hindi, Marathi
- **Web:** Streamlit + Plotly
- **APIs:** OpenWeather, Groq AI, Google Gemini
- **PDF:** fpdf2
- **Data:** Real IMD rainfall data (1901-2015)

## 📊 Features
1. 🔮 Drought Predictor - ML + DL with live weather
2. 🌧️ Rainfall Forecast - LSTM 6-month prediction
3. 🌿 Disease Detection - CNN leaf image analysis
4. 🤖 AI Farmer Chatbot - Groq/Gemini powered
5. 📊 Data Insights - 24 years rainfall trends
6. 🧮 Farm Calculator - Water, fertilizer, costs
7. 💰 Crop Price Predictor - Profit estimation
8. 📄 PDF Reports - Downloadable analysis
9. 👩‍💻 About - Project overview

## 📈 Model Performance
- Random Forest: 99%+ (5-fold CV validated)
- Neural Network: 88%+
- Data: Real IMD 1901-2015 dataset
- States: 28 Indian States
- Years: 2000-2023

## 🌐 Languages Supported
- English
- Hindi (हिंदी)
- Marathi (मराठी)

## ▶️ How to Run
```bash
pip install -r requirements.txt
python create_dataset.py
python src/ml_model.py
python src/dl_model.py
streamlit run app.py
```

## 📁 Project Structure
drought_predictor/

├── data/               # Real IMD rainfall dataset

├── models/             # Trained ML & DL models

├── src/

│   ├── ml_model.py     # Random Forest

│   ├── dl_model.py     # PyTorch Neural Net

│   ├── lstm_forecast.py # LSTM forecasting

│   ├── disease_detection.py # CNN disease

│   ├── chatbot.py      # Groq/Gemini AI

│   ├── weather.py      # OpenWeather API

│   ├── hindi_support.py # Hindi NLP

│   ├── marathi_support.py # Marathi NLP

│   ├── price_predictor.py # Crop prices

│   └── report_generator.py # PDF reports

├── app.py              # Main Streamlit app

└── requirements.txt

## 👩‍💻 Developer
**Aditi Kamble** | AI/ML Developer
- GitHub: github.com/Aditi-Kamble
- LinkedIn: linkedin.com/in/your-profile
- Live: aditi-kamble-drought-predictor.streamlit.app

## 📝 License
MIT License - Free to use and modify