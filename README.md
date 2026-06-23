# 🌾 AI Drought Predictor & Farmer Assistant

An end-to-end AI/ML/DL/NLP project that predicts drought severity 
and helps Indian farmers with crop recommendations.

## 🚀 Features
- Drought level prediction using Random Forest (ML)
- Deep Learning model using PyTorch Neural Network (DL)
- NLP-powered Farmer Chatbot using spaCy
- Interactive web app built with Streamlit
- Covers 15 Indian states, 2000-2023 data

## 🛠️ Tech Stack
- Python 3.14
- scikit-learn (Random Forest)
- PyTorch (Neural Network)
- spaCy (NLP)
- Streamlit (Web App)
- Plotly (Charts)

## ▶️ How to Run
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python create_dataset.py
python src/ml_model.py
python src/dl_model.py
streamlit run app.py

## 📊 Results
- ML Model Accuracy: 95%+
- DL Model Accuracy: 88%+
- States Covered: 15
- Years of Data: 2000-2023

## 👨‍💻 Project Structure
drought_predictor/
├── data/               # Dataset & plots
├── models/             # Saved ML & DL models
├── src/
│   ├── ml_model.py     # Random Forest
│   ├── dl_model.py     # PyTorch Neural Net
│   └── chatbot.py      # NLP Chatbot
├── app.py              # Streamlit Web App
└── README.md