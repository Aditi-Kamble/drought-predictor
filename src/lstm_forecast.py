import numpy as np
import pandas as pd
import os

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

# ── LSTM Model ────────────────────────────────
if TORCH_AVAILABLE:
    class RainfallLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=64,
                     num_layers=2, output_size=1):
            super(RainfallLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers  = num_layers
            self.lstm = nn.LSTM(
                input_size, hidden_size,
                num_layers, batch_first=True,
                dropout=0.2
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(
                self.num_layers, x.size(0),
                self.hidden_size)
            c0 = torch.zeros(
                self.num_layers, x.size(0),
                self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm_model(state_name=None):
    # Load data
    df = pd.read_csv('data/rainfall_data.csv')

    if state_name and state_name in df['State'].values:
        state_df = df[df['State'] == state_name].sort_values('Year')
        rainfall  = state_df['Annual_Rainfall_mm'].values
    else:
        rainfall = df.groupby(
            'Year')['Annual_Rainfall_mm'].mean().values

    # Create monthly simulation from annual data
    monthly_rainfall = []
    monthly_factors  = [
        0.02, 0.02, 0.03, 0.04, 0.06,
        0.15, 0.22, 0.20, 0.12, 0.07,
        0.03, 0.02
    ]
    for annual in rainfall:
        for factor in monthly_factors:
            noise = np.random.normal(0, annual * 0.02)
            monthly_rainfall.append(
                max(0, annual * factor + noise))

    monthly_rainfall = np.array(monthly_rainfall)

    # If torch not available use simple fallback
    if not TORCH_AVAILABLE:
        return None, None, monthly_rainfall

    # Scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(
        monthly_rainfall.reshape(-1, 1))

    # Create sequences
    seq_length = 12
    X, y = create_sequences(scaled, seq_length)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Train model
    model     = RainfallLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss    = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return model, scaler, monthly_rainfall

def forecast_rainfall(state_name=None, months_ahead=6):
    model, scaler, historical = train_lstm_model(state_name)

    # Fallback if torch not available
    if not TORCH_AVAILABLE or model is None:
        last_12  = historical[-12:]
        avg      = np.mean(last_12)
        std      = np.std(last_12)
        predictions = np.array([
            max(0, avg + np.random.normal(0, std * 0.3))
            for _ in range(months_ahead)
        ])
        return predictions, historical[-12:]

    # LSTM forecast
    scaled_historical = scaler.transform(
        historical.reshape(-1, 1))
    last_sequence = scaled_historical[-12:].reshape(1, 12, 1)

    model.eval()
    predictions = []
    current_seq = torch.FloatTensor(last_sequence)

    with torch.no_grad():
        for _ in range(months_ahead):
            pred = model(current_seq)
            predictions.append(pred.item())
            new_seq = torch.cat([
                current_seq[:, 1:, :],
                pred.unsqueeze(0)
            ], dim=1)
            current_seq = new_seq

    # Inverse transform
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(
        predictions).flatten()
    predictions = np.maximum(predictions, 0)

    return predictions, historical[-12:]

def get_forecast_summary(predictions):
    total = sum(predictions)
    avg   = total / len(predictions)

    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']

    import datetime
    current_month = datetime.datetime.now().month
    month_names   = []
    for i in range(len(predictions)):
        idx = (current_month + i) % 12
        month_names.append(months[idx])

    if avg < 30:
        outlook = "Severe Drought Risk"
        advice  = "Prepare for severe water shortage. Plant only drought-resistant crops."
    elif avg < 60:
        outlook = "Moderate Drought Risk"
        advice  = "Water conservation critical. Choose drought-tolerant crops."
    elif avg < 100:
        outlook = "Mild Drought Possible"
        advice  = "Monitor rainfall closely. Have contingency crop plan ready."
    else:
        outlook = "Good Rainfall Expected"
        advice  = "Good season ahead! Plan water-intensive crops."

    return {
        'predictions': predictions,
        'month_names': month_names,
        'total':       round(total, 1),
        'avg':         round(avg, 1),
        'outlook':     outlook,
        'advice':      advice
    }