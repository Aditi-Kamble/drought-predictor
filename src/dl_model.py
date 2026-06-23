import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle
import os

print("="*50)
print("   DROUGHT PREDICTION - DL MODEL (PyTorch)")
print("="*50)

# 1. Load Data
df = pd.read_csv('data/rainfall_data.csv')
df['Drought_Level'] = df['Drought_Level'].replace({'Severe': 'Moderate'})
print(f"\n✅ Data loaded: {df.shape}")

# 2. Features & Target
features = [
    'Annual_Rainfall_mm', 'Avg_Temperature_C',
    'Avg_Humidity_percent', 'Soil_Moisture_percent',
    'Rainfall_Deficit_percent'
]
X = df[features].values
y = df['Drought_Level'].values

# 3. Encode & Scale
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"📋 Classes: {list(le.classes_)}")
print(f"📊 Number of classes: {len(le.classes_)}")

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# 5. Convert to Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.LongTensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# 6. Build Neural Network
class DroughtNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DroughtNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = DroughtNet(input_size=len(features), num_classes=len(le.classes_))
print(f"\n🧠 Neural Network Architecture:")
print(model)

# 7. Train
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
train_losses = []
train_accs   = []

print(f"\n⚡ Training for {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_dl:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = epoch_loss / len(train_dl)
    avg_acc  = correct / total * 100
    train_losses.append(avg_loss)
    train_accs.append(avg_acc)

    if (epoch+1) % 10 == 0:
        print(f"   Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.1f}%")

# 8. Evaluate
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)
    y_pred = predicted.numpy()

acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"   TEST ACCURACY: {acc*100:.2f}%")
print(f"{'='*50}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 9. Plot Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, color='red', linewidth=2)
ax1.set_title('Training Loss over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

ax2.plot(train_accs, color='green', linewidth=2)
ax2.set_title('Training Accuracy over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True)

plt.suptitle('DroughtNet - Deep Learning Training', fontsize=14)
plt.tight_layout()
plt.savefig('data/dl_training_curves.png', dpi=150)
plt.show()

# 10. Save Model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/dl_model.pth')

# Save dl scaler & encoder separately
with open('models/dl_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/dl_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save model config for loading later
import json
config = {
    'input_size': len(features),
    'num_classes': len(le.classes_),
    'features': features
}
with open('models/dl_config.json', 'w') as f:
    json.dump(config, f)

print("\n✅ DL Model saved to models/dl_model.pth")

# 11. Sample Prediction
print("\n🔮 Sample Prediction Test:")
model.eval()
sample = torch.FloatTensor(scaler.transform([[300, 36, 25, 15, 45]]))
with torch.no_grad():
    output = model(sample)
    probs  = torch.softmax(output, dim=1)
    pred   = torch.argmax(probs, dim=1)

print(f"   Input: Low rainfall(300mm), High temp(36C), Dry soil")
print(f"   Prediction : {le.inverse_transform(pred.numpy())[0]}")
print(f"   Confidence : {probs.max().item()*100:.1f}%")
print(f"   All Probs  : { {c:round(p,3) for c,p in zip(le.classes_, probs[0].numpy())} }")
print("\n🎯 Phase 4 Complete! DL Model ready.")