import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("="*50)
print("   DROUGHT PREDICTION - ML MODEL (Random Forest)")
print("="*50)

# 1. Load Data
df = pd.read_csv('data/rainfall_data.csv')
print(f"\n✅ Data loaded: {df.shape}")

# 2. Fix: Merge 'Severe' into 'Moderate' since very few samples
df['Drought_Level'] = df['Drought_Level'].replace({'Severe': 'Moderate'})
print(f"\n📋 Drought Distribution after fix:")
print(df['Drought_Level'].value_counts())

# 3. Features & Target
features = [
    'Annual_Rainfall_mm', 'Avg_Temperature_C',
    'Avg_Humidity_percent', 'Soil_Moisture_percent',
    'Rainfall_Deficit_percent'
]

X = df[features]
y = df['Drought_Level']

# 4. Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n📋 Classes: {list(le.classes_)}")

# 5. Split Data (no stratify issue now)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
print(f"\n📊 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# 6. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 7. Train Model
print("\n🌲 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)
print("✅ Training complete!")

# 8. Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"   MODEL ACCURACY: {acc*100:.2f}%")
print(f"{'='*50}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 9. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Drought Prediction', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('data/confusion_matrix.png', dpi=150)
plt.show()

# 10. Feature Importance
plt.figure(figsize=(8, 5))
importance = pd.Series(model.feature_importances_, index=features)
importance.sort_values().plot(kind='barh', color='steelblue')
plt.title('Feature Importance - What Drives Drought?', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('data/feature_importance.png', dpi=150)
plt.show()

# 11. Save Model
os.makedirs('models', exist_ok=True)
with open('models/ml_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\n✅ Model saved to models/ml_model.pkl")

# 12. Sample Prediction Test
print("\n🔮 Sample Prediction Test:")
sample = pd.DataFrame([{
    'Annual_Rainfall_mm': 300,
    'Avg_Temperature_C': 36,
    'Avg_Humidity_percent': 25,
    'Soil_Moisture_percent': 15,
    'Rainfall_Deficit_percent': 45
}])
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)
prob = model.predict_proba(sample_scaled)
print(f"   Input: Low rainfall (300mm), High temp (36C), Dry soil")
print(f"   Prediction: {le.inverse_transform(pred)[0]}")
print(f"   Confidence: {max(prob[0])*100:.1f}%")
print("\n🎯 Phase 3 Complete! ML Model ready.")