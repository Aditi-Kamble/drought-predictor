import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

print("="*50)
print("   DROUGHT PREDICTION - ML MODEL (Random Forest)")
print("="*50)

# ── 1. Load Data ──────────────────────────────
df = pd.read_csv('data/rainfall_data.csv')
print(f"\n✅ Data loaded: {df.shape}")

# ── 2. Check Severe Class ─────────────────────
severe_count = (df['Drought_Level'] == 'Severe').sum()
print(f"\n📊 Drought Distribution:")
print(df['Drought_Level'].value_counts())
print(f"\n📊 Severe drought samples: {severe_count}")

if severe_count < 5:
    print("⚠️  Too few Severe samples, merging with Moderate")
    df['Drought_Level'] = df['Drought_Level'].replace(
        {'Severe': 'Moderate'})
else:
    print("✅ Keeping all 4 drought classes!")

print(f"\n📊 Final Distribution:")
print(df['Drought_Level'].value_counts())

# ── 3. Features & Target ──────────────────────
features = [
    'Annual_Rainfall_mm',
    'Avg_Temperature_C',
    'Avg_Humidity_percent',
    'Soil_Moisture_percent',
    'Rainfall_Deficit_percent'
]

X = df[features]
y = df['Drought_Level']

# ── 4. Encode Labels ──────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n📋 Classes: {list(le.classes_)}")

# ── 5. Split Data ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42
)
print(f"\n📊 Train: {X_train.shape[0]} | "
      f"Test: {X_test.shape[0]}")

# ── 6. Scale Features ─────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 7. Train Model ────────────────────────────
print("\n🌲 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)
print("✅ Training complete!")

# ── 8. Evaluate ───────────────────────────────
y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"   TEST ACCURACY: {acc*100:.2f}%")
print(f"{'='*50}")
print("\n📋 Classification Report:")
print(classification_report(
    y_test, y_pred, target_names=le.classes_))

# ── 9. Cross Validation ───────────────────────
print("\n📊 5-Fold Cross Validation:")
cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model, X_train_scaled, y_train,
    cv=cv, scoring='accuracy')

print(f"CV Scores:   "
      f"{[f'{s:.3f}' for s in cv_scores]}")
print(f"Mean CV Acc: {cv_scores.mean()*100:.2f}%")
print(f"Std CV Acc:  {cv_scores.std()*100:.2f}%")

# Save CV results
cv_results = {
    'cv_scores': cv_scores.tolist(),
    'mean_acc':  round(cv_scores.mean()*100, 2),
    'std_acc':   round(cv_scores.std()*100, 2),
    'test_acc':  round(acc*100, 2),
}
os.makedirs('models', exist_ok=True)
with open('models/cv_results.json', 'w') as f:
    json.dump(cv_results, f)
print("✅ CV results saved!")

# ── 10. Confusion Matrix ──────────────────────
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            cmap='YlOrRd',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix - Drought Prediction',
          fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
os.makedirs('data', exist_ok=True)
plt.savefig('data/confusion_matrix.png', dpi=150)
plt.show()

# ── 11. Feature Importance ────────────────────
plt.figure(figsize=(8, 5))
importance = pd.Series(
    model.feature_importances_, index=features)
importance.sort_values().plot(
    kind='barh', color='steelblue')
plt.title('Feature Importance - What Drives Drought?',
          fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('data/feature_importance.png', dpi=150)
plt.show()

# ── 12. Save Model ────────────────────────────
with open('models/ml_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\n✅ Model saved to models/ml_model.pkl")

# ── 13. Sample Prediction ─────────────────────
print("\n🔮 Sample Prediction Test:")
sample = pd.DataFrame([{
    'Annual_Rainfall_mm':       300,
    'Avg_Temperature_C':        36,
    'Avg_Humidity_percent':     25,
    'Soil_Moisture_percent':    15,
    'Rainfall_Deficit_percent': 45
}])
sample_scaled = scaler.transform(sample)
pred  = model.predict(sample_scaled)
prob  = model.predict_proba(sample_scaled)

print(f"   Input:      Low rainfall, High temp, Dry soil")
print(f"   Prediction: "
      f"{le.inverse_transform(pred)[0]}")
print(f"   Confidence: {max(prob[0])*100:.1f}%")
print(f"\n🎯 Phase 3 Complete! ML Model ready.")
print(f"\n📊 Summary:")
print(f"   Test Accuracy:  {acc*100:.2f}%")
print(f"   CV Mean Acc:    {cv_scores.mean()*100:.2f}%")
print(f"   CV Std:         {cv_scores.std()*100:.2f}%")
print(f"   Classes:        {list(le.classes_)}")