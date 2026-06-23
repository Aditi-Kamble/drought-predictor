import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/rainfall_data.csv')

print("=== DATASET INFO ===")
print(df.shape)
print(df.dtypes)
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())
print("\n=== DROUGHT DISTRIBUTION ===")
print(df['Drought_Level'].value_counts())
print("\n=== STATS ===")
print(df[['Annual_Rainfall_mm','Avg_Temperature_C','Soil_Moisture_percent']].describe())

# Plot 1: Drought distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
df['Drought_Level'].value_counts().plot(kind='bar', color=['green','yellow','orange','red'])
plt.title('Drought Level Distribution')
plt.xticks(rotation=45)

# Plot 2: Rainfall by state
plt.subplot(1, 3, 2)
state_rain = df.groupby('State')['Annual_Rainfall_mm'].mean().sort_values()
state_rain.plot(kind='barh', color='steelblue')
plt.title('Avg Rainfall by State (mm)')

# Plot 3: Rainfall trend over years
plt.subplot(1, 3, 3)
year_rain = df.groupby('Year')['Annual_Rainfall_mm'].mean()
year_rain.plot(kind='line', color='blue', marker='o', markersize=3)
plt.title('India Avg Rainfall Trend (2000-2023)')
plt.xlabel('Year')
plt.ylabel('mm')

plt.tight_layout()
plt.savefig('data/exploration_plots.png', dpi=150)
plt.show()
print("\n✅ Plots saved to data/exploration_plots.png")