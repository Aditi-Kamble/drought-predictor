import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Indian states
states = [
    'Andhra Pradesh', 'Bihar', 'Gujarat', 'Haryana', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Punjab', 'Rajasthan',
    'Tamil Nadu', 'Uttar Pradesh', 'West Bengal', 'Odisha', 'Jharkhand'
]

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
years = list(range(2000, 2024))

rows = []

for state in states:
    # Base rainfall profile per state (realistic mm values)
    base = {
        'Rajasthan': [5,5,5,5,10,30,100,80,30,5,5,5],
        'Maharashtra': [2,2,3,5,20,150,250,220,150,50,20,5],
        'Kerala': [20,25,40,100,200,350,400,380,250,150,80,30],
        'Punjab': [30,25,20,10,15,40,120,100,50,10,5,20],
        'Gujarat': [2,2,2,2,5,60,180,160,60,10,2,2],
        'Karnataka': [5,8,12,30,80,100,120,110,150,120,40,10],
        'Tamil Nadu': [30,20,10,15,30,40,60,80,120,180,140,80],
        'Madhya Pradesh': [15,10,10,5,10,80,200,220,150,30,10,10],
        'Uttar Pradesh': [20,15,10,5,10,50,180,200,120,20,5,15],
        'Bihar': [20,15,10,5,15,80,200,220,180,40,10,15],
        'West Bengal': [25,25,30,50,120,200,320,300,250,120,40,15],
        'Andhra Pradesh': [10,10,10,20,40,60,120,130,150,100,60,15],
        'Haryana': [25,20,15,8,12,35,100,90,50,8,5,18],
        'Odisha': [15,20,25,30,60,150,280,300,220,100,40,15],
        'Jharkhand': [20,20,20,25,50,150,280,290,200,60,20,15],
    }

    state_base = base.get(state, [20]*12)

    for year in years:
        # Climate trend: rainfall decreasing slightly over years
        climate_factor = 1 - (year - 2000) * 0.003

        # Random yearly variation
        year_variation = np.random.uniform(0.7, 1.3)

        monthly_rain = []
        for m_idx, month in enumerate(months):
            rain = state_base[m_idx] * climate_factor * year_variation
            rain += np.random.normal(0, state_base[m_idx] * 0.1)
            rain = max(0, rain)
            monthly_rain.append(round(rain, 2))

        annual_rain = sum(monthly_rain)

        # Temperature & humidity (correlated with drought)
        avg_temp = np.random.uniform(24, 38)
        avg_humidity = np.random.uniform(30, 85)
        soil_moisture = np.random.uniform(10, 80)

        # Drought label based on annual rainfall vs state normal
        normal_rain = sum(state_base)
        deficit = (normal_rain - annual_rain) / normal_rain * 100

        if deficit > 40:
            drought_level = 'Severe'
        elif deficit > 25:
            drought_level = 'Moderate'
        elif deficit > 10:
            drought_level = 'Mild'
        else:
            drought_level = 'Normal'

        row = {
            'State': state,
            'Year': year,
            'Annual_Rainfall_mm': round(annual_rain, 2),
            'Avg_Temperature_C': round(avg_temp, 2),
            'Avg_Humidity_percent': round(avg_humidity, 2),
            'Soil_Moisture_percent': round(soil_moisture, 2),
            'Rainfall_Deficit_percent': round(deficit, 2),
            'Drought_Level': drought_level
        }

        # Add monthly columns
        for m, val in zip(months, monthly_rain):
            row[f'Rain_{m}_mm'] = val

        rows.append(row)

df = pd.DataFrame(rows)

os.makedirs('data', exist_ok=True)
df.to_csv('data/rainfall_data.csv', index=False)

print(f"✅ Dataset created successfully!")
print(f"📊 Shape: {df.shape}")
print(f"\n🌧️ Drought Level Distribution:")
print(df['Drought_Level'].value_counts())
print(f"\n📋 Sample Data:")
print(df[['State','Year','Annual_Rainfall_mm','Drought_Level']].head(10))