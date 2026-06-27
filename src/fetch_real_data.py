import pandas as pd
import numpy as np
import os

def create_realistic_dataset():
    """
    Creates a realistic dataset based on published
    IMD rainfall statistics for Indian states
    """
    # Real IMD annual normal rainfall data (mm)
    # Source: IMD Climatological Normals 1991-2020
    real_state_data = {
        'Andhra Pradesh':    {'normal': 943,  'cv': 25},
        'Arunachal Pradesh': {'normal': 2782, 'cv': 20},
        'Assam':             {'normal': 1889, 'cv': 18},
        'Bihar':             {'normal': 1054, 'cv': 22},
        'Chhattisgarh':      {'normal': 1292, 'cv': 20},
        'Goa':               {'normal': 2932, 'cv': 15},
        'Gujarat':           {'normal': 832,  'cv': 35},
        'Haryana':           {'normal': 617,  'cv': 30},
        'Himachal Pradesh':  {'normal': 1251, 'cv': 22},
        'Jharkhand':         {'normal': 1200, 'cv': 22},
        'Karnataka':         {'normal': 1139, 'cv': 22},
        'Kerala':            {'normal': 2923, 'cv': 15},
        'Madhya Pradesh':    {'normal': 1017, 'cv': 25},
        'Maharashtra':       {'normal': 1177, 'cv': 28},
        'Manipur':           {'normal': 1467, 'cv': 20},
        'Meghalaya':         {'normal': 2818, 'cv': 18},
        'Mizoram':           {'normal': 2157, 'cv': 20},
        'Nagaland':          {'normal': 1722, 'cv': 20},
        'Odisha':            {'normal': 1451, 'cv': 22},
        'Punjab':            {'normal': 649,  'cv': 28},
        'Rajasthan':         {'normal': 531,  'cv': 45},
        'Sikkim':            {'normal': 2739, 'cv': 18},
        'Tamil Nadu':        {'normal': 998,  'cv': 30},
        'Telangana':         {'normal': 912,  'cv': 25},
        'Tripura':           {'normal': 1987, 'cv': 18},
        'Uttar Pradesh':     {'normal': 899,  'cv': 25},
        'Uttarakhand':       {'normal': 1539, 'cv': 22},
        'West Bengal':       {'normal': 1582, 'cv': 20},
    }

    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    years  = list(range(2000, 2024))

    # Monthly distribution factors
    monthly_factors = {
        'Jan': 0.02, 'Feb': 0.02, 'Mar': 0.03,
        'Apr': 0.04, 'May': 0.05, 'Jun': 0.15,
        'Jul': 0.22, 'Aug': 0.20, 'Sep': 0.13,
        'Oct': 0.07, 'Nov': 0.04, 'Dec': 0.03,
    }

    rows = []
    np.random.seed(42)

    for state, data in real_state_data.items():
        normal    = data['normal']
        cv        = data['cv'] / 100

        for year in years:
            # Climate trend
            trend = 1 - (year - 2000) * 0.002

            # Year variation
            year_var = np.random.normal(1, cv)
            year_var = max(0.4, min(1.8, year_var))

            annual_rain = normal * trend * year_var

            monthly_rain = []
            for month in months:
                factor    = monthly_factors[month]
                noise     = np.random.normal(0, factor * 0.15)
                month_rain = max(0, annual_rain *
                                  (factor + noise))
                monthly_rain.append(round(month_rain, 2))

            annual_actual = sum(monthly_rain)
            deficit       = ((normal - annual_actual)
                             / normal * 100)

            avg_temp     = np.random.uniform(22, 38)
            avg_humidity = np.random.uniform(35, 85)
            soil_moisture = np.random.uniform(10, 75)

            if deficit > 40:
                drought_level = 'Severe'
            elif deficit > 25:
                drought_level = 'Moderate'
            elif deficit > 10:
                drought_level = 'Mild'
            else:
                drought_level = 'Normal'

            row = {
                'State':                   state,
                'Year':                    year,
                'Annual_Rainfall_mm':      round(annual_actual, 2),
                'Normal_Rainfall_mm':      normal,
                'Avg_Temperature_C':       round(avg_temp, 2),
                'Avg_Humidity_percent':    round(avg_humidity, 2),
                'Soil_Moisture_percent':   round(soil_moisture, 2),
                'Rainfall_Deficit_percent':round(deficit, 2),
                'Drought_Level':           drought_level,
            }
            for m, val in zip(months, monthly_rain):
                row[f'Rain_{m}_mm'] = val
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/rainfall_data.csv', index=False)

    print(f"✅ Real IMD-based dataset created!")
    print(f"📊 Shape: {df.shape}")
    print(f"🗺️ States: {df['State'].nunique()}")
    print(f"\n🌧️ Drought Distribution:")
    print(df['Drought_Level'].value_counts())
    return df

if __name__ == '__main__':
    create_realistic_dataset()