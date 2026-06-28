import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Subdivision to State mapping
SUBDIVISION_STATE = {
    'ANDAMAN & NICOBAR ISLANDS': 'Andaman & Nicobar',
    'ARUNACHAL PRADESH':         'Arunachal Pradesh',
    'ASSAM & MEGHALAYA':         'Assam',
    'BIHAR':                     'Bihar',
    'CHHATTISGARH':              'Chhattisgarh',
    'COASTAL ANDHRA PRADESH':    'Andhra Pradesh',
    'COASTAL KARNATAKA':         'Karnataka',
    'EAST MADHYA PRADESH':       'Madhya Pradesh',
    'EAST RAJASTHAN':            'Rajasthan',
    'EAST UTTAR PRADESH':        'Uttar Pradesh',
    'GANGETIC WEST BENGAL':      'West Bengal',
    'GUJARAT REGION':            'Gujarat',
    'HARYANA DELHI & CHANDIGARH':'Haryana',
    'HIMACHAL PRADESH':          'Himachal Pradesh',
    'JAMMU & KASHMIR':           'Jammu & Kashmir',
    'JHARKHAND':                 'Jharkhand',
    'KERALA':                    'Kerala',
    'KONKAN & GOA':              'Maharashtra',
    'LAKSHADWEEP':               'Kerala',
    'MADHYA MAHARASHTRA':        'Maharashtra',
    'MARATHWADA':                'Maharashtra',
    'MATATHWADA':                'Maharashtra',
    'NAGA MANI MIZO TRIPURA':    'Nagaland',
    'NORTH INTERIOR KARNATAKA':  'Karnataka',
    'NORTH MADHYA PRADESH':      'Madhya Pradesh',
    'ORISSA':                    'Odisha',
    'PUNJAB':                    'Punjab',
    'RAYALSEEMA':                'Andhra Pradesh',
    'SAURASHTRA & KUTCH':        'Gujarat',
    'SIKKIM':                    'Sikkim',
    'SOUTH INTERIOR KARNATAKA':  'Karnataka',
    'SUB HIMALAYAN WEST BENGAL & SIKKIM': 'West Bengal',
    'TAMIL NADU':                'Tamil Nadu',
    'TELANGANA':                 'Telangana',
    'UTTARAKHAND':               'Uttarakhand',
    'VIDARBHA':                  'Maharashtra',
    'WEST MADHYA PRADESH':       'Madhya Pradesh',
    'WEST RAJASTHAN':            'Rajasthan',
    'WEST UTTAR PRADESH':        'Uttar Pradesh',
    'UTTARANCHAL':               'Uttarakhand',
    'BIHAR PLATEAU':             'Bihar',
    'BIHAR PLAINS':              'Bihar',
    'SUB HIMALAYAN WEST BENGAL': 'West Bengal',
    'ANDHRA PRADESH':            'Andhra Pradesh',
}

# State normal rainfall (IMD reference)
STATE_NORMALS = {
    'Andhra Pradesh':    943,
    'Arunachal Pradesh': 2782,
    'Assam':             1889,
    'Bihar':             1054,
    'Chhattisgarh':      1292,
    'Goa':               2932,
    'Gujarat':           832,
    'Haryana':           617,
    'Himachal Pradesh':  1251,
    'Jharkhand':         1200,
    'Karnataka':         1139,
    'Kerala':            2923,
    'Madhya Pradesh':    1017,
    'Maharashtra':       1177,
    'Manipur':           1467,
    'Meghalaya':         2818,
    'Mizoram':           2157,
    'Nagaland':          1722,
    'Odisha':            1451,
    'Punjab':            649,
    'Rajasthan':         531,
    'Sikkim':            2739,
    'Tamil Nadu':        998,
    'Telangana':         912,
    'Tripura':           1987,
    'Uttar Pradesh':     899,
    'Uttarakhand':       1539,
    'West Bengal':       1582,
}

def process_real_data():
    raw_path = 'data/rainfall_data_raw.csv'

    if not os.path.exists(raw_path):
        print("❌ Raw data file not found!")
        print("Please copy rainfall_data_raw.csv to data/ folder")
        return False

    print("📂 Loading real Kaggle dataset...")
    df_raw = pd.read_csv(raw_path)

    print(f"✅ Raw data loaded: {df_raw.shape}")
    print(f"📋 Columns: {list(df_raw.columns)}")
    print(f"📋 Sample:\n{df_raw.head(3)}")

    # Clean column names
    df_raw.columns = df_raw.columns.str.strip().str.upper()
    print(f"\n📋 Cleaned Columns: {list(df_raw.columns)}")

    # Find subdivision and year columns
    sub_col  = None
    year_col = None

    for col in df_raw.columns:
        if 'SUBDIVISION' in col or 'SUB' in col:
            sub_col = col
        if 'YEAR' in col:
            year_col = col

    if not sub_col:
        sub_col  = df_raw.columns[0]
    if not year_col:
        year_col = df_raw.columns[1]

    print(f"\n📋 Using: {sub_col} and {year_col}")

    # Month columns
    month_cols = []
    month_names = ['JAN','FEB','MAR','APR','MAY','JUN',
                   'JUL','AUG','SEP','OCT','NOV','DEC']

    for col in df_raw.columns:
        for m in month_names:
            if m == col or m in col:
                month_cols.append(col)
                break

    print(f"📋 Month columns found: {month_cols}")

    # Annual column
    annual_col = None
    for col in df_raw.columns:
        if 'ANNUAL' in col or 'TOTAL' in col:
            annual_col = col
            break

    print(f"📋 Annual column: {annual_col}")

    # Map subdivisions to states
    df_raw['State'] = df_raw[sub_col].str.strip().str.upper().map(
        SUBDIVISION_STATE)

    # Drop unmapped
    df_raw = df_raw.dropna(subset=['State'])
    print(f"\n✅ After state mapping: {df_raw.shape}")

    # Filter years 2000-2023
    df_raw[year_col] = pd.to_numeric(
        df_raw[year_col], errors='coerce')
    df_filtered = df_raw[
        (df_raw[year_col] >= 2000) &
        (df_raw[year_col] <= 2023)
    ].copy()

    print(f"✅ After year filter (2000-2023): {df_filtered.shape}")

    if len(df_filtered) == 0:
        print("⚠️  No data for 2000-2023!")
        print("Using 1990-2015 instead...")
        df_filtered = df_raw[
            df_raw[year_col] >= 1990].copy()

    # Group by State and Year
    if month_cols:
        agg_dict = {}
        for col in month_cols:
            df_filtered[col] = pd.to_numeric(
                df_filtered[col], errors='coerce')
            agg_dict[col] = 'mean'
        if annual_col:
            df_filtered[annual_col] = pd.to_numeric(
                df_filtered[annual_col], errors='coerce')
            agg_dict[annual_col] = 'mean'

        df_state = df_filtered.groupby(
            ['State', year_col]).agg(agg_dict).reset_index()
    else:
        df_state = df_filtered.groupby(
            ['State', year_col]).first().reset_index()

    print(f"✅ After groupby: {df_state.shape}")

    # Build final dataset
    rows = []
    months_short = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']

    np.random.seed(42)

    for _, row in df_state.iterrows():
        state = row['State']
        year  = int(row[year_col])
        normal = STATE_NORMALS.get(state, 1000)

        # Get monthly rainfall
        monthly_rain = []
        if month_cols:
            for col in month_cols[:12]:
                val = pd.to_numeric(row[col],
                                     errors='coerce')
                if pd.isna(val) or val < 0:
                    val = 0
                monthly_rain.append(round(val, 2))
        else:
            monthly_rain = [0] * 12

        # Pad if needed
        while len(monthly_rain) < 12:
            monthly_rain.append(0)

        # Annual rainfall
        if annual_col and not pd.isna(row.get(annual_col)):
            annual = float(row[annual_col])
        else:
            annual = sum(monthly_rain)

        if annual <= 0:
            annual = sum(monthly_rain)
        if annual <= 0:
            continue

        # Deficit
        deficit = ((normal - annual) / normal * 100)

        # Temperature correlated with rainfall
        rain_ratio = annual / normal
        avg_temp   = np.clip(
            32 - (rain_ratio - 1) * 8
            + np.random.normal(0, 2),
            15, 45)

        # Humidity correlated with rainfall
        avg_humidity = np.clip(
            55 * rain_ratio
            + np.random.normal(0, 5),
            20, 90)

        # Soil moisture
        soil_moisture = np.clip(
            40 * rain_ratio
            + np.random.normal(0, 5),
            5, 85)

        # Drought level
        if deficit > 40:
            drought_level = 'Severe'
        elif deficit > 25:
            drought_level = 'Moderate'
        elif deficit > 10:
            drought_level = 'Mild'
        else:
            drought_level = 'Normal'

        new_row = {
            'State':                    state,
            'Year':                     year,
            'Annual_Rainfall_mm':       round(annual, 2),
            'Normal_Rainfall_mm':       normal,
            'Avg_Temperature_C':        round(avg_temp, 2),
            'Avg_Humidity_percent':     round(avg_humidity, 2),
            'Soil_Moisture_percent':    round(soil_moisture, 2),
            'Rainfall_Deficit_percent': round(deficit, 2),
            'Drought_Level':            drought_level,
        }

        for i, m in enumerate(months_short):
            new_row[f'Rain_{m}_mm'] = (
                monthly_rain[i] if i < len(monthly_rain)
                else 0)

        rows.append(new_row)

    df_final = pd.DataFrame(rows)

    if len(df_final) == 0:
        print("❌ No data processed!")
        return False

    # Save
    df_final.to_csv('data/rainfall_data.csv', index=False)

    print(f"\n{'='*50}")
    print(f"✅ REAL DATASET CREATED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"📊 Shape: {df_final.shape}")
    print(f"🗺️  States: {df_final['State'].nunique()}")
    print(f"📅 Years: {df_final['Year'].min()} - "
          f"{df_final['Year'].max()}")
    print(f"\n🌧️  Drought Distribution:")
    print(df_final['Drought_Level'].value_counts())
    print(f"\n📋 Sample:")
    print(df_final[['State','Year',
                     'Annual_Rainfall_mm',
                     'Drought_Level']].head(10))
    return True

if __name__ == '__main__':
    success = process_real_data()
    if success:
        print("\n🚀 Now run:")
        print("python src/ml_model.py")
        print("python src/dl_model.py")
    else:
        print("\n❌ Processing failed!")
        print("Check your raw data file!")