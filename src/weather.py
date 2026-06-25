import requests
import os

def get_weather(city_name, api_key):
    try:
        # Fetch current weather
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': f"{city_name},IN",
            'appid': api_key,
            'units': 'metric'
        }
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code != 200:
            return None, f"City not found: {city_name}"

        # Extract weather data
        weather = {
            'city': data['name'],
            'state': data.get('sys', {}).get('country', 'IN'),
            'temperature': round(data['main']['temp'], 1),
            'humidity': round(data['main']['humidity'], 1),
            'description': data['weather'][0]['description'].title(),
            'wind_speed': round(data['wind']['speed'] * 3.6, 1),
            'feels_like': round(data['main']['feels_like'], 1),
            'pressure': data['main']['pressure'],
            'visibility': data.get('visibility', 0) / 1000,
            'icon': data['weather'][0]['icon']
        }
        return weather, None

    except Exception as e:
        return None, str(e)

def get_rainfall_estimate(humidity, description):
    desc = description.lower()
    if 'heavy rain' in desc or 'thunderstorm' in desc:
        base = 150
    elif 'rain' in desc or 'drizzle' in desc:
        base = 80
    elif 'cloud' in desc:
        base = 40
    elif 'clear' in desc:
        base = 10
    else:
        base = 25

    # Adjust based on humidity
    if humidity > 80:
        base *= 1.3
    elif humidity < 40:
        base *= 0.6

    return round(base, 1)

def estimate_soil_moisture(humidity, description):
    desc = description.lower()
    if 'rain' in desc or 'thunderstorm' in desc:
        return round(min(85, humidity * 1.1), 1)
    elif 'cloud' in desc:
        return round(humidity * 0.8, 1)
    else:
        return round(humidity * 0.5, 1)

def estimate_rainfall_deficit(monthly_rain, normal_rain=800):
    annual_estimate = monthly_rain * 12
    deficit = ((normal_rain - annual_estimate) / normal_rain) * 100
    return round(max(0, min(80, deficit)), 1)