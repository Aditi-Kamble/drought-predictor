import random
import os
from dotenv import load_dotenv

load_dotenv()

# Try new Google Genai
try:
    from google import genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        try:
            import streamlit as st
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        except:
            GEMINI_API_KEY = ""
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("✅ Gemini AI Active!")
    else:
        GEMINI_AVAILABLE = False
        print("⚠️ Gemini API key not found!")
except Exception as e:
    GEMINI_AVAILABLE = False
    print(f"⚠️ Gemini not available: {e}")

# Knowledge Base
CROP_ADVICE = {
    'normal': {
        'crops': ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize'],
        'advice': [
            "Rainfall is normal this season. You can grow water-intensive crops like Rice and Sugarcane.",
            "Good rainfall expected! Ideal time to plant Wheat or Maize.",
            "Normal conditions — maintain regular irrigation schedule.",
        ]
    },
    'mild': {
        'crops': ['Maize', 'Soybean', 'Groundnut', 'Sunflower', 'Pulses'],
        'advice': [
            "Mild drought detected. Shift to moderate water crops like Maize or Soybean.",
            "Consider Groundnut or Sunflower — they need less water than Rice.",
            "Use drip irrigation to conserve water. Pulses are a good choice now.",
        ]
    },
    'moderate': {
        'crops': ['Bajra', 'Jowar', 'Moong', 'Moth Bean', 'Cluster Bean'],
        'advice': [
            "Moderate drought! Switch to drought-tolerant crops like Bajra or Jowar.",
            "Moong and Moth Bean survive well in low water conditions.",
            "Avoid water-intensive crops. Focus on Bajra.",
        ]
    },
    'severe': {
        'crops': ['Bajra', 'Moth Bean', 'Cactus Pear', 'Amaranth'],
        'advice': [
            "Severe drought alert! Only grow highly drought-resistant crops like Bajra.",
            "Moth Bean and Amaranth can survive extreme dry conditions.",
            "Consider leaving fields fallow and focusing on soil conservation.",
        ]
    }
}

IRRIGATION_TIPS = [
    "Use drip irrigation — saves up to 50% water compared to flood irrigation.",
    "Water crops early morning (5-7 AM) to reduce evaporation losses.",
    "Mulching around plants reduces soil moisture loss by 30%.",
    "Check soil moisture before irrigating — do not water if soil is still moist.",
    "Collect rainwater in farm ponds for use during dry spells.",
]

GOVERNMENT_SCHEMES = [
    "PM Fasal Bima Yojana — crop insurance for drought losses. Visit pmfby.gov.in",
    "PM Krishi Sinchai Yojana — subsidized drip/sprinkler irrigation.",
    "Kisan Credit Card — emergency credit for farmers during drought.",
    "MGNREGA — guaranteed 100 days of work during drought.",
    "Drought Relief Fund — contact your District Collector office.",
]

SOIL_TIPS = [
    "Add organic compost to improve soil water retention during drought.",
    "Deep ploughing before monsoon helps store more rainwater in soil.",
    "Avoid tilling dry soil — it increases moisture evaporation.",
    "Test your soil pH — drought-affected soils often become more alkaline.",
    "Green manuring with Dhaincha improves soil moisture holding capacity.",
]

WEATHER_TIPS = [
    "Monitor IMD forecasts daily at mausam.imd.gov.in",
    "Install a simple rain gauge on your farm to track local rainfall.",
    "Join your local Kisan WhatsApp group for real-time weather alerts.",
    "Agromet Advisory Services by IMD provide crop-specific weather guidance.",
]

# Gemini Response
def get_gemini_response(user_input, drought_level):
    try:
        prompt = f"""You are an expert AI farming assistant for Indian farmers.

Current drought level in the farmer's area: {drought_level.upper()}

Farmer's question: {user_input}

Please provide helpful, practical advice specific to Indian farming conditions.
Keep your response concise (3-5 sentences), friendly and easy to understand.
Focus on actionable advice the farmer can implement immediately.
If recommending crops, suggest ones suitable for {drought_level} drought 
conditions in India.
End with one encouraging sentence for the farmer."""

        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return None

# Intent Detection
def detect_intent(text):
    text_lower = text.lower()
    crop_keywords       = ['crop', 'plant', 'grow', 'cultivate', 'seed',
                           'farming', 'harvest', 'sow', 'vegetables']
    irrigation_keywords = ['water', 'irrigation', 'irrigate', 'drip',
                           'sprinkler', 'moisture', 'wet']
    drought_keywords    = ['drought', 'dry', 'rain', 'rainfall',
                           'monsoon', 'shortage', 'deficit', 'arid']
    scheme_keywords     = ['scheme', 'government', 'insurance', 'compensation',
                           'subsidy', 'loan', 'yojana', 'help', 'money']
    soil_keywords       = ['soil', 'land', 'earth', 'compost',
                           'fertilizer', 'manure', 'nutrients']
    weather_keywords    = ['weather', 'forecast', 'temperature',
                           'climate', 'imd', 'prediction']
    greeting_keywords   = ['hello', 'hi', 'hey', 'namaste',
                           'good morning', 'help']

    if any(w in text_lower for w in greeting_keywords):
        return 'greeting'
    if any(w in text_lower for w in scheme_keywords):
        return 'scheme'
    if any(w in text_lower for w in soil_keywords):
        return 'soil'
    if any(w in text_lower for w in weather_keywords):
        return 'weather'
    if any(w in text_lower for w in irrigation_keywords):
        return 'irrigation'
    if any(w in text_lower for w in drought_keywords):
        return 'drought_info'
    if any(w in text_lower for w in crop_keywords):
        return 'crop'
    return 'unknown'

def extract_drought_level(text):
    text_lower = text.lower()
    if 'severe'   in text_lower: return 'severe'
    if 'moderate' in text_lower: return 'moderate'
    if 'mild'     in text_lower: return 'mild'
    if 'normal'   in text_lower: return 'normal'
    return None

# Fallback Response
def get_fallback_response(user_input, drought_level):
    intent = detect_intent(user_input)
    level  = extract_drought_level(user_input) or drought_level

    if intent == 'greeting':
        return ("Namaste Kisan! I am your AI Farming Assistant.\n"
                "I can help you with:\n"
                "  Crop recommendations for drought conditions\n"
                "  Irrigation tips to save water\n"
                "  Government schemes for drought relief\n"
                "  Soil management advice\n\n"
                "What would you like to know today?")
    elif intent == 'crop':
        info  = CROP_ADVICE[level]
        crops = ', '.join(info['crops'])
        return (f"Crop Recommendation ({level.title()} Drought):\n\n"
                f"Best crops: {crops}\n\n"
                f"{random.choice(info['advice'])}")
    elif intent == 'irrigation':
        return f"Irrigation Tip:\n\n{random.choice(IRRIGATION_TIPS)}"
    elif intent == 'scheme':
        return f"Government Scheme:\n\n{random.choice(GOVERNMENT_SCHEMES)}"
    elif intent == 'soil':
        return f"Soil Management:\n\n{random.choice(SOIL_TIPS)}"
    elif intent == 'weather':
        return f"Weather Advisory:\n\n{random.choice(WEATHER_TIPS)}"
    elif intent == 'drought_info':
        info = CROP_ADVICE[level]
        return (f"Drought Info ({level.title()}):\n\n"
                f"Recommended: {', '.join(info['crops'])}\n"
                f"{random.choice(info['advice'])}\n\n"
                f"Water tip: {random.choice(IRRIGATION_TIPS)}")
    else:
        return ("Try asking me:\n"
                "  'What crops should I grow?'\n"
                "  'How to save water?'\n"
                "  'What government schemes are available?'\n"
                "  'How to manage soil in drought?'")

# Main Response Function
def get_response(user_input, drought_level='moderate'):
    # Check if Hindi
    from src.hindi_support import is_hindi, get_hindi_response
    if is_hindi(user_input):
        # Try Gemini for Hindi first
        if GEMINI_AVAILABLE:
            hindi_prompt = f"""आप एक भारतीय किसान सहायक हैं।
सूखे का स्तर: {drought_level}
किसान का प्रश्न: {user_input}
कृपया हिंदी में संक्षिप्त और व्यावहारिक सलाह दें।
3-4 वाक्यों में उत्तर दें।"""
            try:
                response = gemini_client.models.generate_content(
                    model='gemini-2.0-flash-lite',
                    contents=hindi_prompt
                )
                if response.text:
                    return response.text
            except:
                pass
        return get_hindi_response(
            user_input, drought_level, CROP_ADVICE)

    # English — Try Gemini first
    if GEMINI_AVAILABLE:
        gemini_response = get_gemini_response(
            user_input, drought_level)
        if gemini_response:
            return gemini_response

    # Fallback
    return get_fallback_response(user_input, drought_level)

# Terminal Chatbot
def run_chatbot():
    print("="*55)
    print("   AI FARMER CHATBOT — Powered by Gemini AI")
    print("="*55)

    current_drought = 'moderate'
    print(f"\nBot: Namaste Kisan! Drought level: {current_drought}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nBot: Dhanyavaad! Good luck with your farming! 🌾")
            break
        print(f"\nBot: {get_response(user_input, current_drought)}\n")
        print("-"*55)

if __name__ == '__main__':
    run_chatbot()