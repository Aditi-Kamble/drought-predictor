import random

# Simple keyword-based NLP without spacy dependency
def simple_tokenize(text):
    return text.lower().split()

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
            "Avoid water-intensive crops. Focus on Bajra — it needs 40% less water than Rice.",
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
    "Sprinkler irrigation is 20-30% more efficient than traditional flood methods.",
]

GOVERNMENT_SCHEMES = [
    "PM Fasal Bima Yojana — crop insurance scheme for drought losses. Visit pmfby.gov.in",
    "PM Krishi Sinchai Yojana — subsidized drip/sprinkler irrigation. Contact local agriculture office.",
    "Drought Relief Fund — contact your District Collector office for compensation.",
    "Kisan Credit Card — emergency credit for farmers during drought. Visit any nationalized bank.",
    "MGNREGA — guaranteed 100 days of work during drought for farm laborers.",
]

SOIL_TIPS = [
    "Add organic compost to improve soil water retention during drought.",
    "Deep ploughing before monsoon helps store more rainwater in soil.",
    "Green manuring with Dhaincha improves soil moisture holding capacity.",
    "Avoid tilling dry soil — it increases moisture evaporation.",
    "Test your soil pH — drought-affected soils often become more alkaline.",
]

WEATHER_TIPS = [
    "Monitor IMD forecasts daily at mausam.imd.gov.in",
    "Install a simple rain gauge on your farm to track local rainfall accurately.",
    "Join your local Kisan WhatsApp group for real-time weather alerts.",
    "Agromet Advisory Services by IMD provide crop-specific weather guidance.",
]

def detect_intent(text):
    text_lower = text.lower()

    crop_keywords       = ['crop', 'plant', 'grow', 'cultivate', 'seed', 'farming', 'harvest', 'sow']
    irrigation_keywords = ['water', 'irrigation', 'irrigate', 'drip', 'sprinkler', 'moisture']
    drought_keywords    = ['drought', 'dry', 'rain', 'rainfall', 'monsoon', 'shortage', 'deficit']
    scheme_keywords     = ['scheme', 'government', 'insurance', 'compensation', 'subsidy', 'loan', 'yojana']
    soil_keywords       = ['soil', 'land', 'earth', 'compost', 'fertilizer', 'manure']
    weather_keywords    = ['weather', 'forecast', 'temperature', 'climate', 'imd']
    greeting_keywords   = ['hello', 'hi', 'hey', 'namaste', 'help']

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

def get_response(user_input, drought_level='moderate'):
    intent = detect_intent(user_input)
    level  = extract_drought_level(user_input) or drought_level

    if intent == 'greeting':
        return ("Namaste Kisan! I am your AI Farming Assistant.\n"
                "I can help you with:\n"
                "  Crop recommendations for drought conditions\n"
                "  Irrigation tips to save water\n"
                "  Government schemes for drought relief\n"
                "  Soil management advice\n"
                "  Weather forecast guidance\n\n"
                "What would you like to know today?")

    elif intent == 'crop':
        info   = CROP_ADVICE[level]
        crops  = ', '.join(info['crops'])
        advice = random.choice(info['advice'])
        return (f"Crop Recommendation ({level.title()} Drought):\n\n"
                f"Best crops: {crops}\n\n"
                f"Advice: {advice}")

    elif intent == 'irrigation':
        return f"Irrigation Tip:\n\n{random.choice(IRRIGATION_TIPS)}"

    elif intent == 'scheme':
        return f"Government Scheme:\n\n{random.choice(GOVERNMENT_SCHEMES)}"

    elif intent == 'soil':
        return f"Soil Management Tip:\n\n{random.choice(SOIL_TIPS)}"

    elif intent == 'weather':
        return f"Weather Advisory:\n\n{random.choice(WEATHER_TIPS)}"

    elif intent == 'drought_info':
        info = CROP_ADVICE[level]
        return (f"Drought Information ({level.title()} Level):\n\n"
                f"Recommended crops: {', '.join(info['crops'])}\n"
                f"Action: {random.choice(info['advice'])}\n\n"
                f"Water tip: {random.choice(IRRIGATION_TIPS)}")

    else:
        return ("Try asking me:\n"
                "  'What crops should I grow?'\n"
                "  'How to save water?'\n"
                "  'What government schemes are available?'\n"
                "  'How to manage soil in drought?'")

def run_chatbot():
    print("="*55)
    print("   AI FARMER CHATBOT - Drought Advisory System")
    print("="*55)
    current_drought = 'moderate'
    print(f"\nBot: Namaste Kisan! Drought level: {current_drought}\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit']:
            print("\nBot: Dhanyavaad! Good luck with your farming!")
            break
        print(f"\nBot: {get_response(user_input, current_drought)}\n")

if __name__ == '__main__':
    run_chatbot()