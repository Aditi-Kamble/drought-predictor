# Hindi Language Support for Farmer Chatbot

# Hindi to English translation dictionary
HINDI_TO_ENGLISH = {
    # Greetings
    'नमस्ते': 'hello',
    'हेलो': 'hello',
    'हाय': 'hi',
    'मदद': 'help',

    # Crops
    'फसल': 'crop',
    'फसलें': 'crops',
    'कौन सी फसल': 'which crop',
    'क्या उगाएं': 'what to grow',
    'बीज': 'seed',
    'खेती': 'farming',
    'बुवाई': 'sow',
    'कटाई': 'harvest',

    # Water & Irrigation
    'पानी': 'water',
    'सिंचाई': 'irrigation',
    'ड्रिप': 'drip irrigation',
    'नमी': 'moisture',
    'बारिश': 'rainfall',
    'वर्षा': 'rainfall',

    # Drought
    'सूखा': 'drought',
    'सूखे में': 'in drought',
    'पानी की कमी': 'water shortage',
    'अकाल': 'drought',
    'कम बारिश': 'low rainfall',

    # Soil
    'मिट्टी': 'soil',
    'भूमि': 'land',
    'खाद': 'fertilizer',
    'कम्पोस्ट': 'compost',
    'जैविक': 'organic',

    # Government
    'सरकारी योजना': 'government scheme',
    'योजना': 'scheme',
    'बीमा': 'insurance',
    'मुआवजा': 'compensation',
    'सब्सिडी': 'subsidy',
    'कर्ज': 'loan',
    'किसान क्रेडिट': 'kisan credit',

    # Weather
    'मौसम': 'weather',
    'तापमान': 'temperature',
    'जलवायु': 'climate',
    'पूर्वानुमान': 'forecast',

    # Common words
    'कैसे': 'how',
    'क्या': 'what',
    'कब': 'when',
    'कहाँ': 'where',
    'बताओ': 'tell me',
    'बताइए': 'please tell',
    'जानकारी': 'information',
    'सलाह': 'advice',
    'अच्छा': 'good',
    'बुरा': 'bad',
}

# Hindi crop names
CROPS_IN_HINDI = {
    'Rice':         'धान (Rice)',
    'Wheat':        'गेहूं (Wheat)',
    'Sugarcane':    'गन्ना (Sugarcane)',
    'Cotton':       'कपास (Cotton)',
    'Maize':        'मक्का (Maize)',
    'Bajra':        'बाजरा (Bajra)',
    'Jowar':        'ज्वार (Jowar)',
    'Moong':        'मूंग (Moong)',
    'Moth Bean':    'मोठ (Moth Bean)',
    'Cluster Bean': 'ग्वार (Cluster Bean)',
    'Soybean':      'सोयाबीन (Soybean)',
    'Groundnut':    'मूंगफली (Groundnut)',
    'Sunflower':    'सूरजमुखी (Sunflower)',
    'Pulses':       'दालें (Pulses)',
    'Amaranth':     'राजगिरा (Amaranth)',
}

# Hindi responses
HINDI_RESPONSES = {
    'greeting': """नमस्ते किसान भाई! 🌾
मैं आपका AI कृषि सहायक हूं।
मैं इन विषयों में मदद कर सकता हूं:
  • फसल की सलाह
  • सिंचाई के टिप्स
  • सरकारी योजनाएं
  • मिट्टी प्रबंधन
  • मौसम की जानकारी

आप क्या जानना चाहते हैं?""",

    'irrigation': [
        "ड्रिप सिंचाई का उपयोग करें — बाढ़ सिंचाई की तुलना में 50% पानी बचता है।",
        "सुबह 5-7 बजे फसलों को पानी दें — वाष्पीकरण कम होता है।",
        "पौधों के आसपास मल्चिंग करें — मिट्टी की नमी 30% तक बचती है।",
        "सिंचाई से पहले मिट्टी की नमी जांचें।",
        "खेत में बारिश का पानी संग्रहित करें।",
    ],

    'scheme': [
        "PM फसल बीमा योजना — सूखे से फसल नुकसान का बीमा। pmfby.gov.in पर जाएं।",
        "PM कृषि सिंचाई योजना — ड्रिप/स्प्रिंकलर पर सब्सिडी।",
        "किसान क्रेडिट कार्ड — सूखे में आपातकालीन ऋण।",
        "MGNREGA — सूखे में 100 दिन के काम की गारंटी।",
    ],

    'soil': [
        "जैविक खाद डालें — मिट्टी की पानी रोकने की क्षमता बढ़ती है।",
        "मानसून से पहले गहरी जुताई करें।",
        "सूखी मिट्टी में जुताई न करें — नमी वाष्पित होती है।",
        "मिट्टी का pH परीक्षण करें।",
    ],

    'weather': [
        "IMD का मौसम पूर्वानुमान रोज देखें: mausam.imd.gov.in",
        "अपने खेत में वर्षामापी लगाएं।",
        "किसान WhatsApp ग्रुप से जुड़ें — मौसम अलर्ट मिलेंगे।",
    ],

    'unknown': """मुझे समझ नहीं आया। कृपया इस तरह पूछें:
  • 'कौन सी फसल उगाएं?'
  • 'पानी कैसे बचाएं?'
  • 'सरकारी योजना क्या है?'
  • 'मिट्टी की देखभाल कैसे करें?'"""
}

def is_hindi(text):
    hindi_chars = 0
    for char in text:
        if '\u0900' <= char <= '\u097F':
            hindi_chars += 1
    return hindi_chars > 0

def translate_hindi_to_english(text):
    text_lower = text.lower()
    for hindi, english in HINDI_TO_ENGLISH.items():
        text_lower = text_lower.replace(hindi, english)
    return text_lower

def get_hindi_crop_response(level, crop_advice):
    info  = crop_advice.get(level, crop_advice['moderate'])
    crops = info['crops']
    hindi_crops = [CROPS_IN_HINDI.get(c, c) for c in crops]

    drought_names = {
        'normal':   'सामान्य',
        'mild':     'हल्का सूखा',
        'moderate': 'मध्यम सूखा',
        'severe':   'गंभीर सूखा'
    }

    advice_hindi = {
        'normal':   'इस मौसम में धान और गेहूं अच्छी फसल देंगे।',
        'mild':     'मक्का और सोयाबीन कम पानी में भी अच्छे होते हैं।',
        'moderate': 'बाजरा और ज्वार सूखे में सबसे अच्छे हैं।',
        'severe':   'केवल बाजरा और मोठ जैसी सूखा-प्रतिरोधी फसलें लगाएं।'
    }

    return (f"फसल सलाह ({drought_names.get(level, level)}):\n\n"
            f"अनुशंसित फसलें: {', '.join(hindi_crops)}\n\n"
            f"सलाह: {advice_hindi.get(level, '')}")

def detect_hindi_intent(text):
    text_lower = text.lower()

    crop_words       = ['फसल', 'उगाएं', 'बोएं', 'खेती',
                        'बीज', 'कटाई', 'crop', 'grow']
    irrigation_words = ['पानी', 'सिंचाई', 'ड्रिप',
                        'नमी', 'water', 'irrigation']
    drought_words    = ['सूखा', 'अकाल', 'बारिश', 'वर्षा',
                        'drought', 'rain']
    scheme_words     = ['योजना', 'बीमा', 'सब्सिडी', 'कर्ज',
                        'scheme', 'government']
    soil_words       = ['मिट्टी', 'भूमि', 'खाद',
                        'soil', 'fertilizer']
    weather_words    = ['मौसम', 'तापमान', 'जलवायु',
                        'weather', 'forecast']
    greeting_words   = ['नमस्ते', 'हेलो', 'हाय', 'मदद',
                        'hello', 'hi', 'help']

    if any(w in text for w in greeting_words):
        return 'greeting'
    if any(w in text for w in scheme_words):
        return 'scheme'
    if any(w in text for w in soil_words):
        return 'soil'
    if any(w in text for w in weather_words):
        return 'weather'
    if any(w in text for w in irrigation_words):
        return 'irrigation'
    if any(w in text for w in drought_words):
        return 'drought_info'
    if any(w in text for w in crop_words):
        return 'crop'
    return 'unknown'

import random

def get_hindi_response(user_input, drought_level, crop_advice):
    intent = detect_hindi_intent(user_input)

    if intent == 'greeting':
        return HINDI_RESPONSES['greeting']
    elif intent == 'crop':
        return get_hindi_crop_response(drought_level, crop_advice)
    elif intent == 'irrigation':
        return random.choice(HINDI_RESPONSES['irrigation'])
    elif intent == 'scheme':
        return random.choice(HINDI_RESPONSES['scheme'])
    elif intent == 'soil':
        return random.choice(HINDI_RESPONSES['soil'])
    elif intent == 'weather':
        return random.choice(HINDI_RESPONSES['weather'])
    else:
        return HINDI_RESPONSES['unknown']