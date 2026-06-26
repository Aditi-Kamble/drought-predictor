import random

# Marathi crop names
CROPS_IN_MARATHI = {
    'Rice':         'भात (Rice)',
    'Wheat':        'गहू (Wheat)',
    'Sugarcane':    'ऊस (Sugarcane)',
    'Cotton':       'कापूस (Cotton)',
    'Maize':        'मका (Maize)',
    'Bajra':        'बाजरी (Bajra)',
    'Jowar':        'ज्वारी (Jowar)',
    'Moong':        'मूग (Moong)',
    'Moth Bean':    'मटकी (Moth Bean)',
    'Cluster Bean': 'गवार (Cluster Bean)',
    'Soybean':      'सोयाबीन (Soybean)',
    'Groundnut':    'भुईमूग (Groundnut)',
    'Sunflower':    'सूर्यफूल (Sunflower)',
    'Pulses':       'डाळी (Pulses)',
    'Amaranth':     'राजगिरा (Amaranth)',
}

MARATHI_RESPONSES = {
    'greeting': """नमस्कार शेतकरी बंधू! 🌾
मी तुमचा AI कृषी सहाय्यक आहे।
मी या विषयांमध्ये मदत करू शकतो:
  • पीक सल्ला
  • सिंचन टिप्स
  • सरकारी योजना
  • माती व्यवस्थापन
  • हवामान माहिती

तुम्हाला काय जाणून घ्यायचे आहे?""",

    'irrigation': [
        "ठिबक सिंचन वापरा — पूर सिंचनापेक्षा 50% पाणी वाचते।",
        "सकाळी 5-7 वाजता पिकांना पाणी द्या — बाष्पीभवन कमी होते।",
        "झाडांभोवती आच्छादन करा — जमिनीतील ओलावा 30% वाचतो।",
        "सिंचनापूर्वी जमिनीतील ओलावा तपासा।",
        "शेतात पावसाचे पाणी साठवा।",
    ],

    'scheme': [
        "PM पीक विमा योजना — दुष्काळात पीक नुकसानीचा विमा। pmfby.gov.in वर जा।",
        "PM कृषी सिंचाई योजना — ठिबक/तुषार सिंचनावर अनुदान।",
        "किसान क्रेडिट कार्ड — दुष्काळात आपत्कालीन कर्ज।",
        "MGNREGA — दुष्काळात 100 दिवसांच्या कामाची हमी।",
        "मागेल त्याला शेततळे — महाराष्ट्र सरकारची योजना।",
    ],

    'soil': [
        "सेंद्रिय खत टाका — जमिनीची पाणी धरण्याची क्षमता वाढते।",
        "पावसाळ्यापूर्वी खोल नांगरणी करा।",
        "कोरड्या जमिनीत नांगरणी करू नका — ओलावा उडून जातो।",
        "जमिनीचा pH तपासा — दुष्काळग्रस्त जमीन अधिक क्षारीय होते।",
        "हिरवळीचे खत वापरा — जमिनीतील ओलावा टिकतो।",
    ],

    'weather': [
        "IMD चे हवामान अंदाज रोज पहा: mausam.imd.gov.in",
        "शेतात पर्जन्यमापक लावा — स्थानिक पाऊस मोजा।",
        "किसान WhatsApp ग्रुपमध्ये सामील व्हा — हवामान अलर्ट मिळतील।",
        "IMD च्या कृषी हवामान सेवेचा लाभ घ्या।",
    ],

    'unknown': """मला समजले नाही. कृपया असे विचारा:
  • 'कोणते पीक घ्यावे?'
  • 'पाणी कसे वाचवावे?'
  • 'सरकारी योजना काय आहे?'
  • 'माती व्यवस्थापन कसे करावे?'"""
}

MARATHI_KEYWORDS = {
    'crop':       ['पीक', 'पिके', 'पेरणी', 'काढणी',
                   'शेती', 'बियाणे', 'लागवड'],
    'irrigation': ['पाणी', 'सिंचन', 'ठिबक', 'ओलावा',
                   'तुषार'],
    'drought':    ['दुष्काळ', 'कोरडा', 'पाऊस', 'पर्जन्य',
                   'टंचाई'],
    'scheme':     ['योजना', 'विमा', 'अनुदान', 'कर्ज',
                   'भरपाई'],
    'soil':       ['माती', 'जमीन', 'खत', 'सेंद्रिय',
                   'कंपोस्ट'],
    'weather':    ['हवामान', 'तापमान', 'पाऊस', 'अंदाज'],
    'greeting':   ['नमस्कार', 'नमस्ते', 'हॅलो', 'मदत',
                   'सांगा'],
}

def is_marathi(text):
    marathi_count = 0
    for char in text:
        if '\u0900' <= char <= '\u097F':
            marathi_count += 1
    # Check Marathi specific keywords
    for word in MARATHI_KEYWORDS['greeting'] + \
                MARATHI_KEYWORDS['crop']:
        if word in text:
            return True
    return marathi_count > 2

def detect_marathi_intent(text):
    for intent, keywords in MARATHI_KEYWORDS.items():
        if any(w in text for w in keywords):
            return intent
    return 'unknown'

def get_marathi_crop_response(level):
    from src.hindi_support import CROPS_IN_HINDI

    crops_map = {
        'normal':   ['भात', 'गहू', 'ऊस', 'कापूस', 'मका'],
        'mild':     ['मका', 'सोयाबीन', 'भुईमूग',
                     'सूर्यफूल', 'डाळी'],
        'moderate': ['बाजरी', 'ज्वारी', 'मूग',
                     'मटकी', 'गवार'],
        'severe':   ['बाजरी', 'मटकी', 'राजगिरा'],
    }

    advice_map = {
        'normal':   'या हंगामात भात व गहू चांगले उत्पन्न देतील।',
        'mild':     'मका आणि सोयाबीन कमी पाण्यातही चांगले होतात।',
        'moderate': 'बाजरी आणि ज्वारी दुष्काळात सर्वोत्तम आहेत।',
        'severe':   'फक्त बाजरी आणि मटकीसारखी दुष्काळप्रतिरोधक पिके घ्या।',
    }

    drought_names = {
        'normal':   'सामान्य',
        'mild':     'सौम्य दुष्काळ',
        'moderate': 'मध्यम दुष्काळ',
        'severe':   'तीव्र दुष्काळ',
    }

    crops = crops_map.get(level, crops_map['moderate'])
    return (f"पीक सल्ला ({drought_names.get(level, level)}):\n\n"
            f"शिफारस केलेली पिके: {', '.join(crops)}\n\n"
            f"सल्ला: {advice_map.get(level, '')}")

def get_marathi_response(user_input, drought_level):
    intent = detect_marathi_intent(user_input)

    if intent == 'greeting':
        return MARATHI_RESPONSES['greeting']
    elif intent == 'crop':
        return get_marathi_crop_response(drought_level)
    elif intent == 'irrigation':
        return random.choice(MARATHI_RESPONSES['irrigation'])
    elif intent == 'scheme':
        return random.choice(MARATHI_RESPONSES['scheme'])
    elif intent == 'soil':
        return random.choice(MARATHI_RESPONSES['soil'])
    elif intent == 'weather':
        return random.choice(MARATHI_RESPONSES['weather'])
    else:
        return MARATHI_RESPONSES['unknown']