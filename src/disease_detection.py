from PIL import Image
import numpy as np
import random

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

# ── Disease Information Database ─────────────
DISEASE_INFO = {
    'Healthy': {
        'severity':    'None',
        'color':       '#388e3c',
        'description': 'Your crop looks healthy! No disease detected.',
        'treatment':   'Continue regular care and monitoring.',
        'prevention':  [
            'Maintain proper spacing between plants',
            'Ensure good drainage',
            'Regular monitoring for early detection',
            'Use disease-resistant varieties',
        ]
    },
    'Early Blight': {
        'severity':    'Moderate',
        'color':       '#f57c00',
        'description': 'Early blight is a fungal disease causing dark spots on leaves.',
        'treatment':   'Apply copper-based fungicide every 7-10 days.',
        'prevention':  [
            'Remove infected leaves immediately',
            'Avoid overhead watering',
            'Apply neem oil spray as preventive',
            'Maintain good air circulation',
        ]
    },
    'Late Blight': {
        'severity':    'Severe',
        'color':       '#d32f2f',
        'description': 'Late blight is a serious disease that can destroy entire crop.',
        'treatment':   'Apply Mancozeb or Metalaxyl fungicide immediately.',
        'prevention':  [
            'Remove and destroy infected plants',
            'Apply fungicide before rain season',
            'Use certified disease-free seeds',
            'Crop rotation every season',
        ]
    },
    'Leaf Rust': {
        'severity':    'Moderate',
        'color':       '#f57c00',
        'description': 'Leaf rust appears as orange-brown pustules on leaves.',
        'treatment':   'Apply Propiconazole or Tebuconazole fungicide.',
        'prevention':  [
            'Plant resistant varieties',
            'Remove infected crop debris',
            'Apply fungicide at first sign',
            'Avoid excessive nitrogen fertilizer',
        ]
    },
    'Powdery Mildew': {
        'severity':    'Mild',
        'color':       '#fbc02d',
        'description': 'White powdery coating on leaves reducing photosynthesis.',
        'treatment':   'Apply sulfur-based fungicide or neem oil.',
        'prevention':  [
            'Improve air circulation',
            'Avoid overhead irrigation',
            'Apply potassium bicarbonate spray',
            'Remove affected leaves',
        ]
    },
    'Bacterial Blight': {
        'severity':    'Severe',
        'color':       '#d32f2f',
        'description': 'Bacterial infection causing water-soaked lesions on leaves.',
        'treatment':   'Apply copper oxychloride spray.',
        'prevention':  [
            'Use disease-free certified seeds',
            'Avoid working in wet fields',
            'Destroy infected plant material',
            'Practice crop rotation',
        ]
    },
    'Nutrient Deficiency': {
        'severity':    'Mild',
        'color':       '#fbc02d',
        'description': 'Yellow or pale leaves indicate nutrient deficiency.',
        'treatment':   'Apply balanced NPK fertilizer.',
        'prevention':  [
            'Regular soil testing',
            'Apply micronutrients as needed',
            'Maintain proper soil pH',
            'Use organic compost regularly',
        ]
    }
}

# ── CNN Model ─────────────────────────────────
if TORCH_AVAILABLE:
    class CropDiseaseCNN(nn.Module):
        def __init__(self, num_classes=7):
            super(CropDiseaseCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(128),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128 * 16 * 16, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

# ── Image Preprocessing ───────────────────────
def preprocess_image(image):
    if not TORCH_AVAILABLE:
        return None
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)

# ── Image Color Analysis ──────────────────────
def analyze_image_features(image):
    img_array = np.array(image.resize((128, 128)))
    if len(img_array.shape) == 3:
        r_mean = img_array[:,:,0].mean()
        g_mean = img_array[:,:,1].mean()
        b_mean = img_array[:,:,2].mean()
    else:
        r_mean = g_mean = b_mean = img_array.mean()
    variance = img_array.std()
    return r_mean, g_mean, b_mean, variance

# ── Disease Prediction ────────────────────────
def predict_disease(image):
    r, g, b, variance = analyze_image_features(image)

    scores = {}

    # Healthy: high green ratio
    green_ratio = g / (r + g + b + 1)
    scores['Healthy'] = green_ratio * 100

    # Early Blight: brownish
    brown_score = (r * 0.6 + b * 0.2) / (g + 1)
    scores['Early Blight'] = min(brown_score * 30, 95)

    # Late Blight: dark patches
    dark_score = (255 - (r + g + b) / 3) / 255
    scores['Late Blight'] = dark_score * 70

    # Leaf Rust: orange tones
    orange_score = r / (g + b + 1)
    scores['Leaf Rust'] = min(orange_score * 40, 90)

    # Powdery Mildew: white/gray
    white_score = min(r, g, b) / (max(r, g, b) + 1)
    scores['Powdery Mildew'] = white_score * 60

    # Bacterial Blight: yellow-brown
    yellow_score = (r + g) / (2 * b + 1)
    scores['Bacterial Blight'] = min(
        yellow_score * 20, 85)

    # Nutrient Deficiency: pale yellow
    pale_score = (r + g) / (b + 150)
    scores['Nutrient Deficiency'] = min(
        pale_score * 25, 80)

    # Add slight randomness for demo variety
    for key in scores:
        scores[key] += random.uniform(-5, 5)
        scores[key] = max(0, min(100, scores[key]))

    # Normalize to probabilities
    total = sum(scores.values())
    probabilities = {
        k: v / total * 100
        for k, v in scores.items()
    }

    # Get top prediction
    predicted   = max(probabilities,
                      key=probabilities.get)
    confidence  = probabilities[predicted]

    # Sort all probabilities
    sorted_probs = dict(sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    return predicted, confidence, sorted_probs

# ── Multilingual Advice ───────────────────────
def get_disease_advice_marathi(disease_name):
    advice = {
        'Healthy':
            'तुमचे पीक निरोगी आहे! नियमित काळजी घ्या.',
        'Early Blight':
            'तांबे-आधारित बुरशीनाशक फवारा. संक्रमित पाने काढा.',
        'Late Blight':
            'मॅन्कोझेब किंवा मेटॅलॅक्सिल लगेच फवारा.',
        'Leaf Rust':
            'प्रोपिकोनाझोल बुरशीनाशक वापरा.',
        'Powdery Mildew':
            'सल्फर-आधारित बुरशीनाशक किंवा कडुलिंबाचे तेल फवारा.',
        'Bacterial Blight':
            'कॉपर ऑक्सीक्लोराइड फवारा. ओल्या शेतात काम करू नका.',
        'Nutrient Deficiency':
            'NPK खत द्या. माती परीक्षण करा.',
    }
    return advice.get(disease_name,
                      'कृषी तज्ञाचा सल्ला घ्या.')

def get_disease_advice_hindi(disease_name):
    advice = {
        'Healthy':
            'आपकी फसल स्वस्थ है! नियमित देखभाल जारी रखें.',
        'Early Blight':
            'तांबे आधारित फफूंदनाशक का छिड़काव करें.',
        'Late Blight':
            'मैन्कोजेब या मेटालैक्सिल तुरंत लगाएं.',
        'Leaf Rust':
            'प्रोपिकोनाजोल फफूंदनाशक का उपयोग करें.',
        'Powdery Mildew':
            'सल्फर आधारित फफूंदनाशक या नीम तेल छिड़कें.',
        'Bacterial Blight':
            'कॉपर ऑक्सीक्लोराइड का छिड़काव करें.',
        'Nutrient Deficiency':
            'NPK उर्वरक दें. मिट्टी परीक्षण कराएं.',
    }
    return advice.get(disease_name,
                      'कृषि विशेषज्ञ से सलाह लें.')