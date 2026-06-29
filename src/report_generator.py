from fpdf import FPDF
import datetime

def clean_text(text):
    if not text:
        return "Unknown"
    replacements = {
        '\u2014': '-',
        '\u2013': '-',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2022': '-',
        '\u00b0': ' degrees',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode(
        'ascii', 'ignore').decode('ascii').strip()

class DroughtReport(FPDF):
    def header(self):
        self.set_fill_color(45, 90, 39)
        self.rect(0, 0, 210, 25, 'F')
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.cell(0, 25,
                  'AI Drought Predictor Report',
                  align='C')
        self.ln(30)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10,
            f'Generated on '
            f'{datetime.datetime.now().strftime("%d %B %Y")}'
            f' | AI Drought Predictor',
            align='C')

def generate_report(city, weather_data, ml_label,
                    ml_conf, dl_label, dl_conf,
                    crops, deficit):
    pdf = DroughtReport()
    pdf.add_page()

    city_clean = clean_text(str(city))

    # Title
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_fill_color(232, 245, 232)
    pdf.cell(0, 12,
             'Drought Analysis Report - ' + city_clean,
             fill=True, ln=True, align='C')
    pdf.ln(5)

    # Date
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8,
             'Date: ' + datetime.datetime.now().strftime(
                 "%d %B %Y, %I:%M %p"),
             ln=True)
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)

    # Weather Data
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(74, 158, 63)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'Weather Data',
             fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(3)

    if weather_data:
        weather_items = [
            ('Temperature',
             str(weather_data.get(
                 'temperature', 'N/A')) + ' C'),
            ('Humidity',
             str(weather_data.get(
                 'humidity', 'N/A')) + '%'),
            ('Wind Speed',
             str(weather_data.get(
                 'wind_speed', 'N/A')) + ' km/h'),
            ('Conditions',
             clean_text(str(weather_data.get(
                 'description', 'N/A')))),
        ]
        for label, value in weather_items:
            pdf.cell(80, 8, label + ':', border=0)
            pdf.cell(0, 8,
                     clean_text(str(value)), ln=True)
    pdf.ln(5)

    # Prediction Results
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(74, 158, 63)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'Drought Prediction Results',
             fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(3)

    results = [
        ('ML Model (Random Forest)',
         ml_label + ' (' + str(round(ml_conf, 1))
         + '% confidence)'),
        ('DL Model (Neural Network)',
         dl_label + ' (' + str(round(dl_conf, 1))
         + '% confidence)'),
        ('Rainfall Deficit',
         str(deficit) + '%'),
    ]
    for label, value in results:
        pdf.cell(80, 8, label + ':', border=0)
        pdf.cell(0, 8,
                 clean_text(str(value)), ln=True)
    pdf.ln(5)

    # Severity
    severity_colors = {
        'Normal':   (56,  142, 60),
        'Mild':     (251, 192, 45),
        'Moderate': (245, 124,  0),
        'Severe':   (211,  47, 47),
    }
    color = severity_colors.get(ml_label, (0, 0, 0))
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 14,
             'Drought Level: ' + ml_label,
             fill=True, ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Crop Recommendations
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(74, 158, 63)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'Recommended Crops',
             fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(3)

    for i, crop in enumerate(crops, 1):
        pdf.cell(0, 8,
                 str(i) + '. ' + clean_text(str(crop)),
                 ln=True)
    pdf.ln(5)

    # Action Items
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(74, 158, 63)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'Immediate Action Items',
             fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(3)

    actions = {
        'Normal': [
            'Continue regular irrigation schedule',
            'Plant water-intensive crops as planned',
            'Monitor weather forecasts regularly',
        ],
        'Mild': [
            'Switch to moderate water crops',
            'Install drip irrigation system',
            'Create water storage in farm',
        ],
        'Moderate': [
            'Plant only drought-tolerant crops',
            'Apply for PM Fasal Bima Yojana',
            'Implement water harvesting',
            'Contact local agriculture office',
        ],
        'Severe': [
            'Plant only Bajra and Moth Bean',
            'Apply for drought relief compensation',
            'Implement emergency water conservation',
            'Contact District Collector office',
        ],
    }
    for action in actions.get(ml_label, []):
        pdf.cell(0, 8,
                 '- ' + clean_text(action), ln=True)
    pdf.ln(5)

    # Government Schemes
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(74, 158, 63)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'Government Schemes',
             fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(3)

    schemes = [
        'PM Fasal Bima Yojana - pmfby.gov.in',
        'PM Krishi Sinchai Yojana - Agri Office',
        'Kisan Credit Card - Nationalized Bank',
        'MGNREGA - District Collector Office',
    ]
    for scheme in schemes:
        pdf.cell(0, 8,
                 '- ' + clean_text(scheme), ln=True)

    # Footer note
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 6,
        'This report is generated by AI Drought '
        'Predictor. Please consult local agriculture '
        'experts for final decisions. '
        'GitHub: github.com/Aditi-Kamble/drought-predictor')

    return pdf.output()