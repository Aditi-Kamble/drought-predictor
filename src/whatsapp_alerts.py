from twilio.rest import Client
import os

def send_whatsapp_alert(to_number, drought_level,
                         city, crops):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN", "")

    if not account_sid or not auth_token:
        return False, "Twilio credentials not found"

    try:
        client = Client(account_sid, auth_token)

        emoji = {
            'Normal':   '🟢',
            'Mild':     '🟡',
            'Moderate': '🟠',
            'Severe':   '🔴'
        }.get(drought_level, '🟡')

        message_body = f"""🌾 *AI Drought Alert*

{emoji} *Drought Level: {drought_level}*
📍 Location: {city}

🌱 *Recommended Crops:*
{chr(10).join(f'• {c}' for c in crops[:3])}

💡 *Quick Tips:*
- Use drip irrigation
- Store rainwater
- Apply for PM Fasal Bima Yojana

🔗 Check full report:
aditi-kamble-drought-predictor.streamlit.app

_Sent by AI Drought Predictor_"""

        message = client.messages.create(
            body=message_body,
            from_='whatsapp:+14155238886',
            to=f'whatsapp:+91{to_number}'
        )
        return True, message.sid

    except Exception as e:
        return False, str(e)