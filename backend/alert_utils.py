import os
import time
import requests
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AlertManager:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.last_alert_time = {
            'trespassing': 0,
            'loitering': 0,
            'crowd': 0
        }
        self.cooldown = 15  # Seconds between alerts of the same type

    def send_telegram_message(self, message):
        """Sends a message to the configured Telegram chat."""
        if not self.bot_token or not self.chat_id:
            print("Error: Telegram credentials not found in env.")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code != 200:
                print(f"Failed to send Telegram alert: {response.text}")
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")

    def process_alerts(self, alerts):
        """Checks alert conditions and triggers notifications if cooldown allows."""
        current_time = time.time()
        
        # Check Trespassing
        if alerts.get('trespassing'):
            if current_time - self.last_alert_time['trespassing'] > self.cooldown:
                self.trigger_alert("🚨 TRESPASSING ALERT! Person detected in restricted zone.", 'trespassing')

        # Check Loitering
        if alerts.get('loitering'):
             if current_time - self.last_alert_time['loitering'] > self.cooldown:
                self.trigger_alert("⚠️ LOITERING ALERT! Suspicious activity detected.", 'loitering')

        # Check Crowd
        if alerts.get('crowd'):
             if current_time - self.last_alert_time['crowd'] > self.cooldown:
                count = alerts.get('count', 0)
                self.trigger_alert(f"👥 CROWD ALERT! High traffic detected. Count: {count}", 'crowd')

    def trigger_alert(self, message, alert_type):
        """Updates cooldown and runs sender in a thread."""
        self.last_alert_time[alert_type] = time.time()
        # Send in a separate thread to not block the video processing loop
        threading.Thread(target=self.send_telegram_message, args=(message,), daemon=True).start()
