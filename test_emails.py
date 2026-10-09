from flask import Flask 
import os 
from flask_mail import Mail, Message
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
if not os.getenv("MAIL_USERNAME") or not os.getenv("MAIL_PASSWORD") or not os.getenv("ADMIN_EMAIL"):
  raise ValueError("Missing email settings. Check your .env")
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_USERNAME")

mail = Mail(app)

with app.app_context():
  message  = Message(
    subject="Healtcare Prediction Email test",
    recipients=[os.environ["ADMIN_EMAIL"]],
    body="Email configuration is working"
  )
  mail.send(message)

print("working")