from flask import Flask, request, render_template, send_from_directory
import pickle
import numpy as np
import os 
from database import init_db, get_connection
import json
from datetime import datetime, timezone
from flask_mail import Mail, Message
from dotenv import load_dotenv
import secrets 
import hashlib
from datetime import timedelta
from flask import url_for, abort

app = Flask(__name__)

load_dotenv()
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_USERNAME")

mail = Mail(app)

def save_prediction_report(disease, features, prediction_result):
    connection = get_connection()
    try:
        cursor = connection.execute(
            """
                INSERT INTO prediction_reports (
                    disease, 
                    patient_inputs,
                    prediction,
                    prediction_result,
                    status, 
                    created_at
                )
                VALUES(?, ?, ?, ?, 'PENDING', ?)
            """,
            (
                disease,
                json.dumps(features),
                prediction_result,
                prediction_result,
                datetime.now(timezone.utc).isoformat()
            )
        )
        connection.commit()
        report_id = cursor.lastrowid
    finally:
        connection.close()
    return report_id

def send_prediction_email(report_id, disease, features, prediction_result):
    admin_email = os.getenv("ADMIN_EMAIL")
    if not admin_email:
        print("Email not sent: ADMIN_EMAIL is missing")
        return
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=2)
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE prediction_reports
            SET action_token_hash = ?, expires_at = ? 
            WHERE id = ? AND status = 'PENDING'
            """,
            (token_hash, expires_at.isoformat(), report_id),
        )
        connection.commit()
    finally:
        connection.close()
    
    try:
        approve_url = url_for(
            "decide_prediction", report_id=report_id,
            token=token, decision="APPROVED", _external=True
        )
        reject_url = url_for(
            "decide_prediction", report_id=report_id,
            token=token, decision="REJECTED", _external=True
        )
        message = Message(
            subject=f"Action required: {disease.title()} prediction",
            recipients=[admin_email],
            body=(
                f"A new {disease} prediction requires review. \n\n"
                f"Patient inputs: {features}\n"
                f"Prediction: {prediction_result}\n"
                f"Report ID: {report_id}\n\n"
                f"APPROVE: {approve_url}\n\n"
                f"REJECT: {reject_url}\n\n"
                "These links will expire in 2 minutes and can only be used once."
            ),
        )
        mail.send(message)
        print(f"Action email sent for report {report_id}.")
    except Exception as e:
        print(f"Action email failed: {e}")
    # try: 
    #     message = Message(
    #         subject=f"Healthcare Prediction Report - {disease.title()}",
    #         recipients=[admin_email],
    #         body=(
    #             f"A new {disease} prediction was recorded. \n\n"
    #             f"Prediction result: {prediction_result}\n\n"
    #             "Log in to the healthcare application to review the report."
    #         ),
    #     )
    #     mail.send(message)
    #     print(f"Email sent for {disease} prediction")
    # except Exception as error:
    #     print("Email failed!")

@app.route("/decide/<int:report_id>/<decision>", methods=["GET", "POST"])
def decide_prediction(report_id, decision):
    if decision not in ("APPROVED", "REJECTED"):
        return "Invalid decision.", 400
    token =(
        request.args.get("token", "") 
        if request.method == "GET"
        else request.form.get("token", "")
    ) 
    if not token:
        return "Invalid action link.", 400
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = datetime.now(timezone.utc)

    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT status, expires_at, action_token_hash
            FROM prediction_reports
            WHERE id = ?
            """,
            (report_id,),
        ).fetchone()

        if row is None:
            return "Report not found.", 404

        if row[0] != "PENDING" or not row[2] or not secrets.compare_digest(row[2], token_hash):
            return "Invalid or already used action link", 410

        expires_at = datetime.fromisoformat(row[1])

        if now >= expires_at:
            return "This link has expired", 410

        if request.method == "GET":
            return f"""
                <h2>Confirm {decision.title()}?</h2>
                <p>Report #{report_id}</p>
                <p>This action cannot be undone.</p>
                <form method="POST">
                    <input type="hidden" name="token" value="{token}">
                    <button type="submit">Confirm {decision.title()}</button>
                </form>
            """

        posted_token = request.form.get("token", "")
        posted_hash = hashlib.sha256(posted_token.encode()).hexdigest()

        if not secrets.compare_digest(posted_hash, token_hash):
            return "Invalid action link", 400

        cursor = connection.execute(
            """
            UPDATE prediction_reports
            SET status = ?, decision_at = ?, action_token_hash = NULL
            WHERE id = ?
                AND status = 'PENDING'
                AND action_token_hash = ?
                AND expires_at > ?
            """,
            (
                decision, 
                now.isoformat(),
                report_id,
                token_hash,
                now.isoformat(),
            ),
        )
        connection.commit()
        if cursor.rowcount != 1:
            return "Link expired or used", 410
        return f"Report #{report_id} successfully {decision.lower()}."
    finally:
        connection.close()



@app.route('/plots/<path:filename>')
def plots(filename):
    return send_from_directory(os.path.join(app.root_path, 'plots'), filename)

model_diabetes = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-assesment-kabir-chauhan-Healthcare_Prediction-v2-muyix4yt/models/diabetes_model.pkl', 'rb'))
scalar_diabetes = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-assesment-kabir-chauhan-Healthcare_Prediction-v2-muyix4yt/models/diabetes_scaler.pkl', 'rb'))

model_heart = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-assesment-kabir-chauhan-Healthcare_Prediction-v2-muyix4yt/models/heart_model.pkl', 'rb'))
scalar_heart = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-assesment-kabir-chauhan-Healthcare_Prediction-v2-muyix4yt/models/heart_scaler.pkl', 'rb'))

model_parkinsons = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-assesment-kabir-chauhan-Healthcare_Prediction-v2-muyix4yt/models/parkinsons_model.pkl', 'rb'))
scalar_parkinsons = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-assesment-kabir-chauhan-Healthcare_Prediction-v2-muyix4yt/models/parkinsons_scaler.pkl', 'rb'))

# Routes 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes', methods = ['GET', 'POST'])
def diabetes():
    result = None
    result_class = ""

    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_input = scalar_diabetes.transform([features])
        prediction = model_diabetes.predict(final_input)[0]
        print("User input:", features)
        print("Model prediction:", prediction)
         
        if prediction == 1:
            result = "Diabetes Detected"
            result_class = "result-bad"
        else:
            result = "No Diabetes Detected"
            result_class = "result-good"
        report_id = save_prediction_report("diabetes", features, result)
        send_prediction_email(report_id, "diabetes", features, result)
    
    cm_plot = r'healthcare_project\plots\diabetes_cm.png'
    fi_plot = r'healthcare_project\plots\diabetes_features.png'
    
    return render_template('diabetes.html', result = result, 
    result_class = result_class, cm_plot = cm_plot, fi_plot = fi_plot)

@app.route('/heart', methods = ['GET', 'POST'])
def heart():
    result = None
    result_class = ""

    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_inputer = scalar_heart.transform([features])
        prediction = model_heart.predict(final_inputer)[0]
        print("User input:", features)
        print("Model prediction:", prediction)
        
        if prediction == 1:
            result = "No Heart disease Detected" 
            result_class = "result-good"
        else:
            result = "Heart Disease Detected"
            result_class = "result-bad"
        report_id= save_prediction_report("heart", features, result)
        send_prediction_email(report_id, "heart", features, result)

    cm_plot = r'healthcare_project\plots\heartdisease_cm.png'
    fi_plot = r'healthcare_project\plots\heartdisease_features.png'

    return render_template('heart.html', result = result, 
    result_class = result_class, cm_plot = cm_plot, fi_plot = fi_plot)

@app.route('/parkinson', methods = ['GET', 'POST'])
def parkinson():
    result = None
    result_class = ""

    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_input = scalar_parkinsons.transform([features])
        prediction = model_parkinsons.predict(final_input)[0]
        
        if prediction == 1:
            result = "Parkinson Detected"
            result_class = "result-bad"
        else:
            result = "No Parkinson Detected"
            result_class = "result-good"
    
    cm_plot = r'healthcare_project\plots\parkinsons_cm.png'
    fi_plot = r'healthcare_project\plots\parkinsons_feature.png'
    
    return render_template('parkinson.html', result = result, result_class = result_class, cm_plot = cm_plot, fi_plot = fi_plot)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
    

