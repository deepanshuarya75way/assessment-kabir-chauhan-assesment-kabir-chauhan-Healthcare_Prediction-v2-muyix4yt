from flask import Flask, request, render_template, send_from_directory
import pickle
import numpy as np
import os 

app = Flask(__name__)

@app.route('/plots/<path:filename>')
def plots(filename):
    return send_from_directory(os.path.join(app.root_path, 'plots'), filename)

model_diabetes = pickle.load(open(r'models\diabetes_model.pkl', 'rb'))
scalar_diabetes = pickle.load(open(r'models\diabetes_scaler.pkl', 'rb'))

model_heart = pickle.load(open(r'models\heart_model.pkl', 'rb'))
scalar_heart = pickle.load(open(r'models\heart_scaler.pkl', 'rb'))

model_parkinsons = pickle.load(open(r'models\parkinsons_model.pkl', 'rb'))
scalar_parkinsons = pickle.load(open(r'models\parkinsons_scaler.pkl', 'rb'))

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
    
    cm_plot = r'healthcare_project\plots\diabetes_cm.png'
    fi_plot = r'healthcare_project\plots\diabetes_features.png'
    
    return render_template('diabetes.html', result = result, result_class = result_class, cm_plot = cm_plot, fi_plot = fi_plot)

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
        
    cm_plot = r'healthcare_project\plots\heartdisease_cm.png'
    fi_plot = r'healthcare_project\plots\heartdisease_features.png'

    return render_template('heart.html', result = result, result_class = result_class, cm_plot = cm_plot, fi_plot = fi_plot)

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
    app.run(debug=True)
    

