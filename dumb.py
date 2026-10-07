import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open(r'healthcare_project\models\diabetes_model.pkl', 'rb'))
scaler = pickle.load(open(r'healthcare_project\models\diabetes_scaler.pkl', 'rb'))

# Create a dummy input (replace values according to your dataset)
sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # example from PIMA dataset

# Scale and predict
scaled = scaler.transform(sample)
pred = model.predict(scaled)[0]

print("Prediction:", "Diabetes Detected" if pred == 1 else "No Diabetes Detected")
