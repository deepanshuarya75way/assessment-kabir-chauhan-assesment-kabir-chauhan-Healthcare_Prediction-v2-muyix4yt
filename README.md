# Healthcare Disease Prediction System

A Flask-based machine learning web application that predicts the likelihood of three common health conditions using trained scikit-learn models:

- **Diabetes**
- **Heart Disease**
- **Parkinson's Disease**

The project provides a simple web interface where users enter the required clinical/medical features and receive a model-based prediction.

> **Disclaimer:** This project is for educational and demonstration purposes only. Its predictions are not medical diagnoses and should not be used as a substitute for professional medical advice.

---

## Features

- 🩸 Diabetes prediction
- ❤️ Heart disease prediction
- 🧠 Parkinson's disease prediction
- Flask web application
- Separate trained ML model for each disease
- Feature scaling using saved `StandardScaler` objects
- HTML/Jinja2 templates for the frontend
- Static CSS styling
- Prediction result page
- Saved `.pkl` model and scaler files for inference
- Plot/image serving through Flask

---

## Tech Stack

### Backend
- Python 3.11
- Flask

### Machine Learning
- scikit-learn
- NumPy
- Pandas
- Pickle

### Frontend
- HTML
- CSS
- Jinja2 templates

### Model
- Random Forest classifiers
- StandardScaler preprocessing

---

## Project Structure

```text
healthcare_project/
│
├── app.py
├── model.py
├── dummy.py
├── requirements.txt
│
├── datasets/
│   └── ... dataset files
│
├── models/
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── parkinsons_model.pkl
│   └── parkinsons_scaler.pkl
│
├── plots/
│   └── ... generated plots
│
├── static/
│   └── style.css
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── diabetes.html
│   ├── heart.html
│   ├── parkinson.html
│   └── result.html
│
└── README.md
```

---

## How It Works

The application follows a simple machine-learning inference pipeline:

```text
User Input
    ↓
Flask Route
    ↓
Input Validation / Conversion
    ↓
Feature Scaling
    ↓
Trained ML Model
    ↓
Prediction
    ↓
Result Page
```

Each disease has its own trained model and scaler.

For example:

```text
Diabetes input
    ↓
diabetes_scaler.pkl
    ↓
diabetes_model.pkl
    ↓
Prediction
```

The same pattern is used for heart disease and Parkinson's disease.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd healthcare_project
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
```

Activate it using Git Bash:

```bash
source venv/Scripts/activate
```

Or using Command Prompt:

```cmd
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, the main dependencies are:

```bash
pip install flask numpy pandas scikit-learn
```

---

## Running the Application

Make sure your terminal is inside the project directory:

```text
healthcare_project/
```

Then run:

```bash
python app.py
```

Flask should start the development server.

Open the application in your browser:

```text
http://127.0.0.1:5000
```

---

## Machine Learning Models

The application uses separately saved model artifacts.

| Condition | Model | Preprocessor |
|---|---|---|
| Diabetes | `diabetes_model.pkl` | `diabetes_scaler.pkl` |
| Heart Disease | `heart_model.pkl` | `heart_scaler.pkl` |
| Parkinson's Disease | `parkinsons_model.pkl` | `parkinsons_scaler.pkl` |

The `.pkl` files contain the trained estimators and preprocessing objects required during inference.

### Important

The model files currently in this project were serialized with an earlier scikit-learn version. When loaded with a newer scikit-learn version, you may see an `InconsistentVersionWarning`.

For reproducible deployment, it is recommended to use the same scikit-learn version used during model training, or retrain and serialize the models using the target environment's version.

---

## Model Training

Training code is included in `model.py`.

The general workflow is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Random Forest Training
   ↓
Model Evaluation
   ↓
Save Model + Scaler
```

The resulting model and scaler files are stored in the `models/` directory and loaded by `app.py`.

---

## Application Pages

### Home Page

Provides access to the three prediction modules:

- Diabetes
- Heart Disease
- Parkinson's Disease

### Diabetes Prediction

Accepts the required diabetes-related clinical features and returns the model prediction.

### Heart Disease Prediction

Accepts the required cardiovascular/clinical features and returns the model prediction.

### Parkinson's Disease Prediction

Accepts voice-measurement features used by the Parkinson's disease model and returns the model prediction.

### Result Page

Displays the prediction returned by the selected machine-learning model.

---

## Running in Development

For development, Flask can also be started using:

```bash
flask --app app run --debug
```

This enables Flask's development reloader and debugging features.

**Do not use Flask's development server as a production deployment server.**

---

## Troubleshooting

### `FileNotFoundError` for a model

Make sure the expected files exist inside:

```text
models/
```

The filenames must match the names used in `app.py`.

For example:

```text
models/parkinsons_model.pkl
models/parkinsons_scaler.pkl
```

### scikit-learn `InconsistentVersionWarning`

This means the model was saved with a different scikit-learn version than the one currently running.

Check your installed version:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

For reproducibility, install the version used during training or retrain the models with the version used by the deployment environment.

### Flask cannot find a template

Make sure the HTML files are inside:

```text
templates/
```

and that the filenames used by `render_template()` exactly match the files on disk.

### Port already in use

Run Flask on another port:

```bash
flask --app app run --port 5001
```

Then open:

```text
http://127.0.0.1:5001
```

---

## Future Improvements

Possible improvements include:

- Add stronger input validation
- Improve model evaluation and documentation
- Add ROC-AUC and precision/recall metrics
- Add probability/confidence visualization
- Add model versioning
- Use a reproducible Python environment
- Add automated tests
- Improve accessibility and responsive UI
- Add a database for storing prediction history
- Deploy using a production WSGI server
- Containerize the application with Docker
- Add authentication if prediction history is stored
- Add monitoring and logging for deployed models

---

## Disclaimer

This application is an educational machine-learning project.

The predictions generated by the models are statistical outputs from trained machine-learning algorithms. They are **not medical diagnoses**, and the application should not be used to make medical decisions.

Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.

---

## Author

**Kabir**

Built as a machine-learning and Flask web development project.
