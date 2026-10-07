import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, recall_score, precision_score


if not os.path.exists(r"healthcare_project\models"):
    os.makedirs(r"healthcare_project\models")
if not os.path.exists(r"healthcare_project\plots"):
    os.makedirs(r"healthcare_project\plots")
    
# Diabetes model

print('Diabetes Model')
print('Training the model:')
df_diabetes = pd.read_csv(r'healthcare_project\datasets\diabetes.csv')
x_d = df_diabetes.drop('Outcome', axis=1)
y_d = df_diabetes['Outcome']

X_train, X_test, y_train, y_test = train_test_split(x_d, y_d, test_size = 0.2, random_state = 42)
scaler_diabetes = StandardScaler()
X_train = scaler_diabetes.fit_transform(X_train)
X_test = scaler_diabetes.transform(X_test)

model_diabetes = RandomForestClassifier(n_estimators = 200, random_state = 42)
model_diabetes.fit(X_train, y_train)

y_pred = model_diabetes.predict(X_test)
print("Metrics for this model: ")
print("Accuracy: ", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("Confusion matrix\n", confusion_matrix(y_test, y_pred))
print("Classification report\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt='d', cmap='Blues')
plt.title("Diabetes Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(r"healthcare_project\plots\diabetes_cm.png")
plt.clf()

importance = model_diabetes.feature_importances_
sns.barplot(x=importance, y=x_d.columns)
plt.title('Diabetes Feature Importance')
plt.savefig(r"healthcare_project\plots\diabetes_features.png")
plt.clf()

metrics_diabetes = {
    "accuracy": round(accuracy_score(y_test, y_pred)*100, 2),
    "precision": round(precision_score(y_test, y_pred)*100, 2),
    "recall": round(recall_score(y_test, y_pred)*100, 2),
    "f1_score": round(f1_score(y_test, y_pred)*100, 2)
}
print("Diabetes metrics dictionary:", metrics_diabetes)

pickle.dump(model_diabetes, open(r'healthcare_project\models\diabetes_model.pkl', 'wb'))
pickle.dump(scaler_diabetes, open(r'healthcare_project\models\diabetes_scaler.pkl','wb'))

print("\n")
# Heart disease Model

print("Heart disease model")
print("Training the model: ")

df_heart = pd.read_csv(r'healthcare_project\datasets\heart.csv')
x_h = df_heart.drop('target', axis = 1)
y_h = df_heart['target']

X_train, X_test, y_train, y_test = train_test_split(x_h, y_h, test_size = 0.2, random_state = 42)
scaler_heart = StandardScaler()
X_train = scaler_heart.fit_transform(X_train)
X_test = scaler_heart.transform(X_test)

model_heart = RandomForestClassifier(n_estimators = 200, random_state = 42)
model_heart.fit(X_train, y_train)

y_pred = model_heart.predict(X_test)
print("Metrics for this model")
print("Accuracy: ", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("Confusion matrix \n", confusion_matrix(y_test, y_pred))
print("Classification report\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = 'd', cmap = "Reds")
plt.title('Heart Disease Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(r'healthcare_project\plots\heartdisease_cm.png')
plt.clf()

importance = model_heart.feature_importances_
sns.barplot(x = importance, y = x_h.columns)
plt.title("Heart model Feature importance")
plt.savefig(r'healthcare_project\plots\heartdisease_features.png')

metrics_heart = {
    "accuracy": round(accuracy_score(y_test, y_pred)*100, 2),
    "precision": round(precision_score(y_test, y_pred)*100, 2),
    "recall": round(recall_score(y_test, y_pred)*100, 2),
    "f1_score": round(f1_score(y_test, y_pred)*100, 2)
}
print("Heart metrics dictionary:", metrics_heart)

pickle.dump(model_heart, open(r'healthcare_project\models\heart_model.pkl', 'wb'))
pickle.dump(scaler_heart, open(r'healthcare_project\models\heart_scaler.pkl','wb'))
print("\n")
# Parkinson's Model

print("Parkinson's model")
print("Training the model: ")

df_parkinson = pd.read_csv(r'healthcare_project\datasets\parkinsons.csv')
x_p = df_parkinson.drop(['name', 'status'], axis = 1)
y_p = df_parkinson['status']

X_train, X_test, y_train, y_test = train_test_split(x_p, y_p, test_size = 0.2, random_state = 42)
scaler_parkinson = StandardScaler() 
X_train = scaler_parkinson.fit_transform(X_train)
X_test = scaler_parkinson.transform(X_test)

model_parkinson = RandomForestClassifier(n_estimators = 200, random_state = 42)
model_parkinson.fit(X_train, y_train)

y_pred = model_parkinson.predict(X_test)
print("Metrics for Parkinson's model: ")
print("Accuracy: ", round(accuracy_score(y_test, y_pred)*100, 2))
print("Confusion matrix \n", confusion_matrix(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = 'd', cmap = "Greens")
plt.title("Parkinson's disease model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(r'healthcare_project\plots\parkinsons_cm.png')
plt.clf()

importance = model_parkinson.feature_importances_
sns.barplot(x = importance, y = x_p.columns)
plt.title("Parkinson's Feature importance")
plt.savefig(r'healthcare_project\plots\parkinsons_feature.png')
plt.clf()

metrics_parkinson = {
    "accuracy": round(accuracy_score(y_test, y_pred)*100, 2),
    "precision": round(precision_score(y_test, y_pred)*100, 2),
    "recall": round(recall_score(y_test, y_pred)*100, 2),
    "f1_score": round(f1_score(y_test, y_pred)*100, 2)
}
print("Parkinson metrics dictionary:", metrics_parkinson)


pickle.dump(model_parkinson, open(r'healthcare_project\models\parkinsons_model.pkl', 'wb'))
pickle.dump(scaler_parkinson, open(r'healthcare_project\models\parkinsons_scalar.pkl', 'wb'))
