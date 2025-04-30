import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------------------------------
# Diabetes Prediction Model
# -------------------------------

def train_diabetes_model():
    print("Training Diabetes Prediction Model...")
    
    # Load the data
    diabetes = pd.read_csv("./diabetes2/diabetes.csv")
    
    # Store feature names for later use
    feature_names = list(diabetes.drop('Outcome', axis=1).columns)
    
    # Replace zeros with NaN and then fill with mean values
    columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in columns_to_process:
        diabetes[column] = diabetes[column].replace(0, np.nan)
        diabetes[column].fillna(diabetes[column].mean(), inplace=True)
    
    # Split the data
    X = diabetes.drop('Outcome', axis=1)
    y = diabetes['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Diabetes Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model, scaler and feature names
    joblib.dump(rf_classifier, 'diabetes_rf_model.pkl')
    joblib.dump(scaler, 'diabetes_scaler.pkl')
    joblib.dump(feature_names, 'diabetes_feature_names.pkl')
    
    print("Diabetes model saved as 'diabetes_rf_model.pkl'")
    print("Diabetes scaler saved as 'diabetes_scaler.pkl'")
    
    return rf_classifier, scaler, feature_names

# -------------------------------
# Heart Disease Prediction Model
# -------------------------------

def train_heart_disease_model():
    print("\nTraining Heart Disease Prediction Model...")
    
    # Load the data
    heart_data = pd.read_csv("./heart2/heartDisease.csv")
    
    # Store feature names for later use
    feature_names = list(heart_data.drop('target', axis=1).columns)
    
    # Split the data
    X = heart_data.drop('target', axis=1)
    y = heart_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Heart Disease Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model, scaler and feature names
    joblib.dump(rf_classifier, 'heart_disease_rf_model.pkl')
    joblib.dump(scaler, 'heart_disease_scaler.pkl')
    joblib.dump(feature_names, 'heart_disease_feature_names.pkl')
    
    print("Heart disease model saved as 'heart_disease_rf_model.pkl'")
    print("Heart disease scaler saved as 'heart_disease_scaler.pkl'")
    
    return rf_classifier, scaler, feature_names

# -------------------------------
# Kidney Disease Prediction Model
# -------------------------------

def train_kidney_disease_model():
    print("\nTraining Kidney Disease Prediction Model...")
    
    # Load the data
    kidney_data = pd.read_csv("./kidney/Kidney_data.csv")
    
    # Remove unnecessary columns
    if 'id' in kidney_data.columns:
        kidney_data = kidney_data.drop('id', axis=1)
    
    # Clean string values: strip whitespace and handle special cases
    # Clean all object columns
    for column in kidney_data.select_dtypes(include=['object']).columns:
        kidney_data[column] = kidney_data[column].astype(str).str.strip()
        # Replace any variations of 'yes' with standard 'yes'
        kidney_data[column] = kidney_data[column].str.replace('\tyes', 'yes', regex=True)
        kidney_data[column] = kidney_data[column].str.replace(' yes', 'yes', regex=True)
        # Replace any variations of 'no' with standard 'no'
        kidney_data[column] = kidney_data[column].str.replace('\tno', 'no', regex=True)
        kidney_data[column] = kidney_data[column].str.replace(' no', 'no', regex=True)
    
    # Convert target variable to binary
    kidney_data['classification'] = kidney_data['classification'].map({'ckd': 1, 'notckd': 0})
    
    # Process categorical features
    # Replace categorical values with numeric values
    kidney_data['rbc'] = kidney_data['rbc'].replace({'normal': 0, 'abnormal': 1})
    kidney_data['pc'] = kidney_data['pc'].replace({'normal': 0, 'abnormal': 1})
    kidney_data['pcc'] = kidney_data['pcc'].replace({'notpresent': 0, 'present': 1})
    kidney_data['ba'] = kidney_data['ba'].replace({'notpresent': 0, 'present': 1})
    kidney_data['htn'] = kidney_data['htn'].replace({'no': 0, 'yes': 1})
    kidney_data['dm'] = kidney_data['dm'].replace({'no': 0, 'yes': 1})
    kidney_data['cad'] = kidney_data['cad'].replace({'no': 0, 'yes': 1})
    kidney_data['appet'] = kidney_data['appet'].replace({'good': 0, 'poor': 1})
    kidney_data['pe'] = kidney_data['pe'].replace({'no': 0, 'yes': 1})
    kidney_data['ane'] = kidney_data['ane'].replace({'no': 0, 'yes': 1})
    
    # Convert the object columns to numeric
    for col in kidney_data.select_dtypes(include=['object']).columns:
        kidney_data[col] = pd.to_numeric(kidney_data[col], errors='coerce')
    
    # Handle numeric features
    numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    
    # Fill missing values with mean for numeric columns
    for col in kidney_data.columns:
        if col != 'classification':
            kidney_data[col] = kidney_data[col].fillna(kidney_data[col].mean() if kidney_data[col].dtype != 'object' else kidney_data[col].mode()[0])
    
    # Store feature names for later use
    feature_names = list(kidney_data.drop('classification', axis=1).columns)
    
    # Make sure all data is numeric for training
    for col in kidney_data.columns:
        if kidney_data[col].dtype == 'object':
            print(f"Column {col} still has object type. Sample values: {kidney_data[col].unique()[:5]}")
    
    # Split the data
    X = kidney_data.drop('classification', axis=1)
    y = kidney_data['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Kidney Disease Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model, scaler and feature names
    joblib.dump(rf_classifier, 'kidney_disease_rf_model.pkl')
    joblib.dump(scaler, 'kidney_disease_scaler.pkl')
    joblib.dump(feature_names, 'kidney_disease_feature_names.pkl')
    
    print("Kidney disease model saved as 'kidney_disease_rf_model.pkl'")
    print("Kidney disease scaler saved as 'kidney_disease_scaler.pkl'")
    
    return rf_classifier, scaler, feature_names

# -------------------------------
# Model Prediction Functions
# -------------------------------

def predict_diabetes(input_data):
    # Load the model, scaler and feature names
    model = joblib.load('diabetes_rf_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    feature_names = joblib.load('diabetes_feature_names.pkl')
    
    # Prepare input data as DataFrame with feature names
    if isinstance(input_data, list):
        input_df = pd.DataFrame([input_data], columns=feature_names)
    else:
        input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    return prediction[0], probability[0]

def predict_heart_disease(input_data):
    # Load the model, scaler and feature names
    model = joblib.load('heart_disease_rf_model.pkl')
    scaler = joblib.load('heart_disease_scaler.pkl')
    feature_names = joblib.load('heart_disease_feature_names.pkl')
    
    # Prepare input data as DataFrame with feature names
    if isinstance(input_data, list):
        input_df = pd.DataFrame([input_data], columns=feature_names)
    else:
        input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    return prediction[0], probability[0]

def predict_kidney_disease(input_data):
    # Load the model, scaler and feature names
    model = joblib.load('kidney_disease_rf_model.pkl')
    scaler = joblib.load('kidney_disease_scaler.pkl')
    feature_names = joblib.load('kidney_disease_feature_names.pkl')
    
    # Prepare input data as DataFrame with feature names
    if isinstance(input_data, list):
        input_df = pd.DataFrame([input_data], columns=feature_names)
    else:
        input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    return prediction[0], probability[0]

# -------------------------------
# Main execution
# -------------------------------

if __name__ == "__main__":
    # Train and save all models
    diabetes_model, diabetes_scaler, diabetes_features = train_diabetes_model()
    heart_model, heart_scaler, heart_features = train_heart_disease_model()
    kidney_model, kidney_scaler, kidney_features = train_kidney_disease_model()
    
    print("\nAll models have been successfully trained and saved.")
    
    # Example of how to use the prediction functions
    print("\nExample Predictions:")
    
    # Example input for diabetes prediction (Pregnancies, Glucose, BloodPressure, SkinThickness, 
    # Insulin, BMI, DiabetesPedigreeFunction, Age)
    diabetes_sample = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    diabetes_result, diabetes_prob = predict_diabetes(diabetes_sample)
    print(f"Diabetes Prediction: {diabetes_result} (Probability of class 1: {diabetes_prob[1]:.4f})")
    
    # Example input for heart disease prediction (age, sex, cp, trestbps, chol, fbs, restecg, 
    # thalach, exang, oldpeak, slope, ca, thal)
    heart_sample = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
    heart_result, heart_prob = predict_heart_disease(heart_sample)
    print(f"Heart Disease Prediction: {heart_result} (Probability of class 1: {heart_prob[1]:.4f})")
    
    # Example input for kidney disease prediction (using the first sample from dataset as example)
    kidney_sample = [48.0, 80.0, 1.020, 1.0, 0.0, 0, 0, 0, 0, 121.0, 36.0, 1.2, 137.5, 4.6, 15.4, 44, 7800, 5.2, 1, 1, 0, 0, 0, 0]
    kidney_result, kidney_prob = predict_kidney_disease(kidney_sample)
    print(f"Kidney Disease Prediction: {kidney_result} (Probability of class 1: {kidney_prob[1]:.4f})")