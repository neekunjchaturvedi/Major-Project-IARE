# Improved Heart Disease Classification using Random Forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create a directory to save the models if it doesn't exist
os.makedirs('heart', exist_ok=True)

# Load data
print("Loading the Heart Disease dataset...")
df = pd.read_csv("./heart.csv")

# Exploratory Data Analysis
print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nData Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())

# Understanding categorical features
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
print("\nCategorical features:")
for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())

# Data Preprocessing
print("\nPreprocessing the data...")

# Handle the 'thal' column with proper mapping
thal_mapping = {0: 'unknown', 1: 'normal', 2: 'fixed_defect', 3: 'reversible_defect'}
df['thal'] = df['thal'].map(lambda x: thal_mapping.get(x, 'unknown'))
    
# Check for outliers in numerical columns
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
print("\nChecking for outliers in numerical columns:")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    print(f"{col}: {outliers} outliers")

# Cap outliers instead of removing them
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Feature Engineering
print("\nPerforming feature engineering...")

# 1. Create age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], 
                        labels=['<40', '40-50', '50-60', '60-70', '>70'])

# 2. Create blood pressure categories
df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 160, 200], 
                          labels=['Normal', 'Elevated', 'Stage 1', 'Stage 2'])

# 3. Create cholesterol categories
df['chol_category'] = pd.cut(df['chol'], bins=[0, 200, 240, 600], 
                           labels=['Normal', 'Borderline', 'High'])

# 4. Heart Rate Reserve (Maximum heart rate - resting heart rate)
# Since resting heart rate is not directly available, we'll use a formula
# Estimated resting heart rate = 220 - age
df['est_rest_hr'] = 220 - df['age']
df['heart_rate_reserve'] = df['est_rest_hr'] - df['thalach']

# 5. Create heart rate efficiency ratio
df['heart_efficiency'] = df['thalach'] / df['trestbps']

# 6. Create composite risk score
df['risk_score'] = (df['age'] / 10) + (df['trestbps'] / 100) + (df['chol'] / 200) + df['oldpeak']

# 7. Calculate cardiovascular risk score incorporating multiple factors
df['cardio_risk_score'] = (
    df['age']/40 + 
    df['chol']/200 + 
    df['trestbps']/120 + 
    (1 if df['sex'].all()==1 else 0.8) + 
    (1 if df['fbs'].all()==1 else 0) + 
    df['oldpeak']
)

# 8. Add a chest pain severity index
cp_weights = {0: 0, 1: 1, 2: 2, 3: 3}
df['cp_severity'] = df['cp'].map(cp_weights)

# 9. Create an oxygen supply indicator
df['oxygen_supply_index'] = df['thalach'] / (df['trestbps'] + 1)

# 10. Create interaction features
df['age_thal_interaction'] = df['age'] * df['cp']
df['chol_thal_interaction'] = df['chol'] * df['thalach']

print("\nFeatures after engineering:")
print(df.columns.tolist())

# Save feature engineering info for deployment
feature_engineering_info = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'age_group_bins': [0, 40, 50, 60, 70, 100],
    'bp_category_bins': [0, 120, 140, 160, 200],
    'chol_category_bins': [0, 200, 240, 600],
    'thal_mapping': thal_mapping,
    'cp_weights': cp_weights
}
joblib.dump(feature_engineering_info, 'heart/heart_feature_engineering_info.joblib')
print("Feature engineering info saved for deployment")

# Identify categorical and numerical columns for preprocessing
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'age_group', 'bp_category', 'chol_category']
numerical_cols_extended = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'heart_rate_reserve', 
                         'heart_efficiency', 'risk_score', 'cardio_risk_score', 'cp_severity',
                         'oxygen_supply_index', 'age_thal_interaction', 'chol_thal_interaction']

# Split data into features and target
X = df.drop(['target'], axis=1)
y = df['target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create preprocessing pipelines for both numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For categorical columns, use OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols_extended),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Explicitly fit the preprocessor on the training data
print("Fitting preprocessor on training data...")
preprocessor.fit(X_train)

# Save the fitted preprocessor for deployment
joblib.dump(preprocessor, 'heart/heart_preprocessor.joblib')
print("Fitted preprocessor saved for deployment")

# Apply preprocessing
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save a copy of the preprocessor in the root directory for easier testing
joblib.dump(preprocessor, 'heart_preprocessor.joblib')
print("Fitted preprocessor also saved to root directory for testing")

# Apply SMOTE to balance classes
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Model Training
print("\nTraining the Random Forest model...")

# Parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=5, 
    n_jobs=-1, scoring='accuracy', verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")

# Create stacking ensemble
print("\nCreating stacking ensemble model...")
base_estimators = [
    ('rf', RandomForestClassifier(random_state=42, **best_params)),
    ('svc', SVC(probability=True, random_state=42, gamma='auto'))
]

stack_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

stack_model.fit(X_train_resampled, y_train_resampled)

# Model Evaluation
print("\nEvaluating the models...")

# Random Forest predictions
best_rf = RandomForestClassifier(random_state=42, **best_params)
best_rf.fit(X_train_resampled, y_train_resampled)
rf_y_pred = best_rf.predict(X_test_processed)
rf_accuracy = accuracy_score(y_test, rf_y_pred)

# Stacking model predictions
stack_y_pred = stack_model.predict(X_test_processed)
stack_accuracy = accuracy_score(y_test, stack_y_pred)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print(f"Stacking Ensemble Accuracy: {stack_accuracy:.4f}")

# Use the better model for final evaluation
if stack_accuracy > rf_accuracy:
    best_model = stack_model
    y_pred = stack_y_pred
    print("\nUsing Stacking Ensemble for final evaluation")
    model_name = "heart_stacking_model.joblib"
else:
    best_model = best_rf
    y_pred = rf_y_pred
    print("\nUsing Random Forest for final evaluation")
    model_name = "heart_model.joblib"

# Save the best model for deployment
joblib.dump(best_model, f'heart/{model_name}')
print(f"\nBest model saved as 'heart/{model_name}' for deployment")

# Also save to root directory for easier testing
joblib.dump(best_model, model_name)
print(f"Model also saved to root directory as '{model_name}' for testing")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('heart/heart_confusion_matrix.png')

# ROC Curve
y_prob = best_model.predict_proba(X_test_processed)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('heart/heart_roc_curve.png')

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('heart/heart_precision_recall_curve.png')

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Try to extract feature importance from Random Forest model
if best_model == best_rf:
    try:
        importances = best_rf.feature_importances_
        
        # Create feature names based on preprocessor
        feature_names = []
        # Add numerical feature names directly
        feature_names.extend(numerical_cols_extended)
        
        # Add transformed categorical feature names
        cat_feature_count = X_train_processed.shape[1] - len(numerical_cols_extended)
        for i in range(cat_feature_count):
            feature_names.append(f"categorical_{i}")
        
        # Select only the first len(importances) feature names if needed
        feature_names = feature_names[:len(importances)]
        
        # Create a dataframe for feature importance
        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importances.head(10))
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('heart/heart_feature_importance.png')
        
        # Save feature importances for interpretation
        joblib.dump(feature_importances, 'heart/heart_feature_importances.joblib')
    except Exception as e:
        print(f"Could not extract feature importances: {str(e)}")
else:
    print("\nFeature importance not available for stacking ensemble")

# Save additional model metadata for future reference
model_metadata = {
    'accuracy': rf_accuracy if best_model == best_rf else stack_accuracy,
    'cross_val_score_mean': cv_scores.mean(),
    'cross_val_score_std': cv_scores.std(),
    'best_params': best_params,
    'auc': roc_auc,
    'model_type': 'Random Forest' if best_model == best_rf else 'Stacking Ensemble',
    'preprocessor_path': 'heart/heart_preprocessor.joblib',
    'model_path': f'heart/{model_name}'
}
joblib.dump(model_metadata, 'heart/heart_model_metadata.joblib')
print("\nModel metadata saved for reference")

# Create a simple prediction function to demonstrate model usage
def predict_heart_disease(input_data):
    """
    Function to demonstrate model deployment usage.
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing patient features
    
    Returns:
    --------
    prediction : int
        0 for no heart disease, 1 for heart disease
    probability : float
        Probability of heart disease
    """
    # Create a pandas DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Load saved preprocessor and model with explicit error handling
    try:
        print("Loading preprocessor...")
        # Try both paths - root directory and heart subdirectory
        if os.path.exists('heart_preprocessor.joblib'):
            preprocessor = joblib.load('heart_preprocessor.joblib')
            print("Preprocessor loaded from root directory")
        elif os.path.exists('heart/heart_preprocessor.joblib'):
            preprocessor = joblib.load('heart/heart_preprocessor.joblib')
            print("Preprocessor loaded from heart directory")
        else:
            raise FileNotFoundError("Preprocessor file not found")
            
        print("Loading model...")
        # Try both paths for the model file as well
        if os.path.exists(model_name):
            model = joblib.load(model_name)
            print("Model loaded from root directory")
        elif os.path.exists(f'heart/{model_name}'):
            model = joblib.load(f'heart/{model_name}')
            print("Model loaded from heart directory")
        else:
            raise FileNotFoundError("Model file not found")
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        raise
    
    print("Adding engineered features...")
    # Add engineered features before preprocessing
    # Heart Rate Reserve
    input_df['est_rest_hr'] = 220 - input_df['age']
    input_df['heart_rate_reserve'] = input_df['est_rest_hr'] - input_df['thalach']
    
    # Heart efficiency
    input_df['heart_efficiency'] = input_df['thalach'] / input_df['trestbps']
    
    # Risk score
    input_df['risk_score'] = (input_df['age'] / 10) + (input_df['trestbps'] / 100) + (input_df['chol'] / 200) + input_df['oldpeak']
    
    # Cardiovascular risk score
    input_df['cardio_risk_score'] = (input_df['age']/40 + input_df['chol']/200 + input_df['trestbps']/120 + 
                                    input_df['sex'] + input_df['fbs'] + input_df['oldpeak'])
    
    # Chest pain severity
    input_df['cp_severity'] = input_df['cp'].map(cp_weights)
    
    # Oxygen supply index
    input_df['oxygen_supply_index'] = input_df['thalach'] / (input_df['trestbps'] + 1)
    
    # Interaction features
    input_df['age_thal_interaction'] = input_df['age'] * input_df['cp']
    input_df['chol_thal_interaction'] = input_df['chol'] * input_df['thalach']
    
    # Create categorical features
    input_df['age_group'] = pd.cut(input_df['age'], 
                                bins=[0, 40, 50, 60, 70, 100], 
                                labels=['<40', '40-50', '50-60', '60-70', '>70'])
    
    input_df['bp_category'] = pd.cut(input_df['trestbps'], 
                                 bins=[0, 120, 140, 160, 200], 
                                 labels=['Normal', 'Elevated', 'Stage 1', 'Stage 2'])
    
    input_df['chol_category'] = pd.cut(input_df['chol'], 
                                   bins=[0, 200, 240, 600], 
                                   labels=['Normal', 'Borderline', 'High'])
    
    print("Transforming input data with preprocessor...")
    # Apply the preprocessor pipeline
    input_processed = preprocessor.transform(input_df)
    
    print("Making prediction...")
    # Make prediction
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0][1]
    
    print(f"Prediction: {prediction} ({probability:.2%} probability of heart disease)")
    return prediction, probability

# Test prediction function with an example
print("\nTesting model deployment with an example case...")
test_patient = {
    'age': 55,
    'sex': 1,  # Male
    'cp': 1,   # Atypical angina
    'trestbps': 140,
    'chol': 230,
    'fbs': 0,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.5,
    'slope': 1,
    'ca': 0,
    'thal': 'normal'
}

try:
    prediction, probability = predict_heart_disease(test_patient)
    print(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
    print(f"Probability of Heart Disease: {probability:.2%}")
    print("\nModel testing successful!")
except Exception as e:
    print(f"\nError during testing: {str(e)}")
    print("\nDiagnostic information:")
    
    # Check if model files exist
    print(f"heart_preprocessor.joblib exists: {os.path.exists('heart_preprocessor.joblib')}")
    print(f"heart/heart_preprocessor.joblib exists: {os.path.exists('heart/heart_preprocessor.joblib')}")
    print(f"{model_name} exists: {os.path.exists(model_name)}")
    print(f"heart/{model_name} exists: {os.path.exists(f'heart/{model_name}')}")
    
    # If preprocessor exists but is causing issues, provide detailed info
    if os.path.exists('heart_preprocessor.joblib'):
        try:
            test_preprocessor = joblib.load('heart_preprocessor.joblib')
            print("Preprocessor loaded successfully for inspection")
            print(f"Preprocessor type: {type(test_preprocessor)}")
            print(f"Transformers: {test_preprocessor.transformers}")
        except Exception as e2:
            print(f"Failed to inspect preprocessor: {str(e2)}")

print("\nImproved model training and evaluation completed. Model deployed for future use.")