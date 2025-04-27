# Improved Diabetes Classification using Random Forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create a directory to save the models if it doesn't exist
os.makedirs('diabetes', exist_ok=True)

# Load data
print("Loading the Diabetes dataset...")
df = pd.read_csv("./diabetes.csv")

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

# Check for 0 values in columns where 0 doesn't make sense medically
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nCounting zero values in medically important features:")
for column in zero_columns:
    zero_count = (df[column] == 0).sum()
    zero_percentage = (zero_count / len(df)) * 100
    print(f"{column}: {zero_count} zeros ({zero_percentage:.2f}%)")

# Data Preprocessing
print("\nPreprocessing the data...")

# Check for outliers in numerical columns
numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
print("\nChecking for outliers in numerical columns:")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    print(f"{col}: {outliers} outliers")

# Replace 0s with NaN for columns where 0 is not a valid value
for column in zero_columns:
    df[column] = df[column].replace(0, np.nan)

# Check missing values after replacing zeros
print("\nMissing values after replacing zeros:")
print(df.isnull().sum())

# Use KNN imputation for better handling of the missing values
print("\nApplying KNN imputation...")
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Save imputer for deployment
joblib.dump(imputer, 'diabetes/diabetes_imputer.joblib')
# Also save to root directory for testing
joblib.dump(imputer, 'diabetes_imputer.joblib')
print("KNN imputer saved for deployment")

# Cap outliers instead of removing them
for col in numerical_cols:
    Q1 = df_imputed[col].quantile(0.25)
    Q3 = df_imputed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_imputed[col] = df_imputed[col].clip(lower=lower_bound, upper=upper_bound)

print("\nData after imputation and outlier handling:")
print(df_imputed.describe())

# Feature Engineering
print("\nPerforming feature engineering...")

# 1. Create BMI categories
df_imputed['BMI_Category'] = pd.cut(df_imputed['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], 
                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# 2. Create age groups
df_imputed['Age_Group'] = pd.cut(df_imputed['Age'], bins=[20, 30, 40, 50, 60, 100], 
                       labels=['20-30', '30-40', '40-50', '50-60', '60+'])

# 3. Create glucose level categories
df_imputed['Glucose_Category'] = pd.cut(df_imputed['Glucose'], bins=[0, 70, 99, 126, 300], 
                               labels=['Low', 'Normal', 'Prediabetes', 'Diabetes'])

# 4. Create insulin sensitivity feature (adjusted to avoid division by zero)
df_imputed['Insulin_Glucose_Ratio'] = df_imputed['Insulin'] / (df_imputed['Glucose'] + 1)

# 5. Create BMI*Age interaction feature
df_imputed['BMI_Age_Interaction'] = df_imputed['BMI'] * df_imputed['Age']

# 6. Create a diabetes risk score
df_imputed['Diabetes_Risk_Score'] = (
    (df_imputed['Glucose']/100) + 
    (df_imputed['BMI']/25) + 
    (df_imputed['Age']/50) + 
    (df_imputed['DiabetesPedigreeFunction']*2)
)

# 7. Create blood pressure categories
df_imputed['BP_Category'] = pd.cut(df_imputed['BloodPressure'], 
                          bins=[0, 60, 80, 120, 140, 200], 
                          labels=['Very Low', 'Low', 'Normal', 'High', 'Very High'])

# 8. Create a pregnancy risk factor
df_imputed['Pregnancy_Risk'] = pd.cut(df_imputed['Pregnancies'], 
                            bins=[-1, 0, 2, 4, 12, 20], 
                            labels=['None', 'Few', 'Moderate', 'High', 'Very High'])

# 9. Create insulin categories
df_imputed['Insulin_Category'] = pd.cut(df_imputed['Insulin'], 
                              bins=[0, 16, 166, 500, 1000], 
                              labels=['Low', 'Normal', 'High', 'Very High'])

# 10. Create metabolic syndrome indicator - useful for diabetes risk
conditions = (
    (df_imputed['BMI'] > 30) | 
    (df_imputed['BloodPressure'] > 130) | 
    (df_imputed['Glucose'] > 110)
)
df_imputed['Metabolic_Syndrome_Risk'] = conditions.astype(int)

# Save feature engineering info for deployment
feature_engineering_info = {
    'zero_columns': zero_columns,
    'numerical_cols': numerical_cols,
    'bmi_bins': [0, 18.5, 24.9, 29.9, 100],
    'age_bins': [20, 30, 40, 50, 60, 100],
    'glucose_bins': [0, 70, 99, 126, 300],
    'bp_bins': [0, 60, 80, 120, 140, 200],
    'pregnancies_bins': [-1, 0, 2, 4, 12, 20],
    'insulin_bins': [0, 16, 166, 500, 1000]
}
joblib.dump(feature_engineering_info, 'diabetes/diabetes_feature_engineering_info.joblib')
# Also save to root directory for testing
joblib.dump(feature_engineering_info, 'diabetes_feature_engineering_info.joblib')
print("Feature engineering info saved for deployment")

print("\nFeatures after engineering:")
print(df_imputed.columns.tolist())

# One-hot encode categorical features
categorical_features = ['BMI_Category', 'Age_Group', 'Glucose_Category', 
                        'BP_Category', 'Pregnancy_Risk', 'Insulin_Category']
df_encoded = pd.get_dummies(df_imputed, columns=categorical_features, drop_first=True)

# Save the list of columns for deployment
joblib.dump(df_encoded.columns.tolist(), 'diabetes/diabetes_features.joblib')
# Also save to root directory for testing
joblib.dump(df_encoded.columns.tolist(), 'diabetes_features.joblib')
print("Feature list saved for deployment")

# Split data into features and target
X = df_encoded.drop(['Outcome'], axis=1)
y = df_encoded['Outcome']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling with RobustScaler (better for data with outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for deployment
joblib.dump(scaler, 'diabetes/diabetes_scaler.joblib')
# Also save to root directory for testing
joblib.dump(scaler, 'diabetes_scaler.joblib')
print("Scaler saved for deployment")

# Apply SMOTE to handle class imbalance
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

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
rf_y_pred = best_rf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_y_pred)

# Stacking model predictions
stack_y_pred = stack_model.predict(X_test_scaled)
stack_accuracy = accuracy_score(y_test, stack_y_pred)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print(f"Stacking Ensemble Accuracy: {stack_accuracy:.4f}")

# Use the better model for final evaluation
if stack_accuracy > rf_accuracy:
    best_model = stack_model
    y_pred = stack_y_pred
    print("\nUsing Stacking Ensemble for final evaluation")
    model_name = "diabetes_stacking_model.joblib"
else:
    best_model = best_rf
    y_pred = rf_y_pred
    print("\nUsing Random Forest for final evaluation")
    model_name = "diabetes_model.joblib"

# Save the best model for deployment
joblib.dump(best_model, f'diabetes/{model_name}')
# Also save to root directory for testing
joblib.dump(best_model, model_name)
print(f"\nBest model saved as 'diabetes/{model_name}' for deployment")

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
plt.savefig('diabetes/diabetes_confusion_matrix.png')

# ROC Curve
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
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
plt.savefig('diabetes/diabetes_roc_curve.png')

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('diabetes/diabetes_precision_recall_curve.png')

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Feature importance
if best_model == best_rf:
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importances.head(10))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('diabetes/diabetes_feature_importance.png')
    
    # Save feature importances for interpretation
    joblib.dump(feature_importances, 'diabetes/diabetes_feature_importances.joblib')
else:
    print("\nFeature importance not available for stacking ensemble")

# Save additional model metadata for future reference
model_metadata = {
    'accuracy': rf_accuracy if best_model == best_rf else stack_accuracy,
    'cross_val_score_mean': cv_scores.mean(),
    'cross_val_score_std': cv_scores.std(),
    'best_params': best_params,
    'auc': roc_auc,
    'model_type': 'Random Forest' if best_model == best_rf else 'Stacking Ensemble'
}
joblib.dump(model_metadata, 'diabetes/diabetes_model_metadata.joblib')
print("\nModel metadata saved for reference")

# Create a simple prediction function to demonstrate model usage
def predict_diabetes(input_data):
    """
    Function to demonstrate model deployment usage.
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing patient features
    
    Returns:
    --------
    prediction : int
        0 for no diabetes, 1 for diabetes
    probability : float
        Probability of diabetes
    """
    print("Loading model components...")
    # First try to load from the root directory
    try:
        # Load saved model components with proper error handling
        try:
            print("Attempting to load from current directory...")
            imputer = joblib.load('diabetes_imputer.joblib')
            feature_eng_info = joblib.load('diabetes_feature_engineering_info.joblib')
            scaler = joblib.load('diabetes_scaler.joblib')
            model = joblib.load(model_name)
            features = joblib.load('diabetes_features.joblib')
            print("All components loaded from current directory successfully.")
        except FileNotFoundError:
            # If not found in root, try the diabetes subdirectory
            print("Files not found in current directory. Trying 'diabetes/' subdirectory...")
            imputer = joblib.load('diabetes/diabetes_imputer.joblib')
            feature_eng_info = joblib.load('diabetes/diabetes_feature_engineering_info.joblib')
            scaler = joblib.load('diabetes/diabetes_scaler.joblib')
            model = joblib.load(f'diabetes/{model_name}')
            features = joblib.load('diabetes/diabetes_features.joblib')
            print("All components loaded from 'diabetes/' subdirectory successfully.")
    except Exception as e:
        print(f"Error loading model files: {str(e)}")
        print("\nDiagnostic information:")
        print(f"Current working directory: {os.getcwd()}")
        print(f"diabetes_imputer.joblib exists: {os.path.exists('diabetes_imputer.joblib')}")
        print(f"diabetes/diabetes_imputer.joblib exists: {os.path.exists('diabetes/diabetes_imputer.joblib')}")
        raise
        
    print("Processing input data...")
    # Create a pandas DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Create a copy of the input DataFrame with the Outcome column for imputation
    # (since the imputer was fit with this column)
    imputer_df = input_df.copy()
    imputer_df['Outcome'] = 0  # Add a placeholder value
    
    # Replace zeros with NaN for certain columns
    zero_columns = feature_eng_info['zero_columns']
    for column in zero_columns:
        if column in imputer_df:
            imputer_df[column] = imputer_df[column].replace(0, np.nan)
    
    # Apply imputation
    print("Applying imputation...")
    imputer_df = pd.DataFrame(imputer.transform(imputer_df), columns=imputer_df.columns)
    
    # Copy back the imputed values to the original input DataFrame
    for column in zero_columns:
        if column in input_df:
            input_df[column] = imputer_df[column]
    
    # Apply feature engineering
    print("Adding engineered features...")
    # BMI categories
    input_df['BMI_Category'] = pd.cut(
        input_df['BMI'], 
        bins=feature_eng_info['bmi_bins'], 
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    
    # Age groups
    input_df['Age_Group'] = pd.cut(
        input_df['Age'], 
        bins=feature_eng_info['age_bins'], 
        labels=['20-30', '30-40', '40-50', '50-60', '60+']
    )
    
    # Glucose categories
    input_df['Glucose_Category'] = pd.cut(
        input_df['Glucose'], 
        bins=feature_eng_info['glucose_bins'], 
        labels=['Low', 'Normal', 'Prediabetes', 'Diabetes']
    )
    
    # Blood pressure categories
    input_df['BP_Category'] = pd.cut(
        input_df['BloodPressure'], 
        bins=feature_eng_info['bp_bins'],
        labels=['Very Low', 'Low', 'Normal', 'High', 'Very High']
    )
    
    # Pregnancy risk
    input_df['Pregnancy_Risk'] = pd.cut(
        input_df['Pregnancies'], 
        bins=feature_eng_info['pregnancies_bins'],
        labels=['None', 'Few', 'Moderate', 'High', 'Very High']
    )
    
    # Insulin categories
    input_df['Insulin_Category'] = pd.cut(
        input_df['Insulin'], 
        bins=feature_eng_info['insulin_bins'],
        labels=['Low', 'Normal', 'High', 'Very High']
    )
    
    # Create derived features
    input_df['Insulin_Glucose_Ratio'] = input_df['Insulin'] / (input_df['Glucose'] + 1)
    input_df['BMI_Age_Interaction'] = input_df['BMI'] * input_df['Age']
    input_df['Diabetes_Risk_Score'] = (
        (input_df['Glucose']/100) + 
        (input_df['BMI']/25) + 
        (input_df['Age']/50) + 
        (input_df['DiabetesPedigreeFunction']*2)
    )
    conditions = (
        (input_df['BMI'] > 30) | 
        (input_df['BloodPressure'] > 130) | 
        (input_df['Glucose'] > 110)
    )
    input_df['Metabolic_Syndrome_Risk'] = conditions.astype(int)
    
    # One-hot encode categorical features
    print("One-hot encoding categorical features...")
    categorical_features = ['BMI_Category', 'Age_Group', 'Glucose_Category', 
                           'BP_Category', 'Pregnancy_Risk', 'Insulin_Category']
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
    
    # Ensure all columns from training are present
    print("Aligning features with training data...")
    model_features = [col for col in features if col != 'Outcome']
    for col in model_features:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    
    # Keep only the columns that were used for training (excluding the target)
    input_df_encoded = input_df_encoded[model_features]
    
    # Apply scaling
    print("Applying scaling...")
    input_scaled = scaler.transform(input_df_encoded)
    
    # Make prediction
    print("Making prediction...")
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    print(f"Prediction: {prediction} (Probability: {probability:.2%})")
    return prediction, probability

# Test prediction function with an example
print("\nTesting model deployment with an example case...")
test_patient = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}

try:
    prediction, probability = predict_diabetes(test_patient)
    print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
    print(f"Probability of Diabetes: {probability:.2%}")
    print("\nModel testing successful!")
except Exception as e:
    print(f"\nError during testing: {str(e)}")
    print("\nDiagnostic information:")
    
    # Check if model files exist
    print(f"diabetes_imputer.joblib exists: {os.path.exists('diabetes_imputer.joblib')}")
    print(f"diabetes/diabetes_imputer.joblib exists: {os.path.exists('diabetes/diabetes_imputer.joblib')}")
    print(f"diabetes_feature_engineering_info.joblib exists: {os.path.exists('diabetes_feature_engineering_info.joblib')}")
    print(f"diabetes/diabetes_feature_engineering_info.joblib exists: {os.path.exists('diabetes/diabetes_feature_engineering_info.joblib')}")
    print(f"{model_name} exists: {os.path.exists(model_name)}")
    print(f"diabetes/{model_name} exists: {os.path.exists(f'diabetes/{model_name}')}")

print("\nImproved model training and evaluation completed. Model deployed for future use.")