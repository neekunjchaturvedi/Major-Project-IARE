# Improved Kidney Disease Classification using Random Forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
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
os.makedirs('kidney', exist_ok=True)

# Load data
print("Loading the Kidney Disease dataset...")
df = pd.read_csv("./kidney_disease.csv")

# Exploratory Data Analysis
print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nData Information:")
print(df.info())

print("\nStatistical Summary of Numerical Columns:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())

# Fix column names by removing space
df.columns = df.columns.str.strip()

# Data Preprocessing
print("\nPreprocessing the data...")

# 1. Handling categorical variables
# Identify categorical columns 
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Fix inconsistencies in categorical values
print("\nCleaning categorical variables...")
for col in categorical_cols:
    df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]

# Replace abnormal variations
if df['classification'].dtype == 'object':
    df['classification'] = df['classification'].replace('ckd\t', 'ckd')
    
# Convert target to binary
print("\nConverting target to binary...")
df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0})

# 2. Handling missing values
print("\nHandling missing values...")

# Replace ? or empty strings with NaN
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].replace(['?', ' ', ''], np.nan)

# Check missing values after replacing placeholders
print("\nMissing values after replacing placeholders:")
print(df.isnull().sum())

# 3. Convert numerical columns to the right type
numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for outliers in numerical columns
print("\nChecking for outliers in numerical columns:")
for col in numerical_cols:
    if df[col].isnull().sum() < len(df):  # Only check columns with some non-null values
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        print(f"{col}: {outliers} outliers")

# 4. Handle categorical features - convert to numerical using LabelEncoder
print("\nEncoding categorical features...")
label_encoders = {}
for col in categorical_cols:
    if df[col].dtype == 'object':  # Only encode object columns
        label_encoders[col] = LabelEncoder()
        # Fill missing values with a placeholder before encoding
        df[col].fillna('Missing', inplace=True)
        df[col] = label_encoders[col].fit_transform(df[col])

# Save label encoders for deployment
joblib.dump(label_encoders, 'kidney/kidney_label_encoders.joblib')
# Also save to root directory for testing
joblib.dump(label_encoders, 'kidney_label_encoders.joblib')
print("Label encoders saved for deployment")

# 5. Use KNN imputation for better handling of missing values in numerical columns
print("\nApplying KNN imputation for numerical values...")
numerical_imputer = KNNImputer(n_neighbors=3)
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

# Save numerical imputer for deployment
joblib.dump(numerical_imputer, 'kidney/kidney_numerical_imputer.joblib')
# Also save to root directory for testing
joblib.dump(numerical_imputer, 'kidney_numerical_imputer.joblib')
print("Numerical imputer saved for deployment")

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

# 1. Create renal function indicator
df['renal_function'] = df['bu'] / df['sc']

# 2. Create anemia indicator based on hemoglobin levels
df['anemia_risk'] = np.where(df['hemo'] < 12, 1, 0)

# 3. Create blood pressure categories
df['bp_category'] = pd.cut(df['bp'], bins=[0, 80, 120, 140, 180, 300], 
                         labels=[0, 1, 2, 3, 4])

# 4. Create a feature for kidney function based on creatinine and blood urea
df['kidney_function_score'] = df['sc'] + (df['bu'] / 10)

# 5. Create estimated Glomerular Filtration Rate (eGFR)
# Using simplified MDRD formula: eGFR = 186 × (Creatinine)^-1.154 × (Age)^-0.203 × 0.742(if female)
# Since we don't have gender, we'll use a simplified version
df['eGFR'] = 186 * (df['sc']**-1.154) * (df['age']**-0.203)
df['eGFR'] = df['eGFR'].replace([np.inf, -np.inf], np.nan)  # Handle infinite values
df['eGFR'].fillna(df['eGFR'].median(), inplace=True)  # Fill nulls with median

# 6. Create BUN/Creatinine ratio - important for distinguishing prerenal from intrinsic renal disease
df['BUN_Creatinine_ratio'] = df['bu'] / df['sc']
df['BUN_Creatinine_ratio'] = df['BUN_Creatinine_ratio'].replace([np.inf, -np.inf], np.nan)
df['BUN_Creatinine_ratio'].fillna(df['BUN_Creatinine_ratio'].median(), inplace=True)

# 7. Create age categories
df['age_category'] = pd.cut(df['age'], bins=[0, 30, 45, 65, 100], 
                          labels=['Young', 'Middle', 'Senior', 'Elderly'])

# 8. Create combined risk score based on multiple parameters
df['combined_risk_score'] = (
    df['sc'] * 2 + 
    df['bu'] / 10 + 
    (20 - df['hemo']) * 0.5 + 
    df['al'] * 0.5 + 
    (df['bp'] - 90) / 10
)

# 9. Create hypertension and diabetes interaction
df['htn_dm_interaction'] = df['htn'] * df['dm']

# 10. Create anemia and chronic kidney disease correlation
df['anemia_ckd_corr'] = df['anemia_risk'] * (df['sc'] > 1.2).astype(int)

# 11. Create potential proteinuria indicator (based on albumin)
df['proteinuria_risk'] = np.where(df['al'] > 1, 1, 0)

# 12. Create kidney stone risk based on calcium and other factors
df['kidney_stone_risk'] = np.where(
    (df['sg'] > 1.02) | (df['rc'] < 4.0),
    1, 0
)

# One-hot encode categorical features - FIXED VERSION
age_cat_dummies = pd.get_dummies(df['age_category'], drop_first=True, prefix='age_cat')
df = pd.concat([df, age_cat_dummies], axis=1)

# Save feature engineering info for deployment
feature_engineering_info = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'bp_bins': [0, 80, 120, 140, 180, 300],
    'age_bins': [0, 30, 45, 65, 100]
}
joblib.dump(feature_engineering_info, 'kidney/kidney_feature_engineering_info.joblib')
# Also save to root directory for testing
joblib.dump(feature_engineering_info, 'kidney_feature_engineering_info.joblib')
print("Feature engineering info saved for deployment")

print("\nFeatures after engineering:")
print(df.columns.tolist())

# Handle any remaining missing values in numerical columns
numerical_features = numerical_cols + ['renal_function', 'kidney_function_score', 'eGFR', 
                                      'BUN_Creatinine_ratio', 'combined_risk_score']
for col in numerical_features:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Save the column dtypes as strings for safer handling
column_dtypes = {col: str(df[col].dtype) for col in df.columns}
joblib.dump(column_dtypes, 'kidney/kidney_column_dtypes.joblib')
# Also save to root directory for testing
joblib.dump(column_dtypes, 'kidney_column_dtypes.joblib')
print("Column datatypes saved for deployment")

# Remove categorical columns that can't be used directly in the model
df_model = df.drop(['age_category'], axis=1)

# Split data into features and target
X = df_model.drop(['classification', 'id'], axis=1)
y = df_model['classification']

# Save feature columns for deployment
joblib.dump(X.columns.tolist(), 'kidney/kidney_features.joblib')
# Also save to root directory for testing
joblib.dump(X.columns.tolist(), 'kidney_features.joblib')
print("Feature list saved for deployment")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling with RobustScaler (better for data with outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for deployment
joblib.dump(scaler, 'kidney/kidney_scaler.joblib')
# Also save to root directory for testing
joblib.dump(scaler, 'kidney_scaler.joblib')
print("Scaler saved for deployment")

# Apply SMOTE to balance classes
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
    model_name = "kidney_stacking_model.joblib"
else:
    best_model = best_rf
    y_pred = rf_y_pred
    print("\nUsing Random Forest for final evaluation")
    model_name = "kidney_model.joblib"

# Save the best model for deployment
joblib.dump(best_model, f'kidney/{model_name}')
# Also save to root directory for testing
joblib.dump(best_model, model_name)
print(f"\nBest model saved as 'kidney/{model_name}' for deployment")

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
plt.savefig('kidney/kidney_confusion_matrix.png')

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
plt.savefig('kidney/kidney_roc_curve.png')

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('kidney/kidney_precision_recall_curve.png')

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
    plt.savefig('kidney/kidney_feature_importance.png')
    
    # Save feature importances for interpretation
    joblib.dump(feature_importances, 'kidney/kidney_feature_importances.joblib')
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
joblib.dump(model_metadata, 'kidney/kidney_model_metadata.joblib')
print("\nModel metadata saved for reference")

# Create a simple prediction function to demonstrate model usage
def predict_kidney_disease(input_data):
    """
    Function to demonstrate model deployment usage.
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing patient features
    
    Returns:
    --------
    prediction : int
        0 for no kidney disease, 1 for kidney disease
    probability : float
        Probability of kidney disease
    """
    # Load saved model components with error handling
    try:
        print("Attempting to load model components from current directory...")
        try:
            numerical_imputer = joblib.load('kidney_numerical_imputer.joblib')
            label_encoders = joblib.load('kidney_label_encoders.joblib')
            feature_eng_info = joblib.load('kidney_feature_engineering_info.joblib')
            column_dtypes = joblib.load('kidney_column_dtypes.joblib')
            scaler = joblib.load('kidney_scaler.joblib')
            model = joblib.load(model_name)
            features = joblib.load('kidney_features.joblib')
            print("All components loaded from current directory successfully.")
        except FileNotFoundError:
            print("Files not found in current directory. Trying kidney/ subdirectory...")
            numerical_imputer = joblib.load('kidney/kidney_numerical_imputer.joblib')
            label_encoders = joblib.load('kidney/kidney_label_encoders.joblib')
            feature_eng_info = joblib.load('kidney/kidney_feature_engineering_info.joblib')
            column_dtypes = joblib.load('kidney/kidney_column_dtypes.joblib')
            scaler = joblib.load('kidney/kidney_scaler.joblib')
            model = joblib.load(f'kidney/{model_name}')
            features = joblib.load('kidney/kidney_features.joblib')
            print("All components loaded from kidney/ subdirectory successfully.")
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        print("\nDiagnostic information:")
        print(f"Current working directory: {os.getcwd()}")
        for file_name in ['kidney_numerical_imputer.joblib', 'kidney_label_encoders.joblib', 
                         'kidney_feature_engineering_info.joblib', 'kidney_column_dtypes.joblib',
                         'kidney_scaler.joblib', model_name, 'kidney_features.joblib']:
            print(f"{file_name} exists in current dir: {os.path.exists(file_name)}")
            print(f"kidney/{file_name} exists: {os.path.exists('kidney/' + file_name)}")
        raise
    
    print("Processing input data...")
    # Create a pandas DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Convert numerical columns to the right type
    numerical_cols = feature_eng_info['numerical_cols']
    for col in numerical_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Apply label encoding to categorical columns
    categorical_cols = feature_eng_info['categorical_cols']
    for col in categorical_cols:
        if col in input_df.columns and col in label_encoders:
            # Handle missing values
            if input_df[col].iloc[0] == '' or pd.isna(input_df[col].iloc[0]):
                input_df[col] = 'Missing'
            
            # Apply label encoding
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except:
                # If value not seen during training, use most frequent value
                input_df[col] = 0  # Default value
    
    # Apply imputation to numerical columns
    print("Applying imputation...")
    if set(numerical_cols).issubset(input_df.columns):
        input_df[numerical_cols] = numerical_imputer.transform(input_df[numerical_cols])
    
    # Add derived features (same as in training)
    print("Adding engineered features...")
    input_df['renal_function'] = input_df['bu'] / input_df['sc']
    input_df['anemia_risk'] = np.where(input_df['hemo'] < 12, 1, 0)
    
    input_df['bp_category'] = pd.cut(
        input_df['bp'], 
        bins=feature_eng_info['bp_bins'],
        labels=[0, 1, 2, 3, 4]
    )
    
    input_df['kidney_function_score'] = input_df['sc'] + (input_df['bu'] / 10)
    
    try:
        input_df['eGFR'] = 186 * (input_df['sc']**-1.154) * (input_df['age']**-0.203)
        input_df['eGFR'] = input_df['eGFR'].replace([np.inf, -np.inf], np.nan)
    except:
        input_df['eGFR'] = 0  # Default value if calculation fails
    
    input_df['BUN_Creatinine_ratio'] = input_df['bu'] / input_df['sc']
    input_df['BUN_Creatinine_ratio'] = input_df['BUN_Creatinine_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Create age categories
    input_df['age_category'] = pd.cut(
        input_df['age'], 
        bins=feature_eng_info['age_bins'],
        labels=['Young', 'Middle', 'Senior', 'Elderly']
    )
    
    # Create combined risk score
    input_df['combined_risk_score'] = (
        input_df['sc'] * 2 + 
        input_df['bu'] / 10 + 
        (20 - input_df['hemo']) * 0.5 + 
        input_df['al'] * 0.5 + 
        (input_df['bp'] - 90) / 10
    )
    
    # Create other derived features
    input_df['htn_dm_interaction'] = input_df['htn'] * input_df['dm']
    input_df['anemia_ckd_corr'] = input_df['anemia_risk'] * (input_df['sc'] > 1.2).astype(int)
    input_df['proteinuria_risk'] = np.where(input_df['al'] > 1, 1, 0)
    input_df['kidney_stone_risk'] = np.where(
        (input_df['sg'] > 1.02) | (input_df['rc'] < 4.0),
        1, 0
    )
    
    # One-hot encode age_category
    print("Creating one-hot encoded features...")
    age_cat_dummies = pd.get_dummies(input_df['age_category'], drop_first=True, prefix='age_cat')
    input_df = pd.concat([input_df, age_cat_dummies], axis=1)
    
    # Handle any missing values in numerical features
    print("Handling any missing values...")
    for col in input_df.columns:
        if input_df[col].isnull().sum() > 0:
            # Check if column is numeric - using a safer method than np.issubdtype
            dtype_str = str(input_df[col].dtype)
            is_numeric = ('float' in dtype_str) or ('int' in dtype_str)
            
            if is_numeric:
                input_df[col].fillna(0, inplace=True)
            else:
                # For categorical/object columns
                input_df[col].fillna(input_df[col].mode().iloc[0] if not input_df[col].mode().empty else 0, inplace=True)
    
    # Drop non-numeric columns and columns not used for prediction
    print("Dropping unnecessary columns...")
    input_df = input_df.drop(['id', 'age_category'], axis=1, errors='ignore')
    
    # Make sure we have all the required columns in the right order
    print("Aligning features with training data...")
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[features]
    
    # Apply scaling
    print("Applying scaling...")
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    print("Making prediction...")
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    print(f"Prediction: {prediction} (Probability: {probability:.2%})")
    return prediction, probability

# Test prediction function with an example
print("\nTesting model deployment with an example case...")
test_patient = {
    'id': 1,
    'age': 60,
    'bp': 80,
    'sg': 1.020,
    'al': 1.0,
    'su': 0.0,
    'rbc': 'normal',
    'pc': 'normal',
    'pcc': 'notpresent',
    'ba': 'notpresent',
    'bgr': 120,
    'bu': 36,
    'sc': 1.2,
    'sod': 135,
    'pot': 4.0,
    'hemo': 15.4,
    'pcv': 44,
    'wc': 7800,
    'rc': 5.2,
    'htn': 'yes',
    'dm': 'yes',
    'cad': 'no',
    'appet': 'good',
    'pe': 'no',
    'ane': 'no'
}

try:
    prediction, probability = predict_kidney_disease(test_patient)
    print(f"Prediction: {'Kidney Disease' if prediction == 1 else 'No Kidney Disease'}")
    print(f"Probability of Kidney Disease: {probability:.2%}")
    print("\nModel testing successful!")
except Exception as e:
    print(f"\nError during testing: {str(e)}")
    print("\nDiagnostic information:")
    
    # Check if model files exist
    print(f"kidney_numerical_imputer.joblib exists: {os.path.exists('kidney_numerical_imputer.joblib')}")
    print(f"kidney/kidney_numerical_imputer.joblib exists: {os.path.exists('kidney/kidney_numerical_imputer.joblib')}")
    print(f"kidney_label_encoders.joblib exists: {os.path.exists('kidney_label_encoders.joblib')}")
    print(f"kidney/kidney_label_encoders.joblib exists: {os.path.exists('kidney/kidney_label_encoders.joblib')}")

print("\nImproved model training and evaluation completed. Model deployed for future use.")