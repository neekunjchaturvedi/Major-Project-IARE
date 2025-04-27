# Kidney Disease Classification using Random Forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading the Kidney Disease dataset...")
df = pd.read_csv("kidney_disease.csv")

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

# Check missing values again
print("\nMissing values after replacing placeholders:")
print(df.isnull().sum())

# 3. Convert numerical columns to the right type
numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Handle categorical features - convert to numerical using LabelEncoder
print("\nEncoding categorical features...")
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    # Fill missing values with a placeholder before encoding
    df[col].fillna('Missing', inplace=True)
    df[col] = label_encoders[col].fit_transform(df[col])

# 5. Handle missing values using KNN imputation for numerical features
print("\nImputing missing values...")
# First, fill missing values in numerical columns with median for visualization
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

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

# 6. Convert all remaining categorical features to numeric
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for any remaining missing values
print("\nMissing values after feature engineering:")
print(df.isnull().sum())

# Final imputation for any remaining missing values
imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split data into features and target
X = df.drop(['classification', 'id'], axis=1)
y = df['classification']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
print("\nTraining the Random Forest model...")

# Base model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")

# Train model with best parameters
best_rf = RandomForestClassifier(random_state=42, **best_params)
best_rf.fit(X_train_scaled, y_train)

# Model Evaluation
print("\nEvaluating the model...")

# Predictions
y_pred = best_rf.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

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
plt.savefig('kidney_confusion_matrix.png')

# ROC Curve
try:
    y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]
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
    plt.savefig('kidney_roc_curve.png')
except:
    print("Could not generate ROC curve (may be due to binary encoding issue).")

# Cross-validation score
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Feature importance
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
plt.savefig('kidney_feature_importance.png')

print("\nModel training and evaluation completed. Results saved.")