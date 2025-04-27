# Diabetes Disease Classification using Random Forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

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

# Replace 0s with NaN for columns where 0 is not a valid value
for column in zero_columns:
    df[column] = df[column].replace(0, np.nan)

# Check missing values after replacing zeros
print("\nMissing values after replacing zeros:")
print(df.isnull().sum())

# Impute missing values using median
imputer = SimpleImputer(strategy='median')
df[zero_columns] = imputer.fit_transform(df[zero_columns])

print("\nData after imputation:")
print(df.describe())

# Feature Engineering
print("\nPerforming feature engineering...")

# 1. Create BMI categories
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], 
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# 2. Create age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100], 
                         labels=['20-30', '30-40', '40-50', '50-60', '60+'])

# 3. Create glucose level categories
df['Glucose_Category'] = pd.cut(df['Glucose'], bins=[0, 70, 99, 126, 300], 
                               labels=['Low', 'Normal', 'Prediabetes', 'Diabetes'])

# 4. Create insulin sensitivity feature
df['Insulin_Glucose_Ratio'] = df['Insulin'] / df['Glucose']

# 5. Create BMI*Age interaction feature
df['BMI_Age_Interaction'] = df['BMI'] * df['Age']

# One-hot encode categorical features
categorical_features = ['BMI_Category', 'Age_Group', 'Glucose_Category']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

print("\nFeatures after engineering:")
print(df_encoded.columns.tolist())

# Split data into features and target
X = df_encoded.drop(['Outcome'], axis=1)
y = df_encoded['Outcome']

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
plt.savefig('diabetes_confusion_matrix.png')

# ROC Curve
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
plt.savefig('diabetes_roc_curve.png')

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
plt.savefig('diabetes_feature_importance.png')

print("\nModel training and evaluation completed. Results saved.")