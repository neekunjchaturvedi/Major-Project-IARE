import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime

# Set page configuration
st.set_page_config(
    page_title="Chronic Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define function to check if all model files exist
def check_model_files():
    required_files = {
        'diabetes': [
            'diabetes/diabetes_model.joblib', 
            'diabetes/diabetes_scaler.joblib', 
            'diabetes/diabetes_imputer.joblib',
            'diabetes/diabetes_features.joblib'
        ],
        'heart': [
            'heart/heart_model.joblib', 
            'heart/heart_preprocessor.joblib',
            'heart/heart_feature_engineering_info.joblib'
        ],
        'kidney': [
            'kidney/kidney_model.joblib', 
            'kidney/kidney_scaler.joblib',
            'kidney/kidney_numerical_imputer.joblib',
            'kidney/kidney_label_encoders.joblib',
            'kidney/kidney_feature_engineering_info.joblib',
            'kidney/kidney_features.joblib'
        ]
    }
    
    missing_files = {}
    
    for disease, files in required_files.items():
        missing = [file for file in files if not os.path.exists(file)]
        if missing:
            missing_files[disease] = missing
    
    return missing_files

# Load Diabetes Model and related files
def load_diabetes_model():
    model_dir = "diabetes"
    try:
        print("Trying to load from diabetes directory...")
        model = joblib.load(os.path.join(model_dir, 'diabetes_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'diabetes_scaler.joblib'))
        imputer = joblib.load(os.path.join(model_dir, 'diabetes_imputer.joblib'))
        
        # Try both possible names for feature engineering info
        try:
            feature_info = joblib.load(os.path.join(model_dir, 'diabetes_feature_engineering_info.joblib'))
            info_key = 'feature_engineering_info'
        except FileNotFoundError:
            try:
                feature_info = joblib.load(os.path.join(model_dir, 'diabetes_preprocessing_info.joblib'))
                info_key = 'preprocessing_info'
            except FileNotFoundError:
                # Create a basic version if neither file exists
                feature_info = {
                    'zero_columns': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
                    'bmi_bins': [0, 18.5, 24.9, 29.9, 100],
                    'age_bins': [20, 30, 40, 50, 60, 100],
                    'glucose_bins': [0, 70, 99, 126, 300],
                    'bp_bins': [0, 60, 80, 120, 140, 200],
                    'pregnancies_bins': [-1, 0, 2, 4, 12, 20],
                    'insulin_bins': [0, 16, 166, 500, 1000]
                }
                info_key = 'feature_engineering_info'
        
        features = joblib.load(os.path.join(model_dir, 'diabetes_features.joblib'))
    except FileNotFoundError:
        print("Trying to load from root directory...")
        # Try loading from root directory as fallback
        model = joblib.load('diabetes_model.joblib')
        scaler = joblib.load('diabetes_scaler.joblib')
        imputer = joblib.load('diabetes_imputer.joblib')
        
        # Try both possible names for feature engineering info
        try:
            feature_info = joblib.load('diabetes_feature_engineering_info.joblib')
            info_key = 'feature_engineering_info'
        except FileNotFoundError:
            try:
                feature_info = joblib.load('diabetes_preprocessing_info.joblib')
                info_key = 'preprocessing_info'
            except FileNotFoundError:
                # Create a basic version if neither file exists
                feature_info = {
                    'zero_columns': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
                    'bmi_bins': [0, 18.5, 24.9, 29.9, 100],
                    'age_bins': [20, 30, 40, 50, 60, 100],
                    'glucose_bins': [0, 70, 99, 126, 300],
                    'bp_bins': [0, 60, 80, 120, 140, 200],
                    'pregnancies_bins': [-1, 0, 2, 4, 12, 20],
                    'insulin_bins': [0, 16, 166, 500, 1000]
                }
                info_key = 'feature_engineering_info'
        
        features = joblib.load('diabetes_features.joblib')
    
    return {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        info_key: feature_info,
        'features': features
    }

# Load Heart Disease Model and related files
def load_heart_model():
    model_dir = "heart"
    try:
        model = joblib.load(os.path.join(model_dir, 'heart_model.joblib'))
        preprocessor = joblib.load(os.path.join(model_dir, 'heart_preprocessor.joblib'))
        feature_engineering_info = joblib.load(os.path.join(model_dir, 'heart_feature_engineering_info.joblib'))
    except FileNotFoundError:
        # Try loading from root directory as fallback
        model = joblib.load('heart_model.joblib')
        preprocessor = joblib.load('heart_preprocessor.joblib')
        feature_engineering_info = joblib.load('heart_feature_engineering_info.joblib')
    
    return {
        'model': model,
        'preprocessor': preprocessor,
        'feature_engineering_info': feature_engineering_info
    }

# Load Kidney Disease Model and related files
def load_kidney_model():
    model_dir = "kidney"
    try:
        model = joblib.load(os.path.join(model_dir, 'kidney_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'kidney_scaler.joblib'))
        numerical_imputer = joblib.load(os.path.join(model_dir, 'kidney_numerical_imputer.joblib'))
        label_encoders = joblib.load(os.path.join(model_dir, 'kidney_label_encoders.joblib'))
        feature_engineering_info = joblib.load(os.path.join(model_dir, 'kidney_feature_engineering_info.joblib'))
        features = joblib.load(os.path.join(model_dir, 'kidney_features.joblib'))
    except FileNotFoundError:
        # Try loading from root directory as fallback
        model = joblib.load('kidney_model.joblib')
        scaler = joblib.load('kidney_scaler.joblib')
        numerical_imputer = joblib.load('kidney_numerical_imputer.joblib')
        label_encoders = joblib.load('kidney_label_encoders.joblib')
        feature_engineering_info = joblib.load('kidney_feature_engineering_info.joblib')
        features = joblib.load('kidney_features.joblib')
    
    return {
        'model': model,
        'scaler': scaler,
        'numerical_imputer': numerical_imputer,
        'label_encoders': label_encoders,
        'feature_engineering_info': feature_engineering_info,
        'features': features
    }

# Function to preprocess inputs for diabetes prediction
def preprocess_diabetes_inputs(inputs, model_data):
    """
    Preprocess diabetes inputs to match EXACTLY the features expected by the model.
    """
    # Create a DataFrame with the input values
    input_df = pd.DataFrame([inputs])
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        st.write(f"Debug - Raw diabetes input shape: {input_df.shape}")
    
    # First, extract information about the expected features - excluding 'Outcome' if it's there
    expected_features = [col for col in model_data['features'] if col != 'Outcome']
    expected_count = len(expected_features)
    
    if st.session_state.get('debug_mode', False):
        st.write(f"Debug - Diabetes model expects {expected_count} features")
        if expected_count <= 20:  # Show all if small number
            st.write(f"Expected features: {expected_features}")
        else:
            st.write(f"First 10 expected: {expected_features[:10]}")
            st.write(f"Last 5 expected: {expected_features[-5:]}")
    
    # Create a DataFrame with exactly the right column names and default zeros
    final_df = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # Start by copying the basic features
    base_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    for col in base_features:
        if col in input_df.columns and col in final_df.columns:
            final_df[col] = input_df[col]
    
    # Handle any imputation needed for zeros in medical values
    info_key = 'feature_engineering_info' if 'feature_engineering_info' in model_data else 'preprocessing_info'
    if info_key in model_data:
        feature_info = model_data[info_key]
        zero_columns = feature_info.get('zero_columns', ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'])
        
        for col in zero_columns:
            if col in final_df.columns and final_df[col].iloc[0] == 0:
                final_df[col] = np.nan
    
        # Apply imputation if we have imputer
        if 'imputer' in model_data:
            try:
                # Create temp df for imputation that includes Outcome
                imp_df = final_df.copy()
                if 'Outcome' not in imp_df:
                    imp_df['Outcome'] = 0
                
                # Only select columns the imputer knows about
                imp_cols = list(set(imp_df.columns) & set(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']))
                
                # Apply imputation
                imp_values = model_data['imputer'].transform(imp_df[imp_cols])
                imp_result = pd.DataFrame(imp_values, columns=imp_cols)
                
                # Copy back the imputed values
                for col in zero_columns:
                    if col in imp_result.columns and col in final_df.columns:
                        final_df[col] = imp_result[col]
            except Exception as e:
                if st.session_state.get('debug_mode', False):
                    st.warning(f"Imputation failed: {str(e)}")
                # Fill with medians if imputation fails
                for col in zero_columns:
                    if col in final_df.columns and pd.isna(final_df[col].iloc[0]):
                        if col == 'Glucose':
                            final_df[col] = 120
                        elif col == 'BloodPressure':
                            final_df[col] = 80
                        elif col == 'SkinThickness':
                            final_df[col] = 20
                        elif col == 'Insulin':
                            final_df[col] = 79
                        elif col == 'BMI':
                            final_df[col] = 32
    
    # Now directly fill in derived features if they exist in the expected feature list
    if 'Insulin_Glucose_Ratio' in expected_features:
        final_df['Insulin_Glucose_Ratio'] = final_df['Insulin'] / (final_df['Glucose'] + 1)
    
    if 'BMI_Age_Interaction' in expected_features:
        final_df['BMI_Age_Interaction'] = final_df['BMI'] * final_df['Age']
    
    if 'Diabetes_Risk_Score' in expected_features:
        final_df['Diabetes_Risk_Score'] = (
            (final_df['Glucose']/100) + 
            (final_df['BMI']/25) + 
            (final_df['Age']/50) + 
            (final_df['DiabetesPedigreeFunction']*2)
        )
    
    if 'Metabolic_Syndrome_Risk' in expected_features:
        conditions = (
            (final_df['BMI'] > 30) | 
            (final_df['BloodPressure'] > 130) | 
            (final_df['Glucose'] > 110)
        )
        final_df['Metabolic_Syndrome_Risk'] = conditions.astype(int)
    
    # Handle binary category columns directly (exact matches required)
    # BMI Category
    bmi_value = final_df['BMI'].iloc[0]
    if 'BMI_Category_Normal' in expected_features:
        final_df['BMI_Category_Normal'] = 1 if 18.5 <= bmi_value <= 24.9 else 0
    if 'BMI_Category_Overweight' in expected_features:
        final_df['BMI_Category_Overweight'] = 1 if 25.0 <= bmi_value < 30.0 else 0
    if 'BMI_Category_Obese' in expected_features:
        final_df['BMI_Category_Obese'] = 1 if bmi_value >= 30.0 else 0
    
    # Age Group
    age_value = final_df['Age'].iloc[0]
    if 'Age_Group_30-40' in expected_features:
        final_df['Age_Group_30-40'] = 1 if 30 <= age_value < 40 else 0
    if 'Age_Group_40-50' in expected_features:
        final_df['Age_Group_40-50'] = 1 if 40 <= age_value < 50 else 0
    if 'Age_Group_50-60' in expected_features:
        final_df['Age_Group_50-60'] = 1 if 50 <= age_value < 60 else 0
    if 'Age_Group_60+' in expected_features:
        final_df['Age_Group_60+'] = 1 if age_value >= 60 else 0
    
    # Glucose Category
    glucose_value = final_df['Glucose'].iloc[0]
    if 'Glucose_Category_Normal' in expected_features:
        final_df['Glucose_Category_Normal'] = 1 if 70 <= glucose_value < 100 else 0
    if 'Glucose_Category_Prediabetes' in expected_features:
        final_df['Glucose_Category_Prediabetes'] = 1 if 100 <= glucose_value < 126 else 0
    if 'Glucose_Category_Diabetes' in expected_features:
        final_df['Glucose_Category_Diabetes'] = 1 if glucose_value >= 126 else 0
    
    # BP Category
    bp_value = final_df['BloodPressure'].iloc[0]
    if 'BP_Category_Low' in expected_features:
        final_df['BP_Category_Low'] = 1 if 60 <= bp_value < 80 else 0
    if 'BP_Category_Normal' in expected_features:
        final_df['BP_Category_Normal'] = 1 if 80 <= bp_value < 120 else 0
    if 'BP_Category_High' in expected_features:
        final_df['BP_Category_High'] = 1 if 120 <= bp_value < 140 else 0
    if 'BP_Category_Very_High' in expected_features:
        final_df['BP_Category_Very_High'] = 1 if bp_value >= 140 else 0
    
    # Pregnancy Risk
    preg_value = final_df['Pregnancies'].iloc[0]
    if 'Pregnancy_Risk_Few' in expected_features:
        final_df['Pregnancy_Risk_Few'] = 1 if 1 <= preg_value <= 2 else 0
    if 'Pregnancy_Risk_Moderate' in expected_features:
        final_df['Pregnancy_Risk_Moderate'] = 1 if 3 <= preg_value <= 4 else 0
    if 'Pregnancy_Risk_High' in expected_features:
        final_df['Pregnancy_Risk_High'] = 1 if 5 <= preg_value <= 12 else 0
    if 'Pregnancy_Risk_Very_High' in expected_features:
        final_df['Pregnancy_Risk_Very_High'] = 1 if preg_value > 12 else 0
    
    # Insulin Category
    insulin_value = final_df['Insulin'].iloc[0]
    if 'Insulin_Category_Normal' in expected_features:
        final_df['Insulin_Category_Normal'] = 1 if 16 <= insulin_value < 166 else 0
    if 'Insulin_Category_High' in expected_features:
        final_df['Insulin_Category_High'] = 1 if 166 <= insulin_value < 500 else 0
    if 'Insulin_Category_Very_High' in expected_features:
        final_df['Insulin_Category_Very_High'] = 1 if insulin_value >= 500 else 0
    
    # Double-check that all expected features are present
    for col in expected_features:
        if col not in final_df.columns:
            final_df[col] = 0
            if st.session_state.get('debug_mode', False):
                st.warning(f"Added missing column: {col}")
    
    # Make sure we're returning exactly the right columns in the right order
    final_df = final_df[expected_features]
    
    # Debug checks
    if st.session_state.get('debug_mode', False):
        st.write(f"Debug - Final diabetes DataFrame shape: {final_df.shape}")
        if final_df.shape[1] != expected_count:
            st.error(f"Column count mismatch! Expected {expected_count}, got {final_df.shape[1]}")
    
    # Apply scaling
    input_scaled = model_data['scaler'].transform(final_df)
    
    return input_scaled

# Function to preprocess inputs for heart disease prediction
def preprocess_heart_inputs(inputs, model_data):
    # Create a DataFrame with the input values
    input_df = pd.DataFrame([inputs])
    
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
    cp_weights = {0: 0, 1: 1, 2: 2, 3: 3}
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
    
    # Apply the preprocessor pipeline to transform the data
    input_processed = model_data['preprocessor'].transform(input_df)
    
    return input_processed

# Function to preprocess inputs for kidney disease prediction
def preprocess_kidney_inputs(inputs, model_data):
    """
    Preprocess kidney inputs to match EXACTLY the features expected by the model.
    """
    # Create a DataFrame with the input values
    input_df = pd.DataFrame([inputs])
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        st.write(f"Debug - Raw kidney input shape: {input_df.shape}")
    
    # First, extract information about the expected features
    expected_features = model_data['features']
    expected_count = len(expected_features)
    
    if st.session_state.get('debug_mode', False):
        st.write(f"Debug - Kidney model expects {expected_count} features")
        if expected_count < 10:  # Show all if small number
            st.write(f"Expected features: {expected_features}")
        else:
            st.write(f"First 5 expected: {expected_features[:5]}")
            st.write(f"Last 5 expected: {expected_features[-5:]}")
    
    # Create a DataFrame with exactly the right column names and default zeros
    final_df = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # Get information from model data
    numerical_cols = model_data['feature_engineering_info']['numerical_cols']
    categorical_cols = model_data['feature_engineering_info']['categorical_cols']
    
    # Copy over the basic numerical features first
    for col in numerical_cols:
        if col in input_df.columns and col in final_df.columns:
            final_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Handle categorical features with label encoding
    for col in categorical_cols:
        if col in input_df.columns and col in final_df.columns and col in model_data['label_encoders']:
            try:
                # Handle missing values
                if input_df[col].iloc[0] == '' or pd.isna(input_df[col].iloc[0]):
                    input_df[col] = 'Missing'
                
                # Apply label encoding
                final_df[col] = model_data['label_encoders'][col].transform(input_df[col])
            except:
                # If encoding fails, use a default value of 0
                final_df[col] = 0
    
    # Apply numerical imputation if needed for specific columns
    numeric_cols_with_missing = [col for col in numerical_cols if col in final_df.columns and final_df[col].isnull().any()]
    if numeric_cols_with_missing:
        try:
            # Create a temporary dataframe with just the numeric columns that need imputation
            numeric_df = final_df[numeric_cols_with_missing].copy()
            imputed_values = model_data['numerical_imputer'].transform(numeric_df)
            for i, col in enumerate(numeric_cols_with_missing):
                final_df[col] = imputed_values[:, i]
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.warning(f"Numeric imputation error: {str(e)}")
            # Fill missing values with zeros if imputation fails
            final_df[numeric_cols_with_missing] = final_df[numeric_cols_with_missing].fillna(0)
    
    # Now handle the derived features and make sure they're in the expected list
    if 'renal_function' in expected_features:
        final_df['renal_function'] = final_df['bu'] / (final_df['sc'] + 0.001)  # Add small epsilon to avoid division by zero
    
    if 'anemia_risk' in expected_features:
        final_df['anemia_risk'] = np.where(final_df['hemo'] < 12, 1, 0)
    
    if 'kidney_function_score' in expected_features:
        final_df['kidney_function_score'] = final_df['sc'] + (final_df['bu'] / 10)
    
    if 'eGFR' in expected_features:
        try:
            final_df['eGFR'] = 186 * (final_df['sc']**-1.154) * (final_df['age']**-0.203)
            final_df['eGFR'] = final_df['eGFR'].replace([np.inf, -np.inf], np.nan).fillna(0)
        except:
            final_df['eGFR'] = 0
    
    if 'BUN_Creatinine_ratio' in expected_features:
        final_df['BUN_Creatinine_ratio'] = final_df['bu'] / (final_df['sc'] + 0.001)
        final_df['BUN_Creatinine_ratio'] = final_df['BUN_Creatinine_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if 'combined_risk_score' in expected_features:
        final_df['combined_risk_score'] = (
            final_df['sc'] * 2 + 
            final_df['bu'] / 10 + 
            (20 - final_df['hemo']) * 0.5 + 
            final_df['al'] * 0.5 + 
            (final_df['bp'] - 90) / 10
        )
    
    if 'htn_dm_interaction' in expected_features:
        if 'htn' in final_df.columns and 'dm' in final_df.columns:
            final_df['htn_dm_interaction'] = final_df['htn'] * final_df['dm']
    
    if 'anemia_ckd_corr' in expected_features and 'anemia_risk' in final_df.columns:
        final_df['anemia_ckd_corr'] = final_df['anemia_risk'] * (final_df['sc'] > 1.2).astype(int)
    
    if 'proteinuria_risk' in expected_features:
        final_df['proteinuria_risk'] = np.where(final_df['al'] > 1, 1, 0)
    
    if 'kidney_stone_risk' in expected_features:
        final_df['kidney_stone_risk'] = np.where(
            (final_df['sg'] > 1.02) | (final_df['rc'] < 4.0),
            1, 0
        )
    
    # Handle one-hot encoded age category features directly rather than creating and transforming
    age_cat_cols = [col for col in expected_features if col.startswith('age_cat_')]
    if age_cat_cols:
        # Determine age category
        age_val = final_df['age'].iloc[0]
        age_bins = model_data['feature_engineering_info']['age_bins']
        
        # Set the appropriate age category column to 1
        if age_val >= age_bins[0] and age_val < age_bins[1]:  # Young
            pass  # The baseline category is usually omitted in one-hot encoding
        elif age_val >= age_bins[1] and age_val < age_bins[2]:  # Middle
            if 'age_cat_Middle' in expected_features:
                final_df['age_cat_Middle'] = 1
        elif age_val >= age_bins[2] and age_val < age_bins[3]:  # Senior
            if 'age_cat_Senior' in expected_features:
                final_df['age_cat_Senior'] = 1
        elif age_val >= age_bins[3]:  # Elderly
            if 'age_cat_Elderly' in expected_features:
                final_df['age_cat_Elderly'] = 1
    
    # Handle any bp_category features
    bp_cat_cols = [col for col in expected_features if col.startswith('bp_category_')]
    if 'bp_category' in expected_features:
        # Use the bins from feature engineering info
        bp_bins = model_data['feature_engineering_info']['bp_bins']
        bp_val = final_df['bp'].iloc[0]
        
        # Find which bin the bp value falls into
        bin_idx = 0
        for i in range(len(bp_bins)-1):
            if bp_val >= bp_bins[i] and bp_val < bp_bins[i+1]:
                bin_idx = i
                break
        
        # Set the bp_category value
        final_df['bp_category'] = bin_idx
    
    # Handle any directly encoded columns in expected features
    for col in expected_features:
        if col not in final_df.columns:
            # Check if it's a direct encoding we can set
            if col.endswith('_1') or col.endswith('_0'):
                base_col = col.rsplit('_', 1)[0]
                if base_col in input_df.columns:
                    val = input_df[base_col].iloc[0]
                    if (col.endswith('_1') and val == 1) or (col.endswith('_0') and val == 0):
                        final_df[col] = 1
            else:
                # Set to 0 if we don't know what it is
                final_df[col] = 0
    
    # Make sure we're returning exactly the expected columns in the right order
    final_df = final_df[expected_features]
    
    # Debug checks
    if st.session_state.get('debug_mode', False):
        st.write(f"Debug - Final kidney DF shape: {final_df.shape}")
        if final_df.shape[1] != expected_count:
            st.error(f"Column count mismatch! Expected {expected_count}, got {final_df.shape[1]}")
            st.write(f"Missing columns: {set(expected_features) - set(final_df.columns)}")
            st.write(f"Extra columns: {set(final_df.columns) - set(expected_features)}")
    
    # Apply scaling
    input_scaled = model_data['scaler'].transform(final_df)
    
    return input_scaled

# Main application
def main():
    # Initialize session state for debugging
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Title and introduction
    st.title("Chronic Disease Prediction App 🏥")
    st.markdown("""
    This app uses machine learning to predict the risk of three chronic diseases:
    * **Diabetes**
    * **Heart Disease**
    * **Kidney Disease**
    
    Select a disease from the sidebar and enter the required information to get a prediction.
    """)
    
    # Display current date and user info
    current_date = "2025-04-27 16:42:56"  # You can replace with datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_user = "22951a3363"
    
    st.sidebar.markdown(f"**Current Date and Time (UTC):** {current_date}")
    st.sidebar.markdown(f"**Current User's Login:** {current_user}")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Enable Debug Mode")
    st.session_state.debug_mode = debug_mode
    
    # Additional debug information
    if debug_mode:
        st.sidebar.markdown("### Debug Information")
        st.sidebar.write(f"App Version: 1.0.3")
        st.sidebar.write(f"Streamlit version: {st.__version__}")
        
        # Count expected features for each model
        try:
            diabetes_model = load_diabetes_model()
            diabetes_features = [f for f in diabetes_model['features'] if f != 'Outcome']
            st.sidebar.write(f"Diabetes model features: {len(diabetes_features)}")
        except:
            st.sidebar.write("Couldn't load diabetes model")
            
        try:
            kidney_model = load_kidney_model()
            st.sidebar.write(f"Kidney model features: {len(kidney_model['features'])}")
        except:
            st.sidebar.write("Couldn't load kidney model")
    
    # Check if all required model files exist
    missing_files = check_model_files()
    
    if missing_files and not debug_mode:
        st.error("Some required model files are missing. Please make sure all models are trained correctly.")
        for disease, files in missing_files.items():
            st.warning(f"Missing files for {disease}: {', '.join(files)}")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Disease Selection")
    disease = st.sidebar.radio(
        "Choose a disease to predict:",
        ["Diabetes", "Heart Disease", "Kidney Disease"]
    )
    
    try:
        if disease == "Diabetes":
            predict_diabetes(debug_mode)
        elif disease == "Heart Disease":
            predict_heart_disease(debug_mode)
        elif disease == "Kidney Disease":
            predict_kidney_disease(debug_mode)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Make sure all the required model files are in the correct directories.")
        if debug_mode:
            st.exception(e)

# Diabetes prediction page
def predict_diabetes(debug_mode=False):
    st.header("Diabetes Risk Prediction")
    st.info("Enter your health information to predict diabetes risk.")
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    
    with col2:
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
        age = st.number_input("Age (years)", min_value=21, max_value=100, value=35)
    
    # Create input dictionary with EXACT column names from training
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    
    # Prediction button
    if st.button("Predict Diabetes Risk"):
        with st.spinner("Processing..."):
            try:
                # Load model and related data
                model_data = load_diabetes_model()
                
                if debug_mode:
                    st.write("Debug - Model loaded successfully")
                    st.write(f"Debug - Model data keys: {list(model_data.keys())}")
                    st.write(f"Debug - Input columns: {list(input_data.keys())}")
                    if 'features' in model_data:
                        st.write(f"Debug - Model expects these features: {[f for f in model_data['features'] if f != 'Outcome']}")
                
                # Preprocess inputs - this function now ensures EXACTLY the right number of features
                processed_input = preprocess_diabetes_inputs(input_data, model_data)
                
                if debug_mode:
                    st.write(f"Debug - Processed input shape: {processed_input.shape}")
                
                # Make prediction
                prediction = model_data['model'].predict(processed_input)
                probability = model_data['model'].predict_proba(processed_input)[0][1]
                
                # Show prediction result
                st.subheader("Prediction Result")
                
                if prediction[0] == 1:
                    st.error(f"⚠️ **High Risk of Diabetes** (Probability: {probability:.2%})")
                else:
                    st.success(f"✅ **Low Risk of Diabetes** (Probability: {probability:.2%})")
                
                # Show risk factors
                st.subheader("Key Risk Factors")
                risk_factors = []
                
                if glucose > 125:
                    risk_factors.append(f"High glucose level ({glucose} mg/dL, normal is <100 mg/dL)")
                
                if bmi > 30:
                    risk_factors.append(f"High BMI ({bmi}, obesity threshold is >30)")
                    
                if diabetes_pedigree > 0.8:
                    risk_factors.append(f"High diabetes pedigree function ({diabetes_pedigree}, higher values indicate stronger family history)")
                
                if age > 45:
                    risk_factors.append(f"Age over 45 ({age} years)")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.write("No major risk factors identified.")
                
                # Display recommendations
                st.subheader("Recommendations")
                
                if prediction[0] == 1:
                    st.markdown("""
                    * Consult with a healthcare professional for a proper diagnosis
                    * Monitor blood glucose levels regularly
                    * Adopt a low-sugar, low-carbohydrate diet
                    * Increase physical activity (at least 30 minutes daily)
                    * Maintain a healthy weight
                    """)
                else:
                    st.markdown("""
                    * Continue maintaining a healthy lifestyle
                    * Regular exercise and balanced diet
                    * Periodic health check-ups
                    * Be mindful of portion sizes and sugar intake
                    """)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if debug_mode:
                    st.exception(e)
                    
                    # Try to provide more diagnostics
                    try:
                        model_data = load_diabetes_model()
                        st.write("Model Data Keys:", list(model_data.keys()))
                        if 'features' in model_data:
                            st.write("Expected Features:", model_data['features'])
                    except Exception as e2:
                        st.write(f"Could not load model for diagnostics: {str(e2)}")

# Heart disease prediction page
def predict_heart_disease(debug_mode=False):
    st.header("Heart Disease Risk Prediction")
    st.info("Enter your health information to predict heart disease risk.")
    
    # Create input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", options=["Female", "Male"], index=0)
        cp = st.selectbox("Chest Pain Type", options=[
            "Typical angina", 
            "Atypical angina", 
            "Non-anginal pain", 
            "Asymptomatic"
        ], index=0)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=["No", "Yes"], index=0)
        restecg = st.selectbox("Resting ECG Results", options=[
            "Normal", 
            "ST-T wave abnormality", 
            "Left ventricular hypertrophy"
        ], index=0)
        thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"], index=0)
    
    with col3:
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=0.0, format="%.1f")
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
            "Upsloping", 
            "Flat", 
            "Downsloping"
        ], index=0)
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
        thal = st.selectbox("Thalassemia", options=[
            "Normal", 
            "Fixed defect", 
            "Reversible defect",
            "Unknown"
        ], index=0)
    
    # Map inputs to the format expected by the model
    sex_map = {"Female": 0, "Male": 1}
    cp_map = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
    fbs_map = {"No": 0, "Yes": 1}
    restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    exang_map = {"No": 0, "Yes": 1}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": "normal", "Fixed defect": "fixed_defect", "Reversible defect": "reversible_defect", "Unknown": "unknown"}
    
    # Create input dictionary
    input_data = {
        'age': age,
        'sex': sex_map[sex],
        'cp': cp_map[cp],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_map[fbs],
        'restecg': restecg_map[restecg],
        'thalach': thalach,
        'exang': exang_map[exang],
        'oldpeak': oldpeak,
        'slope': slope_map[slope],
        'ca': ca,
        'thal': thal_map[thal]
    }
    
    # Prediction button
    if st.button("Predict Heart Disease Risk"):
        with st.spinner("Processing..."):
            try:
                # Load model and related data
                model_data = load_heart_model()
                
                if debug_mode:
                    st.write("Debug - Model loaded successfully")
                    st.write(f"Debug - Model data keys: {list(model_data.keys())}")
                
                # Preprocess inputs
                processed_input = preprocess_heart_inputs(input_data, model_data)
                
                if debug_mode:
                    st.write(f"Debug - Input processed successfully")
                
                # Make prediction
                prediction = model_data['model'].predict(processed_input)
                probability = model_data['model'].predict_proba(processed_input)[0][1]
                
                # Show prediction result
                st.subheader("Prediction Result")
                
                if prediction[0] == 1:
                    st.error(f"⚠️ **High Risk of Heart Disease** (Probability: {probability:.2%})")
                else:
                    st.success(f"✅ **Low Risk of Heart Disease** (Probability: {probability:.2%})")
                
                # Show risk factors
                st.subheader("Key Risk Factors")
                risk_factors = []
                
                if age > 55:
                    risk_factors.append(f"Age over 55 ({age} years)")
                
                if sex_map[sex] == 1:
                    risk_factors.append("Male gender (higher risk than females)")
                    
                if cp_map[cp] == 3:
                    risk_factors.append("Asymptomatic chest pain type (often associated with silent ischemia)")
                
                if trestbps > 140:
                    risk_factors.append(f"High blood pressure ({trestbps} mm Hg, high is >140)")
                    
                if chol > 240:
                    risk_factors.append(f"High cholesterol ({chol} mg/dL, high is >240)")
                    
                if fbs_map[fbs] == 1:
                    risk_factors.append("High fasting blood sugar (>120 mg/dL)")
                    
                if exang_map[exang] == 1:
                    risk_factors.append("Exercise-induced angina")
                    
                if oldpeak > 1.5:
                    risk_factors.append(f"High ST depression ({oldpeak})")
                    
                if ca > 0:
                    risk_factors.append(f"{ca} major vessels colored by fluoroscopy")
                    
                if thal_map[thal] == "reversible_defect":
                    risk_factors.append("Reversible thalassemia defect")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.write("No major risk factors identified.")
                
                # Display recommendations
                st.subheader("Recommendations")
                
                if prediction[0] == 1:
                    st.markdown("""
                    * Consult with a cardiologist as soon as possible
                    * Monitor blood pressure and cholesterol regularly
                    * Follow a heart-healthy diet low in sodium and saturated fats
                    * Regular moderate exercise as advised by your doctor
                    * Take prescribed medications consistently
                    * Avoid smoking and limit alcohol consumption
                    * Consider stress management techniques
                    """)
                else:
                    st.markdown("""
                    * Continue heart-healthy lifestyle habits
                    * Regular cardiovascular exercise (150+ minutes per week)
                    * Diet rich in fruits, vegetables, and whole grains
                    * Limit saturated fats, trans fats, and sodium
                    * Regular health check-ups and cholesterol screenings
                    * Maintain healthy weight
                    """)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if debug_mode:
                    st.exception(e)

# Kidney disease prediction page
def predict_kidney_disease(debug_mode=False):
    st.header("Kidney Disease Risk Prediction")
    st.info("Enter your health information to predict kidney disease risk.")
    st.warning("Note: This form has many medical parameters. If you don't know a value, leave it as the default.")
    
    # Create input tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Demographics & Vitals", "Blood Tests", "Urine Tests & Medical History"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=100, value=50)
            bp = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=200, value=80)
            
        with col2:
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=150.0, value=70.0)
            sg = st.selectbox("Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025], index=2)
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=70, max_value=500, value=120)
            bu = st.number_input("Blood Urea (mg/dL)", min_value=10, max_value=200, value=40)
            sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.4, max_value=15.0, value=1.0, format="%.1f")
            sod = st.number_input("Sodium (mEq/L)", min_value=120, max_value=150, value=135)
            
        with col2:
            pot = st.number_input("Potassium (mEq/L)", min_value=2.5, max_value=7.0, value=4.0, format="%.1f")
            hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=12.0, format="%.1f")
            pcv = st.number_input("Packed Cell Volume", min_value=20, max_value=60, value=40)
            wc = st.number_input("White Blood Cell Count (cells/cmm)", min_value=3000, max_value=20000, value=9000)
            
        with col3:
            rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=2.0, max_value=8.0, value=4.5, format="%.1f")
            al = st.selectbox("Albumin", options=[0, 1, 2, 3, 4, 5], index=0, help="0 = negative, 1-5 = increasing levels")
            su = st.selectbox("Sugar", options=[0, 1, 2, 3, 4, 5], index=0, help="0 = negative, 1-5 = increasing levels")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            rbc = st.selectbox("Red Blood Cells in Urine", options=["normal", "abnormal"], index=0)
            pc = st.selectbox("Pus Cells in Urine", options=["normal", "abnormal"], index=0)
            pcc = st.selectbox("Pus Cell Clumps", options=["notpresent", "present"], index=0)
            ba = st.selectbox("Bacteria", options=["notpresent", "present"], index=0)
            
        with col2:
            htn = st.selectbox("Hypertension", options=["no", "yes"], index=0)
            dm = st.selectbox("Diabetes Mellitus", options=["no", "yes"], index=0)
            cad = st.selectbox("Coronary Artery Disease", options=["no", "yes"], index=0)
            appet = st.selectbox("Appetite", options=["good", "poor"], index=0)
            pe = st.selectbox("Pedal Edema", options=["no", "yes"], index=0)
            ane = st.selectbox("Anemia", options=["no", "yes"], index=0)
    
    # Create input dictionary
    input_data = {
        'id': 1,  # Add a placeholder ID since it's needed but will be ignored
        'age': age,
        'bp': bp,
        'sg': sg,
        'al': al,
        'su': su,
        'rbc': rbc,
        'pc': pc,
        'pcc': pcc,
        'ba': ba,
        'bgr': bgr,
        'bu': bu,
        'sc': sc,
        'sod': sod,
        'pot': pot,
        'hemo': hemo,
        'pcv': pcv,
        'wc': wc,
        'rc': rc,
        'htn': htn,
        'dm': dm,
        'cad': cad,
        'appet': appet,
        'pe': pe,
        'ane': ane
    }
    
    # Prediction button
    if st.button("Predict Kidney Disease Risk"):
        with st.spinner("Processing..."):
            try:
                # Load model and related data
                model_data = load_kidney_model()
                
                if debug_mode:
                    st.write("Debug - Model loaded successfully")
                    st.write(f"Debug - Model data keys: {list(model_data.keys())}")
                
                # Preprocess inputs
                processed_input = preprocess_kidney_inputs(input_data, model_data)
                
                if debug_mode:
                    st.write(f"Debug - Input processed successfully")
                
                # Make prediction
                prediction = model_data['model'].predict(processed_input)
                
                try:
                    probability = model_data['model'].predict_proba(processed_input)[0][1]
                    probability_available = True
                except:
                    probability = None
                    probability_available = False
                
                # Show prediction result
                st.subheader("Prediction Result")
                
                if prediction[0] == 1:
                    st.error(f"⚠️ **High Risk of Kidney Disease**" + (f" (Probability: {probability:.2%})" if probability_available else ""))
                else:
                    st.success(f"✅ **Low Risk of Kidney Disease**" + (f" (Probability: {(1-probability):.2%})" if probability_available else ""))
                
                # Show risk factors
                st.subheader("Key Risk Factors")
                risk_factors = []
                
                if sc > 1.2:
                    risk_factors.append(f"Elevated serum creatinine ({sc} mg/dL, normal is 0.6-1.2 mg/dL)")
                    
                if bu > 40:
                    risk_factors.append(f"Elevated blood urea ({bu} mg/dL, normal is 15-40 mg/dL)")
                    
                if hemo < 12:
                    risk_factors.append(f"Low hemoglobin ({hemo} g/dL, normal is >12 g/dL)")
                    
                if al > 0:
                    risk_factors.append(f"Albumin in urine (level {al})")
                    
                if htn == "yes":
                    risk_factors.append("Hypertension")
                    
                if dm == "yes":
                    risk_factors.append("Diabetes Mellitus")
                    
                if appet == "poor":
                    risk_factors.append("Poor appetite")
                    
                if pe == "yes":
                    risk_factors.append("Pedal edema (swelling in feet and ankles)")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.write("No major risk factors identified.")
                
                # Display recommendations
                st.subheader("Recommendations")
                
                if prediction[0] == 1:
                    st.markdown("""
                    * Consult with a nephrologist as soon as possible
                    * Complete kidney function tests (GFR, BUN, creatinine)
                    * Control blood pressure and blood sugar
                    * Limit sodium, potassium, and phosphorus intake as directed
                    * Stay well-hydrated with appropriate fluid intake
                    * Avoid nephrotoxic medications (NSAIDs, certain antibiotics)
                    * Regular monitoring of kidney function
                    """)
                else:
                    st.markdown("""
                    * Maintain adequate hydration (aim for 2-3 liters of water daily)
                    * Balanced diet with moderate protein intake
                    * Limit sodium intake (below 2,300 mg daily)
                    * Regular exercise
                    * Avoid excessive use of over-the-counter painkillers
                    * Regular health check-ups including kidney function tests
                    * Control blood pressure and blood sugar if relevant
                    """)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if debug_mode:
                    st.exception(e)

# Run the application
if __name__ == '__main__':
    main()