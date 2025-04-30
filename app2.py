import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Set page configuration
st.set_page_config(
    page_title="Health Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and associated files
@st.cache_resource
def load_diabetes_model():
    model = joblib.load('diabetes_rf_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    feature_names = joblib.load('diabetes_feature_names.pkl')
    return model, scaler, feature_names

@st.cache_resource
def load_heart_disease_model():
    model = joblib.load('heart_disease_rf_model.pkl')
    scaler = joblib.load('heart_disease_scaler.pkl')
    feature_names = joblib.load('heart_disease_feature_names.pkl')
    return model, scaler, feature_names

@st.cache_resource
def load_kidney_disease_model():
    model = joblib.load('kidney_disease_rf_model.pkl')
    scaler = joblib.load('kidney_disease_scaler.pkl')
    feature_names = joblib.load('kidney_disease_feature_names.pkl')
    return model, scaler, feature_names

# Prediction functions
def predict_diabetes(input_data, model, scaler, feature_names):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    return prediction[0], probability[0]

def predict_heart_disease(input_data, model, scaler, feature_names):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    return prediction[0], probability[0]

def predict_kidney_disease(input_data, model, scaler, feature_names):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    return prediction[0], probability[0]

# Get health recommendations
def get_diabetes_recommendations(input_data, diabetes_features):
    recommendations = []
    bmi_index = diabetes_features.index("BMI")
    glucose_index = diabetes_features.index("Glucose")
    age_index = diabetes_features.index("Age")
    
    # BMI recommendations
    bmi = input_data[bmi_index]
    if bmi > 25:
        recommendations.append("📉 **Maintain a healthy weight**: Your BMI is above 25, which increases diabetes risk. Aim for a BMI between 18.5 and 24.9.")
    
    # Glucose recommendations
    glucose = input_data[glucose_index]
    if glucose > 100:
        recommendations.append("🍎 **Monitor blood sugar levels**: Your glucose level is elevated. Focus on low-glycemic foods like vegetables, whole grains, and lean proteins.")
    
    # General recommendations
    recommendations.extend([
        "🏃 **Regular physical activity**: Aim for at least 150 minutes of moderate aerobic activity per week.",
        "🥗 **Healthy eating**: Follow a diet rich in fruits, vegetables, whole grains, and lean proteins. Limit refined carbohydrates and added sugars.",
        "🚭 **Avoid tobacco**: Smoking increases the risk of diabetes complications.",
        "💧 **Stay hydrated**: Drink plenty of water instead of sugary beverages.",
        "😴 **Get adequate sleep**: Aim for 7-8 hours of quality sleep each night."
    ])
    
    # Age-specific recommendations
    age = input_data[age_index]
    if age > 45:
        recommendations.append("🩺 **Regular check-ups**: As you are over 45, schedule regular diabetes screenings with your healthcare provider.")
    
    return recommendations

def get_heart_disease_recommendations(input_data, heart_features):
    recommendations = []
    
    # Extract specific values from input data
    chol_index = heart_features.index("chol")
    trestbps_index = heart_features.index("trestbps")
    age_index = heart_features.index("age")
    sex_index = heart_features.index("sex")
    
    chol = input_data[chol_index]
    bp = input_data[trestbps_index]
    age = input_data[age_index]
    sex = input_data[sex_index]
    
    # Cholesterol recommendations
    if chol > 200:
        recommendations.append("🥑 **Manage cholesterol**: Your cholesterol level is above 200 mg/dL. Focus on heart-healthy foods like fish, nuts, avocados, olive oil, and fiber-rich foods.")
    
    # Blood pressure recommendations
    if bp > 120:
        recommendations.append("🧂 **Control blood pressure**: Your blood pressure reading is elevated. Reduce sodium intake to less than 2,300 mg per day and consider the DASH diet approach.")
    
    # Age and gender specific recommendations
    if age > 45 and sex == 1:  # Male
        recommendations.append("🩺 **Regular cardiac screenings**: Men over 45 have an increased risk of heart disease. Schedule regular check-ups with your healthcare provider.")
    elif age > 55 and sex == 0:  # Female
        recommendations.append("🩺 **Regular cardiac screenings**: Women over 55 have an increased risk of heart disease. Schedule regular check-ups with your healthcare provider.")
    
    # General recommendations
    recommendations.extend([
        "💪 **Regular exercise**: Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week.",
        "🥗 **Heart-healthy diet**: Follow a Mediterranean or DASH diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats.",
        "🚭 **Quit smoking**: If you smoke, quitting is one of the best things you can do for your heart health.",
        "🧘‍♀️ **Manage stress**: Practice stress-reduction techniques like meditation, deep breathing, or yoga.",
        "🍷 **Limit alcohol**: If you drink alcohol, do so in moderation (up to one drink per day for women and up to two for men).",
        "⚖️ **Maintain a healthy weight**: Aim for a BMI between 18.5 and 24.9 to reduce strain on your heart."
    ])
    
    return recommendations

def get_kidney_disease_recommendations(input_data, kidney_features):
    recommendations = []
    
    # Extract specific values from input data if available
    try:
        bp_index = kidney_features.index("bp")
        bp = input_data[bp_index]
        
        bgr_index = kidney_features.index("bgr")
        bgr = input_data[bgr_index]
        
        sc_index = kidney_features.index("sc")
        sc = input_data[sc_index]
        
        # Blood pressure recommendations
        if bp > 120:
            recommendations.append("🧂 **Control blood pressure**: Your blood pressure is elevated. High blood pressure can worsen kidney disease. Aim to keep it below 120/80 mm Hg.")
        
        # Blood glucose recommendations
        if bgr > 100:
            recommendations.append("🍎 **Manage blood sugar levels**: Your blood glucose reading is high. Diabetes is a leading cause of kidney disease. Keep blood glucose levels in a healthy range.")
        
        # Serum creatinine recommendations
        if sc > 1.2:
            recommendations.append("⚠️ **Monitor kidney function**: Your serum creatinine level is elevated, which may indicate reduced kidney function. Regular monitoring is important.")
            
    except ValueError:
        pass  # If any index is not found, skip that recommendation
        
    # General recommendations for kidney health
    recommendations.extend([
        "💧 **Stay hydrated**: Drink plenty of water daily (about 8-10 glasses) to help your kidneys clear sodium and toxins.",
        "🥗 **Kidney-friendly diet**: Limit sodium, processed foods, and red meat. Focus on fruits, vegetables, whole grains, and lean proteins.",
        "💊 **Medication awareness**: Consult your doctor about medications that might affect your kidneys, especially NSAIDs like ibuprofen.",
        "🏃 **Regular exercise**: Physical activity helps control blood pressure and blood sugar levels, both important for kidney health.",
        "🚭 **Avoid smoking**: Smoking can worsen kidney disease and increases the risk of kidney cancer.",
        "🩺 **Regular check-ups**: Get your kidney function tested regularly, especially if you have high blood pressure, diabetes, or a family history of kidney disease.",
        "⚕️ **Limit alcohol**: Excessive alcohol consumption can raise blood pressure and add extra stress to your kidneys."
    ])
    
    return recommendations

# Header
st.title("🏥 Health Prediction Application")
st.write("Use this application to predict diabetes, heart disease, or kidney disease based on health parameters.")
st.write(f"Current UTC Time: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs for the three models
tab1, tab2, tab3 = st.tabs(["Diabetes Prediction", "Heart Disease Prediction", "Kidney Disease Prediction"])

# Diabetes Prediction Tab
with tab1:
    st.header("Diabetes Prediction")
    st.write("Enter your health details to check for diabetes risk.")
    
    try:
        # Load diabetes model
        diabetes_model, diabetes_scaler, diabetes_features = load_diabetes_model()
        
        # Create form for input
        with st.form("diabetes_form"):
            # Create two columns for input fields
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
                glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            
            with col2:
                insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
                bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
                diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
                age = st.number_input("Age", min_value=21, max_value=100, value=30)

            submit_button = st.form_submit_button("Predict")
            
            if submit_button:
                # Create input array in the correct order matching the features
                input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, diabetes_pedigree, age]
                
                # Get prediction
                prediction, probability = predict_diabetes(input_data, diabetes_model, diabetes_scaler, diabetes_features)
                
                # Show prediction with formatted probability
                st.subheader("Prediction Result")
                
                # Calculate probability percentage for the predicted class
                prob_percent = probability[1] * 100 if prediction == 1 else probability[0] * 100
                
                if prediction == 1:
                    st.error(f"⚠️ Risk of Diabetes Detected ⚠️")
                    st.write(f"The model predicts that you **might have diabetes** with a probability of **{prob_percent:.1f}%**")
                else:
                    st.success(f"✅ No Diabetes Detected")
                    st.write(f"The model predicts that you **do not have diabetes** with a probability of **{prob_percent:.1f}%**")
                
                # Display personalized recommendations
                st.subheader("Health Recommendations")
                
                # Get recommendations based on input data
                recommendations = get_diabetes_recommendations(input_data, diabetes_features)
                
                # Display recommendations in an expander
                with st.expander("View Detailed Recommendations", expanded=True):
                    for rec in recommendations:
                        st.markdown(rec)
                    
                    st.write("\n")
                    st.markdown("""
                    **Remember:** These recommendations are general guidelines. For personalized advice, 
                    always consult with healthcare professionals.
                    """)
                
                # Add disclaimer
                st.info("⚠️ Disclaimer: This is only a prediction and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis.")
                
                # Display the input values for verification
                with st.expander("View Input Parameters", expanded=False):
                    st.subheader("Input Parameters")
                    input_df = pd.DataFrame([input_data], columns=diabetes_features)
                    st.write(input_df)
                
    except Exception as e:
        st.error(f"Error loading the diabetes model: {e}")
        st.warning("Make sure you have trained the model and saved it as 'diabetes_rf_model.pkl'")

# Heart Disease Prediction Tab
with tab2:
    st.header("Heart Disease Prediction")
    st.write("Enter your health details to check for heart disease risk.")
    
    try:
        # Load heart disease model
        heart_model, heart_scaler, heart_features = load_heart_disease_model()
        
        # Create form for input
        with st.form("heart_disease_form"):
            # Create three columns for input fields
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=20, max_value=100, value=50, key="heart_age")
                sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1], key="heart_sex")
                cp_options = [(0, "Typical Angina"), (1, "Atypical Angina"), 
                             (2, "Non-anginal Pain"), (3, "Asymptomatic")]
                cp = st.selectbox("Chest Pain Type", options=cp_options, format_func=lambda x: x[1])
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
                chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
            
            with col2:
                fbs_options = [(0, "No (≤ 120 mg/dl)"), (1, "Yes (> 120 mg/dl)")]
                fbs = st.selectbox("Fasting Blood Sugar", options=fbs_options, format_func=lambda x: x[1])
                
                restecg_options = [(0, "Normal"), (1, "ST-T Wave Abnormality"), (2, "Left Ventricular Hypertrophy")]
                restecg = st.selectbox("Resting ECG Results", options=restecg_options, format_func=lambda x: x[1])
                
                thalach = st.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=220, value=150)
                
                exang_options = [(0, "No"), (1, "Yes")]
                exang = st.selectbox("Exercise Induced Angina", options=exang_options, format_func=lambda x: x[1])
            
            with col3:
                oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
                
                slope_options = [(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")]
                slope = st.selectbox("Slope of Peak Exercise ST Segment", options=slope_options, format_func=lambda x: x[1])
                
                ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
                
                thal_options = [(1, "Normal"), (2, "Fixed Defect"), (3, "Reversible Defect"), (0, "Unknown")]
                thal = st.selectbox("Thalassemia", options=thal_options, format_func=lambda x: x[1])

            submit_button = st.form_submit_button("Predict")
            
            if submit_button:
                # Extract first element from tuples for the categorical variables
                sex_value = sex[0]
                cp_value = cp[0]
                fbs_value = fbs[0]
                restecg_value = restecg[0]
                exang_value = exang[0]
                slope_value = slope[0]
                thal_value = thal[0]
                
                # Create input array in the correct order matching the features
                input_data = [age, sex_value, cp_value, trestbps, chol, fbs_value, 
                              restecg_value, thalach, exang_value, oldpeak, 
                              slope_value, ca, thal_value]
                
                # Get prediction
                prediction, probability = predict_heart_disease(input_data, heart_model, heart_scaler, heart_features)
                
                # Show prediction with formatted probability
                st.subheader("Prediction Result")
                
                # Calculate probability percentage for the predicted class
                prob_percent = probability[1] * 100 if prediction == 1 else probability[0] * 100
                
                if prediction == 1:
                    st.error(f"⚠️ Risk of Heart Disease Detected ⚠️")
                    st.write(f"The model predicts that you **might have heart disease** with a probability of **{prob_percent:.1f}%**")
                else:
                    st.success(f"✅ No Heart Disease Detected")
                    st.write(f"The model predicts that you **do not have heart disease** with a probability of **{prob_percent:.1f}%**")
                
                # Display personalized recommendations
                st.subheader("Health Recommendations")
                
                # Get recommendations based on input data
                recommendations = get_heart_disease_recommendations(input_data, heart_features)
                
                # Display recommendations in an expander
                with st.expander("View Detailed Recommendations", expanded=True):
                    for rec in recommendations:
                        st.markdown(rec)
                    
                    st.write("\n")
                    st.markdown("""
                    **Remember:** These recommendations are general guidelines. For personalized advice, 
                    always consult with healthcare professionals.
                    """)
                
                # Add disclaimer
                st.info("⚠️ Disclaimer: This is only a prediction and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis.")
                
                # Display the input values for verification
                with st.expander("View Input Parameters", expanded=False):
                    st.subheader("Input Parameters")
                    input_df = pd.DataFrame([input_data], columns=heart_features)
                    st.write(input_df)
                
    except Exception as e:
        st.error(f"Error loading the heart disease model: {e}")
        st.warning("Make sure you have trained the model and saved it as 'heart_disease_rf_model.pkl'")

# Kidney Disease Prediction Tab
with tab3:
    st.header("Kidney Disease Prediction")
    st.write("Enter your health details to check for kidney disease risk.")
    
    try:
        # Load kidney disease model
        kidney_model, kidney_scaler, kidney_features = load_kidney_disease_model()
        
        # Create form for input
        with st.form("kidney_disease_form"):
            # Create three columns for input fields
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                age = st.number_input("Age", min_value=1, max_value=100, value=50, key="kidney_age")
                bp = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=180, value=80)
                sg = st.select_slider("Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025], value=1.020)
                al = st.slider("Albumin", min_value=0, max_value=5, value=0)
                su = st.slider("Sugar", min_value=0, max_value=5, value=0)
                bgr = st.number_input("Blood Glucose Random", min_value=70, max_value=500, value=120)
            
            with col2:
                bu = st.number_input("Blood Urea (mg/dL)", min_value=1.5, max_value=400.0, value=40.0)
                sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.4, max_value=77.0, value=1.0, step=0.1)
                sod = st.number_input("Sodium (mEq/L)", min_value=4.5, max_value=163.0, value=137.5)
                pot = st.number_input("Potassium (mEq/L)", min_value=2.5, max_value=47.0, value=4.5, step=0.1)
                hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.1, max_value=17.8, value=12.0, step=0.1)
                pcv = st.number_input("Packed Cell Volume", min_value=9, max_value=54, value=40)
            
            with col3:
                wc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=2600, max_value=26400, value=8000)
                rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=2.1, max_value=8.0, value=5.0, step=0.1)
                
                rbc_options = [(0, "Normal"), (1, "Abnormal")]
                rbc = st.selectbox("Red Blood Cells", options=rbc_options, format_func=lambda x: x[1])
                
                pc_options = [(0, "Normal"), (1, "Abnormal")]
                pc = st.selectbox("Pus Cell", options=pc_options, format_func=lambda x: x[1])
                
                pcc_options = [(0, "Not Present"), (1, "Present")]
                pcc = st.selectbox("Pus Cell Clumps", options=pcc_options, format_func=lambda x: x[1])
                
                ba_options = [(0, "Not Present"), (1, "Present")]
                ba = st.selectbox("Bacteria", options=ba_options, format_func=lambda x: x[1])
            
            with col4:
                htn_options = [(0, "No"), (1, "Yes")]
                htn = st.selectbox("Hypertension", options=htn_options, format_func=lambda x: x[1])
                
                dm_options = [(0, "No"), (1, "Yes")]
                dm = st.selectbox("Diabetes Mellitus", options=dm_options, format_func=lambda x: x[1])
                
                cad_options = [(0, "No"), (1, "Yes")]
                cad = st.selectbox("Coronary Artery Disease", options=cad_options, format_func=lambda x: x[1])
                
                appet_options = [(0, "Good"), (1, "Poor")]
                appet = st.selectbox("Appetite", options=appet_options, format_func=lambda x: x[1])
                
                pe_options = [(0, "No"), (1, "Yes")]
                pe = st.selectbox("Pedal Edema", options=pe_options, format_func=lambda x: x[1])
                
                ane_options = [(0, "No"), (1, "Yes")]
                ane = st.selectbox("Anemia", options=ane_options, format_func=lambda x: x[1])

            submit_button = st.form_submit_button("Predict")
            
            if submit_button:
                # Extract first element from tuples for the categorical variables
                rbc_value = rbc[0]
                pc_value = pc[0]
                pcc_value = pcc[0]
                ba_value = ba[0]
                htn_value = htn[0]
                dm_value = dm[0]
                cad_value = cad[0]
                appet_value = appet[0]
                pe_value = pe[0]
                ane_value = ane[0]
                
                # Create input array in the correct order matching the features
                input_data = [
                    age, bp, sg, al, su, rbc_value, pc_value, pcc_value, ba_value, bgr, 
                    bu, sc, sod, pot, hemo, pcv, wc, rc, htn_value, dm_value, cad_value, 
                    appet_value, pe_value, ane_value
                ]
                
                # Get prediction
                prediction, probability = predict_kidney_disease(input_data, kidney_model, kidney_scaler, kidney_features)
                
                # Show prediction with formatted probability
                st.subheader("Prediction Result")
                
                # Calculate probability percentage for the predicted class
                prob_percent = probability[1] * 100 if prediction == 1 else probability[0] * 100
                
                if prediction == 1:
                    st.error(f"⚠️ Risk of Chronic Kidney Disease Detected ⚠️")
                    st.write(f"The model predicts that you **might have chronic kidney disease** with a probability of **{prob_percent:.1f}%**")
                else:
                    st.success(f"✅ No Chronic Kidney Disease Detected")
                    st.write(f"The model predicts that you **do not have chronic kidney disease** with a probability of **{prob_percent:.1f}%**")
                
                # Display personalized recommendations
                st.subheader("Health Recommendations")
                
                # Get recommendations based on input data
                recommendations = get_kidney_disease_recommendations(input_data, kidney_features)
                
                # Display recommendations in an expander
                with st.expander("View Detailed Recommendations", expanded=True):
                    for rec in recommendations:
                        st.markdown(rec)
                    
                    st.write("\n")
                    st.markdown("""
                    **Remember:** These recommendations are general guidelines. For personalized advice, 
                    always consult with healthcare professionals.
                    """)
                
                # Add disclaimer
                st.info("⚠️ Disclaimer: This is only a prediction and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis.")
                
                # Display the input values for verification
                with st.expander("View Input Parameters", expanded=False):
                    st.subheader("Input Parameters")
                    input_df = pd.DataFrame([input_data], columns=kidney_features)
                    st.write(input_df)
                
    except Exception as e:
        st.error(f"Error loading the kidney disease model: {e}")
        st.warning("Make sure you have trained the model and saved it as 'kidney_disease_rf_model.pkl'")

# Add footer
st.markdown("---")
st.markdown("### About this app")
expander = st.expander("Learn more")
with expander:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Diabetes")
        st.write("""
        Diabetes is a chronic health condition that affects how your body turns food into energy. 
        If you have diabetes, your body either doesn't make enough insulin or can't use the insulin it makes as well as it should.
        
        **Risk factors include:**
        - Family history
        - Overweight or obesity
        - Physical inactivity
        - Age (45 or older)
        - High blood pressure
        - Abnormal cholesterol levels
        """)
    
    with col2:
        st.markdown("#### Heart Disease")
        st.write("""
        Heart disease refers to several types of heart conditions. The most common type is coronary artery disease, 
        which can cause heart attack.
        
        **Risk factors include:**
        - High blood pressure
        - High cholesterol
        - Smoking
        - Diabetes
        - Overweight or obesity
        - Physical inactivity
        - Unhealthy diet
        - Excessive alcohol use
        - Family history
        """)
    
    with col3:
        st.markdown("#### Kidney Disease")
        st.write("""
        Chronic Kidney Disease (CKD) means your kidneys are damaged and can't filter blood the way they should. 
        This damage can cause wastes to build up in your body and lead to other health problems.
        
        **Risk factors include:**
        - Diabetes
        - High blood pressure
        - Heart disease
        - Family history of kidney disease
        - Obesity
        - Smoking
        - Advanced age
        - Abnormal kidney structure
        - Frequent use of medications that can damage the kidneys
        """)

st.write("""
This application uses machine learning models to predict the risk of diabetes, heart disease, and kidney disease based on your health parameters.
The models were trained using Random Forest Classifier on standard datasets. Remember that these predictions are not a substitute for professional medical advice.
""")

st.markdown(f"© {datetime.datetime.now().year} Health Prediction App | Created by Student ID: 22951a3363")