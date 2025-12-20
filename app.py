#---------------------------------------------------------------------------------------------#
# --- Dependencies ---
#---------------------------------------------------------------------------------------------#

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import io
import streamlit as st
from datetime import datetime

#---------------------------------------------------------------------------------------------#
# --- Load trained models ---
#---------------------------------------------------------------------------------------------#

MODEL_PATH = "models"

diabetes_model = joblib.load(os.path.join(MODEL_PATH, "diabetes_model.pkl"))
ihd_model = joblib.load(os.path.join(MODEL_PATH, "ihd_model.pkl"))
stroke_model = joblib.load(os.path.join(MODEL_PATH, "stroke_model.pkl"))
covid_model = joblib.load(os.path.join(MODEL_PATH, "covid_model.pkl"))

#---------------------------------------------------------------------------------------------#
# --- Model accuracy ---
#---------------------------------------------------------------------------------------------#

model_accuracies = {
    "Diabetes": 0.72,
    "IHD": 0.95,
    "Stroke": 0.95,
    "COVID-19 Mortality": 0.95
}

#---------------------------------------------------------------------------------------------#
# --- Sidebar: Patient Information ---
#---------------------------------------------------------------------------------------------#

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 0, 100, 30, key="age")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"], key="gender")
gender_binary = 1 if gender == "Male" else 0

st.sidebar.divider()

#---------------------------------------------------------------------------------------------#
# --- Sidebar: General Health Metrics ---
#---------------------------------------------------------------------------------------------#

# --- BMI / Weight & Height ---
st.sidebar.markdown("### BMI / Weight & Height")

bmi_known = st.sidebar.checkbox("I know my BMI", value=False, key="bmi_known")

if bmi_known:
    # User enters BMI directly
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", 10.0, 70.0, 25.0, key="bmi")
    weight = None
    height = None
else:
    # User enters weight & height, BMI is calculated automatically
    weight = st.sidebar.number_input("Weight (kg)", 30.0, 200.0, 70.0, key="weight")
    height = st.sidebar.number_input("Height (cm)", 100.0, 220.0, 170.0, key="height")
    bmi = weight / ((height / 100) ** 2)

st.sidebar.caption(f"Calculated BMI: {bmi:.1f}")

# --- Blood Pressure ---
st.sidebar.markdown("### Blood Pressure")
bp_known = st.sidebar.checkbox("I know my Blood Pressure", value=False, key="bp_known")

if bp_known:
    systolic = st.sidebar.number_input("Systolic BP (mmHg)", 80, 250, 120, key="systolic")
    diastolic = st.sidebar.number_input("Diastolic BP (mmHg)", 50, 150, 80, key="diastolic")
else:
    st.sidebar.info("Default BP values will be used")
    systolic = 120
    diastolic = 80

# --- Heart Rate ---
st.sidebar.markdown("### Heart Rate")
hr_known = st.sidebar.checkbox("I know my Heart Rate", value=False, key="hr_known")

if hr_known:
    heart_rate = st.sidebar.number_input("Heart Rate (bpm)", 40, 200, 70, key="heart_rate")
else:
    st.sidebar.info("Default heart rate will be used")
    heart_rate = 70

# --- Smoking Status ---
st.sidebar.markdown("### Smoking Status")
smoking_status = st.sidebar.selectbox(
    "Select your smoking status",
    ["Non-smoker", "Light smoker", "Medium smoker", "Heavy smoker"],
    key="smoking_status"
)

# --- Alcohol Consumption ---
st.sidebar.markdown("### Alcohol Consumption")
alcohol_consumption = st.sidebar.selectbox(
    "Select your alcohol consumption",
    ["Sober", "Light drinker", "Medium drinker", "Heavy drinker"],
    key="alcohol_consumption"
)

# --- Physical Activity Level ---
st.sidebar.markdown("### Physical Activity Level")
physical_activity = st.sidebar.selectbox(
    "Select your activity level",
    ["Sedentary", "Light", "Moderate", "Vigorous"],
    key="physical_activity"
)

# Encode categorical inputs with keys matching the selectbox options
smoking_map = {
    "Non-smoker": 0,
    "Light smoker": 1,
    "Medium smoker": 2,
    "Heavy smoker": 3
}
alcohol_map = {
    "Sober": 0,
    "Light drinker": 1,
    "Medium drinker": 2,
    "Heavy drinker": 3
}
activity_map = {
    "Sedentary": 0,
    "Light": 1,
    "Moderate": 2,
    "Vigorous": 3
}

smoking_encoded = smoking_map[smoking_status]
alcohol_encoded = alcohol_map[alcohol_consumption]
activity_encoded = activity_map[physical_activity]

#---------------------------------------------------------------------------------------------#
# --- Main Page ---
#---------------------------------------------------------------------------------------------#

st.title("Interactive Health Risk Dashboard")

st.info(
    """
    **How to use this tool**
    1. Fill out your basic details in the sidebar on the left
    2. Select one or more diseases below
    3. Provide any additional information when prompted
    4. View your personalised risk assessment for the selected disease/s
    """
)

st.markdown("### Select conditions to assess:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    show_diabetes = st.checkbox("🩸 Diabetes")

with col2:
    show_ihd = st.checkbox("❤️ IHD")

with col3:
    show_stroke = st.checkbox("🧠 Stroke")

with col4:
    show_covid = st.checkbox("🦠 COVID-19")

# Initialize results dictionary **before** referencing it anywhere
results = {}
disease_inputs = {}

#---------------------------------------------------------------------------------------------#
# --- disease-specific ---
#---------------------------------------------------------------------------------------------#

# -------------------- Diabetes --------------------

if show_diabetes:
    st.subheader("Diabetes – Additional Information")

    # Pregnancies
    preg_na = st.checkbox("Number of pregnancies: N/A or Unknown")
    if preg_na:
        pregnancies = 0  # default
    else:
        pregnancies = st.number_input("Number of pregnancies", 0, 20, 0)

    # Average Glucose
    glucose_na = st.checkbox("Average Glucose Level: N/A or Unknown")
    if glucose_na:
        avg_glucose = 100  # default
    else:
        avg_glucose = st.number_input("Average Glucose Level", 50, 300, 100)

    # Skin Thickness
    skin_na = st.checkbox("Skin Thickness: N/A or Unknown")
    if skin_na:
        skin_thickness = 20  # default
    else:
        skin_thickness = st.number_input("Skin Thickness (mm)", 1, 99, 20)

    # Insulin
    insulin_na = st.checkbox("Insulin Level: N/A or Unknown")
    if insulin_na:
        insulin = 80  # default
    else:
        insulin = st.number_input("Insulin Level", 1, 900, 80)

    # Diabetes Pedigree Function
    dpf_na = st.checkbox("Diabetes Pedigree Function: N/A or Unknown")
    if dpf_na:
        dpf = 0.5  # default
    else:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

    disease_inputs["diabetes"] = {
        "pregnancies": pregnancies,
        "avg_glucose": avg_glucose,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "dpf": dpf
    }

# -------------------- IHD --------------------

if show_ihd:
    st.subheader("IHD – Additional Information")

    # Chest pain type
    chest_na = st.checkbox("Chest pain type: N/A or Unknown")
    if chest_na:
        chest_pain = 0  # None
    else:
        chest_pain = st.selectbox(
            "Chest pain type",
            ["None", "Atypical", "Typical"]
        )
        chest_pain_map = {"None":0,"Atypical":1,"Typical":2}
        chest_pain = chest_pain_map[chest_pain]

    # Exercise-induced angina
    exercise_na = st.checkbox("Exercise-induced angina(chest pain): N/A or Unknown")
    if exercise_na:
        exercise_angina = 0
    else:
        exercise_angina = st.checkbox("Chest pain during exercise")
    
    disease_inputs["ihd"] = {
        "chest_pain": chest_pain,
        "exercise_angina": int(exercise_angina)
    }

# -------------------- Stroke --------------------

if show_stroke:
    st.subheader("Stroke – Additional Information")

    # Previous TIA / mini-stroke
    tia_na = st.checkbox("Previous TIA / mini-stroke: N/A or Unknown")
    if tia_na:
        previous_tia = 0
    else:
        previous_tia = st.checkbox("Previous TIA / mini-stroke")


    disease_inputs["stroke"] = {
        "previous_tia": int(previous_tia)
    }

# -------------------- COVID-19 --------------------

if show_covid:
    st.subheader("COVID-19 – Additional Information")

    # Initialize covid dictionary in disease_inputs
    if "covid" not in disease_inputs:
        disease_inputs["covid"] = {}

    # Vaccinated
    vacc_na = st.checkbox("Vaccination status: N/A or Unknown")
    if vacc_na:
        disease_inputs["covid"]["vaccinated"] = 0
    else:
        disease_inputs["covid"]["vaccinated"] = int(st.checkbox("Vaccinated"))

    # Chronic conditions
    disease_inputs["covid"]["diabetes"] = int(st.checkbox("History of diabetes"))  
    disease_inputs["covid"]["hypertension"] = int(st.checkbox("History of hypertension"))  
    disease_inputs["covid"]["heart_disease"] = int(st.checkbox("History of heart disease")) 

#---------------------------------------------------------------------------------------------#
# --- Run Predictions ---
#---------------------------------------------------------------------------------------------#

st.divider()
run_prediction = st.button("Calculate Risk")

if run_prediction:
    results = {}  # Initialize results
    # -------------------- Diabetes --------------------
    if show_diabetes:
        X_diabetes = pd.DataFrame([[
            disease_inputs["diabetes"].get("pregnancies", 0),          # Pregnancies
            disease_inputs["diabetes"].get("avg_glucose", 100),         # Glucose
            (systolic + diastolic)/2,                                    # BloodPressure
            disease_inputs["diabetes"].get("skin_thickness", 20),        # SkinThickness
            disease_inputs["diabetes"].get("insulin", 80),              # Insulin
            bmi,                                                         # BMI from sidebar
            disease_inputs["diabetes"].get("dpf", 0.5),                 # DiabetesPedigreeFunction
            age                                                          # Age from sidebar
        ]], columns=[
            'Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age'
        ])

        diabetes_pred = diabetes_model.predict_proba(X_diabetes)[:,1][0]
        results["Diabetes"] = diabetes_pred

    # -------------------- IHD --------------------
    if show_ihd:
        X_ihd = pd.DataFrame([[
            age,                                                        # age
            gender_binary,                                              # sex
            disease_inputs["ihd"].get("chest_pain", 0),                # chest_pain_type
            systolic,                                                   # resting_bp_s
            200,                                                        # cholesterol (default)
            0,                                                          # fasting_blood_sugar (default)
            0,                                                          # resting_ecg (default)
            150,                                                        # max_heart_rate (default)
            disease_inputs["ihd"].get("exercise_angina", 0),           # exercise_angina
            1.0,                                                        # oldpeak (default)
            1                                                           # st_slope (default)
        ]], columns=[
            'age','sex','chest_pain_type','resting_bp_s','cholesterol',
            'fasting_blood_sugar','resting_ecg','max_heart_rate',
            'exercise_angina','oldpeak','st_slope'
        ])

        ihd_pred = ihd_model.predict_proba(X_ihd)[:,1][0]
        results["IHD"] = ihd_pred

    # -------------------- Stroke --------------------
    if show_stroke:
        X_stroke = pd.DataFrame([[
            0,                                                          # id (default)
            gender_binary,                                              # gender
            age,                                                        # age
            0,                                                          # hypertension (default)
            0,                                                          # heart_disease (default)
            1,                                                          # ever_married (default)
            0,                                                          # work_type (default)
            1,                                                          # Residence_type (default)
            disease_inputs["stroke"].get("avg_glucose_level", 100),    # avg_glucose_level
            bmi,                                                        # bmi
            smoking_encoded                                              # smoking_status from sidebar
        ]], columns=[
            'id','gender','age','hypertension','heart_disease',
            'ever_married','work_type','Residence_type',
            'avg_glucose_level','bmi','smoking_status'
        ])

        stroke_pred = stroke_model.predict_proba(X_stroke)[:,1][0]
        results["Stroke"] = stroke_pred

    # -------------------- COVID-19 --------------------
    if show_covid:
        X_covid = pd.DataFrame([[
            age,                                                        # age
            gender_binary,                                              # gender
            int(disease_inputs["covid"].get("vaccinated", 0)),         # vaccination_status
            0, 0, 0, 0, 0, 0,                                          # symptoms defaults
            int(disease_inputs["covid"].get("diabetes", 0)),           # diabetes
            int(disease_inputs["covid"].get("hypertension", 0)),       # hypertension
            int(disease_inputs["covid"].get("heart_disease", 0)),      # heart_disease
            0, 0, 0, 0                                                 # other defaults
        ]], columns=[
            'age','gender','vaccination_status','fever','cough','fatigue',
            'shortness_of_breath','loss_of_smell','headache','diabetes',
            'hypertension','heart_disease','asthma','cancer',
            'hospitalized','icu_admission'
        ])

        covid_pred = covid_model.predict_proba(X_covid)[:,1][0]
        results["COVID-19"] = covid_pred

#---------------------------------------------------------------------------------------------#
# --- Disclaimer / Info --- 
#---------------------------------------------------------------------------------------------#

st.info(
    """
    ⚠️ **Important Disclaimer**  
    This is an **unofficial health risk assessment tool** built for educational and demonstration purposes only.

    Predictions are based on pre-trained models and **should not be considered medical advice**.  
    Small changes in input values may result in non-linear changes in predicted risk due to model behavior.

    For a professional evaluation of your health, please consult a **licensed medical professional**.
    """
)

#---------------------------------------------------------------------------------------------#
# --- Results ---
#---------------------------------------------------------------------------------------------#

if results:
        st.subheader("Predicted Risk Probabilities")
        for disease, risk in results.items():
            st.metric(f"{disease} Risk", f"{risk*100:.1f}%")


#---------------------------------------------------------------------------------------------#
# --- Interpret Risk ---
#---------------------------------------------------------------------------------------------#

# Disease descriptions
disease_descriptions = {
    "IHD": "Ischemic Heart Disease: Reduced blood supply to the heart, can lead to chest pain or heart attack.",
    "Diabetes": "Diabetes: High blood sugar levels due to insulin issues.",
    "Stroke": "Stroke: Interruption of blood flow to the brain causing potential long-term damage.",
    "COVID-19": "COVID-19: Viral infection that can affect respiratory and other systems."
}

def interpret_risk(risk):
    """
    Returns a risk category and general advice based on risk percentage
    """
    if risk < 0.1:
        category = "Low"
        advice = "Your current profile indicates a relatively low risk. Maintain healthy habits."
    elif risk < 0.2:
        category = "Moderate"
        advice = "There may be some risk factors present. Consider lifestyle adjustments and consulting a medical professional if concerned."
    elif risk < 0.4:
        category = "High"
        advice = "You have significant risk factors. Consider consulting a healthcare professional and reviewing lifestyle choices."
    else:
        category = "Very High"
        advice = "Your risk is elevated. Immediate medical consultation is recommended."
    return category, advice

if results:
    st.subheader("Predicted Risk Probabilities")
    st.info("⚠️ **Disclaimer:** This is an unofficial project for educational purposes only. Please consult a licensed medical professional for official advice.")
    
    for disease, risk in results.items():
        category, advice = interpret_risk(risk)
        st.metric(f"{disease} Risk", f"{risk*100:.1f}%", delta=category)
        st.write(advice)

    # Optional: bar chart for visual comparison
    risk_df = pd.DataFrame(list(results.items()), columns=['Disease','Risk'])
    st.bar_chart(risk_df.set_index('Disease'))

#---------------------------------------------------------------------------------------------#
# --- PDF Export ---
#---------------------------------------------------------------------------------------------#

# Disease descriptions
disease_descriptions = {
    "IHD": "Ischemic Heart Disease: Reduced blood supply to the heart, can lead to chest pain or heart attack.",
    "Diabetes": "Diabetes: High blood sugar levels due to insulin issues.",
    "Stroke": "Stroke: Interruption of blood flow to the brain causing potential long-term damage.",
    "COVID-19": "COVID-19: Viral infection that can affect respiratory and other systems."
}

def save_pdf(results_dict, patient_info):
    """
    Generate a detailed Patient Health Risk Report PDF.

    Parameters:
    - results_dict: dict, disease -> risk probability (0-1)
    - patient_info: dict, e.g. {"age":30, "gender":"Male", "bmi":25.0, "smoking":0, "alcohol":1, "activity":2}
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Patient Health Risk Report", ln=True, align="C")
    pdf.ln(5)
    
    # Patient info
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 8, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(200, 8, txt=f"Age: {patient_info.get('age', 'N/A')} | Gender: {patient_info.get('gender', 'N/A')} | BMI: {patient_info.get('bmi', 'N/A'):.1f}", ln=True)
    pdf.cell(200, 8, txt=f"Smoking: {patient_info.get('smoking', 'N/A')} | Alcohol: {patient_info.get('alcohol', 'N/A')} | Physical Activity: {patient_info.get('activity', 'N/A')}", ln=True)
    pdf.ln(5)

    # Define risk categories and tips
    risk_categories = [
        (0.0, 0.1, "Low", "Maintain your healthy lifestyle."),
        (0.1, 0.2, "Moderate", "Consider improving diet, exercise, and regular check-ups."),
        (0.2, 1.0, "High", "Seek advice from a healthcare professional.")
    ]

    # Disease sections
    for disease, risk in results_dict.items():
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 8, txt=f"{disease} Risk: {risk*100:.1f}%", ln=True)
        
        # Determine risk category
        category_text = "Unknown"
        tip_text = ""
        for lower, upper, category, tip in risk_categories:
            if lower <= risk < upper:
                category_text = category
                tip_text = tip
                break
        
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 6, txt=f"Risk Category: {category_text}", ln=True)
        pdf.multi_cell(0, 6, txt=f"Tip: {tip_text}")
        
        # Add disease description
        description_text = disease_descriptions.get(disease, "No description available.")
        pdf.multi_cell(0, 6, txt=f"Description: {description_text}")
        pdf.ln(2)

    # Disclaimer
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 5, txt="Disclaimer: This report is for informational purposes only and is not a medical diagnosis. Please consult a licensed healthcare professional for personalized advice.")
    
    # Output PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes