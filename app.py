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
from datetime import datetime
import gspread
from google.oauth2 import service_account
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
from sklearn.impute import SimpleImputer

#---------------------------------------------------------------------------------------------#
# --- Load trained models ---
#---------------------------------------------------------------------------------------------#

MODEL_PATH = "models"

diabetes_model = joblib.load(os.path.join(MODEL_PATH, "diabetes_model.pkl"))
ihd_model = joblib.load(os.path.join(MODEL_PATH, "ihd_model.pkl"))
stroke_model = joblib.load(os.path.join(MODEL_PATH, "stroke_model.pkl"))
covid_model = joblib.load(os.path.join(MODEL_PATH, "covid_model.pkl"))

with open("models/diabetes_model_feature_means.json") as f:
    diabetes_feature_means = json.load(f)

with open("models/ihd_model_feature_means.json") as f:
    ihd_feature_means = json.load(f)

with open("models/stroke_model_feature_means.json") as f:
    stroke_feature_means = json.load(f)

with open("models/covid_model_feature_means.json") as f:
    covid_feature_means = json.load(f)

# -----------------------------------------------------------------------------------
# --- Function: Safe model prediction with NaN handling ---
# -----------------------------------------------------------------------------------

def safe_predict(model, input_df, feature_means):
    """
    Fill NaNs with training feature means and ensure the DataFrame matches model features.
    """
    # Reindex to model's expected columns
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=np.nan)

    # Fill missing values with training means
    for col in model.feature_names_in_:
        input_df[col] = input_df[col].fillna(feature_means[col])

    # Predict
    return model.predict_proba(input_df)[:, 1][0]

#---------------------------------------------------------------------------------------------#
# --- Model accuracy ---
#---------------------------------------------------------------------------------------------#

model_accuracies = {
    "Diabetes": 0.72,
    "IHD": 0.95,
    "Stroke": 0.95,
    "COVID": 0.95
}

#---------------------------------------------------------------------------------------------#
# --- Sidebar: Patient Information ---
#---------------------------------------------------------------------------------------------#

st.sidebar.header("Patient Information")

# Age input
age_na = st.sidebar.checkbox("Age: N/A or Unknown", value=False, key="age_na")
age = np.nan if age_na else st.sidebar.slider("Age", 0, 100, 30, key="age")
age_unknown = int(age_na)

# Gender input
gender = st.sidebar.selectbox("Gender", ["Female", "Male"], key="gender")
gender_binary = 1 if gender == "Male" else 0

st.sidebar.divider()

# --- BMI / Weight & Height ---
st.sidebar.markdown("### BMI / Weight & Height")

# Check if BMI is known or not
bmi_na = st.sidebar.checkbox("I know my BMI", value=False, key="bmi_na")

# If the user knows their BMI, prompt them to input it directly
if bmi_na:
    bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0, key="bmi")
    bmi_unknown = 0  # BMI is provided
    weight = None  # Disabling weight input
    height = None  # Disabling height input
else:
    # Otherwise, allow them to input weight and height and calculate BMI
    weight = st.sidebar.number_input("Weight (kg)", 30.0, 200.0, 70.0, key="weight")
    height = st.sidebar.number_input("Height (cm)", 100.0, 220.0, 170.0, key="height")
    bmi = weight / ((height / 100) ** 2) if weight and height else np.nan
    bmi_unknown = 1 if np.isnan(bmi) else 0  # BMI is unknown if either weight or height is missing

st.sidebar.caption(f"Calculated BMI: {bmi if bmi is not np.nan else 'N/A'}")

# --- Blood Pressure ---
st.sidebar.markdown("### Blood Pressure")
bp_na = st.sidebar.checkbox("I know my Blood Pressure", value=False, key="bp_na")

# If the user knows their blood pressure, allow them to input systolic and diastolic
if bp_na:
    systolic = st.sidebar.number_input("Systolic BP (mmHg)", 80, 250, 120, key="systolic")
    diastolic = st.sidebar.number_input("Diastolic BP (mmHg)", 50, 150, 80, key="diastolic")
    systolic_unknown = 0  # Systolic is provided
    diastolic_unknown = 0  # Diastolic is provided
else:
    # If not, set both systolic and diastolic to NaN
    systolic = np.nan
    diastolic = np.nan
    systolic_unknown = 1  # Systolic is unknown
    diastolic_unknown = 1  # Diastolic is unknown

# --- Heart Rate ---
st.sidebar.markdown("### Heart Rate")
hr_na = st.sidebar.checkbox("I know my Heart Rate", value=False, key="hr_na")

# If the user knows their heart rate, allow them to input it
if hr_na:
    heart_rate = st.sidebar.number_input("Heart Rate (bpm)", 40, 200, 70, key="heart_rate")
    heart_rate_unknown = 0  # Heart rate is provided
else:
    # If not, set heart rate to NaN
    heart_rate = np.nan
    heart_rate_unknown = 1  # Heart rate is unknown

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
# --- disease-specific (with missing value indicators and np.nan for unknowns) ---
#---------------------------------------------------------------------------------------------#

# -------------------- Diabetes --------------------
if show_diabetes:
    st.subheader("Diabetes – Additional Information")

    # Pregnancies
    preg_na = st.checkbox("Number of pregnancies: N/A or Unknown")
    pregnancies = np.nan if preg_na else st.number_input("Number of pregnancies", 0, 20, 0)
    pregnancies_unknown = int(preg_na)

    # Average Glucose
    glucose_na = st.checkbox("Average Glucose Level: N/A or Unknown")
    avg_glucose = np.nan if glucose_na else st.number_input("Average Glucose Level", 50, 300, 100)
    avg_glucose_unknown = int(glucose_na)

    # Skin Thickness
    skin_na = st.checkbox("Skin Thickness: N/A or Unknown")
    skin_thickness = np.nan if skin_na else st.number_input("Skin Thickness (mm)", 1, 99, 20)
    skin_thickness_unknown = int(skin_na)

    # Insulin
    insulin_na = st.checkbox("Insulin Level: N/A or Unknown")
    insulin = np.nan if insulin_na else st.number_input("Insulin Level", 1, 900, 80)
    insulin_unknown = int(insulin_na)

    # Diabetes Pedigree Function
    dpf_na = st.checkbox("Diabetes Pedigree Function: N/A or Unknown")
    dpf = np.nan if dpf_na else st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    dpf_unknown = int(dpf_na)

    disease_inputs["diabetes"] = {
        "pregnancies": pregnancies,
        "pregnancies_unknown": pregnancies_unknown,
        "avg_glucose": avg_glucose,
        "avg_glucose_unknown": avg_glucose_unknown,
        "skin_thickness": skin_thickness,
        "skin_thickness_unknown": skin_thickness_unknown,
        "insulin": insulin,
        "insulin_unknown": insulin_unknown,
        "dpf": dpf,
        "dpf_unknown": dpf_unknown
    }

# -------------------- IHD --------------------
if show_ihd:
    st.subheader("IHD – Additional Information")

    # Chest pain type
    chest_na = st.checkbox("Chest pain type: N/A or Unknown")
    if chest_na:
        chest_pain = np.nan
    else:
        chest_pain = st.selectbox("Chest pain type", ["None", "Atypical", "Typical"])
        chest_pain_map = {"None": 0, "Atypical": 1, "Typical": 2}
        chest_pain = chest_pain_map[chest_pain]
    chest_pain_unknown = int(chest_na)

    # Exercise-induced angina
    exercise_na = st.checkbox("Exercise-induced angina(chest pain): N/A or Unknown")
    exercise_angina = np.nan if exercise_na else int(st.checkbox("Chest pain during exercise"))
    exercise_angina_unknown = int(exercise_na)

    # Cholesterol Level
    cholesterol_known = st.checkbox("I know my Cholesterol level", value=False, key="cholesterol_known")
    cholesterol = np.nan if not cholesterol_known else st.number_input("Cholesterol Level (mg/dL)", 100, 400, 200, key="cholesterol")
    cholesterol_unknown = int(not cholesterol_known)

    disease_inputs["ihd"] = {
        "chest_pain": chest_pain,
        "chest_pain_unknown": chest_pain_unknown,
        "exercise_angina": exercise_angina,
        "exercise_angina_unknown": exercise_angina_unknown,
        "cholesterol": cholesterol,
        "cholesterol_unknown": cholesterol_unknown
    }

# -------------------- Stroke --------------------
if show_stroke:
    st.subheader("Stroke – Additional Information")

    # Previous TIA / mini-stroke
    tia_na = st.checkbox("Previous TIA / mini-stroke: N/A or Unknown")
    previous_tia = np.nan if tia_na else int(st.checkbox("Previous TIA / mini-stroke"))
    previous_tia_unknown = int(tia_na)

    disease_inputs["stroke"] = {
        "previous_tia": previous_tia,
        "previous_tia_unknown": previous_tia_unknown
    }

    disease_inputs["stroke"].update({
        "systolic": systolic,
        "systolic_unknown": systolic_unknown,
        "diastolic": diastolic,
        "diastolic_unknown": diastolic_unknown,
        "bmi": bmi,
        "bmi_unknown": bmi_unknown,
        "smoking": smoking_encoded
    })

# -------------------- COVID-19 --------------------
if show_covid:
    st.subheader("COVID-19 – Additional Information")

    if "covid" not in disease_inputs:
        disease_inputs["covid"] = {}

    # Symptom inputs
    fever = int(st.checkbox("Fever"))
    dry_cough = int(st.checkbox("Dry cough"))
    sore_throat = int(st.checkbox("Sore throat"))
    fatigue = int(st.checkbox("Fatigue"))
    headache = int(st.checkbox("Headache"))
    shortness_of_breath = int(st.checkbox("Shortness of breath"))
    loss_of_smell = int(st.checkbox("Loss of smell"))
    loss_of_taste = int(st.checkbox("Loss of taste"))
    chest_pain = int(st.checkbox("Chest pain"))

    # Additional clinical info
    oxygen_level = st.number_input("Oxygen level (%)", 70, 100, 98)
    body_temperature = st.number_input("Body temperature (°C)", 35.0, 42.0, 36.8)
    comorbidity = int(st.checkbox("Comorbidity present"))
    travel_history = int(st.checkbox("Recent travel history"))
    contact_with_patient = int(st.checkbox("Contact with COVID-positive patient"))
    covid_result = st.selectbox("Previous COVID result", ["Unknown", "Negative", "Positive"])

    # Assign to disease_inputs
    disease_inputs["covid"] = {
        "age": age,
        "gender": gender,
        "fever": fever,
        "dry_cough": dry_cough,
        "sore_throat": sore_throat,
        "fatigue": fatigue,
        "headache": headache,
        "shortness_of_breath": shortness_of_breath,
        "loss_of_smell": loss_of_smell,
        "loss_of_taste": loss_of_taste,
        "oxygen_level": oxygen_level,
        "body_temperature": body_temperature,
        "comorbidity": comorbidity,
        "travel_history": travel_history,
        "contact_with_patient": contact_with_patient,
        "chest_pain": chest_pain,
        "covid_result": covid_result
    }

#---------------------------------------------------------------------------------------------#
# --- Run Predictions ---
#---------------------------------------------------------------------------------------------#

st.divider()
run_prediction = st.button("Calculate Risk")

if run_prediction:
    results = {}  # Initialize results

    # -------------------- Diabetes --------------------
    if show_diabetes:
        # Compute BP mean if both known
        bp_mean = (systolic + diastolic) / 2 if not systolic_unknown and not diastolic_unknown else np.nan

        # Build full feature dict with NaNs for missing columns
        diabetes_features = {col: np.nan for col in diabetes_model.feature_names_in_}
        diabetes_features.update({
            "Pregnancies": disease_inputs["diabetes"].get("pregnancies", np.nan),
            "Glucose": disease_inputs["diabetes"].get("avg_glucose", np.nan),
            "BloodPressure": bp_mean,
            "SkinThickness": disease_inputs["diabetes"].get("skin_thickness", np.nan),
            "Insulin": disease_inputs["diabetes"].get("insulin", np.nan),
            "BMI": bmi if not bmi_unknown else np.nan,
            "DiabetesPedigreeFunction": disease_inputs["diabetes"].get("dpf", np.nan),
            "Age": age if not age_unknown else np.nan
        })

        # Convert to DataFrame with model feature names to avoid mismatch
        X_diabetes = pd.DataFrame([diabetes_features], columns=diabetes_model.feature_names_in_)
        results["Diabetes"] = safe_predict(diabetes_model, X_diabetes, diabetes_feature_means)

    # -------------------- IHD --------------------
    if show_ihd:
        ihd_features = {col: np.nan for col in ihd_model.feature_names_in_}
        ihd_features.update({
            "age": age if not age_unknown else np.nan,
            "sex": gender_binary,
            "chest_pain_type": disease_inputs["ihd"].get("chest_pain", np.nan),
            "resting_bp_s": systolic if not systolic_unknown else np.nan,
            "cholesterol": disease_inputs["ihd"].get("cholesterol", np.nan),
            "exercise_angina": disease_inputs["ihd"].get("exercise_angina", np.nan)
        })

        X_ihd = pd.DataFrame([ihd_features], columns=ihd_model.feature_names_in_)
        results["IHD"] = safe_predict(ihd_model, X_ihd, ihd_feature_means)

    # -------------------- Stroke --------------------
    if show_stroke:
        stroke_features = {col: np.nan for col in stroke_model.feature_names_in_}
        stroke_features.update({
            "id": 0,
            "gender": gender_binary,
            "age": age if not age_unknown else np.nan,
            "bmi": bmi if not bmi_unknown else np.nan,
            "smoking_status": smoking_encoded
        })

        X_stroke = pd.DataFrame([stroke_features], columns=stroke_model.feature_names_in_)
        results["Stroke"] = safe_predict(stroke_model, X_stroke, stroke_feature_means)

    # -------------------- COVID-19 --------------------
    if show_covid:
    # Initialize features dict with NaNs
        covid_features = {col: np.nan for col in covid_model.feature_names_in_}

    # Update with actual patient inputs
    covid_features.update({
        "age": age if not age_na else np.nan,
        "gender": gender_binary,
        "fever": disease_inputs["covid"].get("fever", np.nan),
        "dry_cough": disease_inputs["covid"].get("dry_cough", np.nan),
        "sore_throat": disease_inputs["covid"].get("sore_throat", np.nan),
        "fatigue": disease_inputs["covid"].get("fatigue", np.nan),
        "headache": disease_inputs["covid"].get("headache", np.nan),
        "shortness_of_breath": disease_inputs["covid"].get("shortness_of_breath", np.nan),
        "loss_of_smell": disease_inputs["covid"].get("loss_of_smell", np.nan),
        "loss_of_taste": disease_inputs["covid"].get("loss_of_taste", np.nan),
        "oxygen_level": disease_inputs["covid"].get("oxygen_level", np.nan),
        "body_temperature": disease_inputs["covid"].get("body_temperature", np.nan),
        "comorbidity": disease_inputs["covid"].get("comorbidity", np.nan),
        "travel_history": disease_inputs["covid"].get("travel_history", np.nan),
        "contact_with_patient": disease_inputs["covid"].get("contact_with_patient", np.nan),
        "chest_pain": disease_inputs["covid"].get("chest_pain", np.nan)
    })

    # Convert to DataFrame
    X_covid = pd.DataFrame([covid_features], columns=covid_model.feature_names_in_)

    # Predict using safe_predict (fills NaNs with feature means)
    results["COVID-19"] = safe_predict(covid_model, X_covid, covid_feature_means)

#---------------------------------------------------------------------------------------------#
# --- Disclaimer / Info --- 
#---------------------------------------------------------------------------------------------#

st.info(
    """
    ⚠️ **Important Disclaimer**  
    This is an **unofficial health risk assessment tool** built for educational and demonstration purposes only.

    For any missing or unknown inputs, the models use average values from the training data, which may reduce the accuracy or reliability of the predictions.

    For a professional evaluation of your health, please consult a **licensed medical professional**.
    """
)

#---------------------------------------------------------------------------------------------#
# --- Results ---
#---------------------------------------------------------------------------------------------#

if results:
        st.subheader("Predicted Risk Probabilities")
        for disease, risk in results.items():
            st.write(f"**{disease}:** {risk:.2%}")


# ---------------------------------------------------------------------------------------------#
# --- Interpret Risk & Top Factors --- 
# ---------------------------------------------------------------------------------------------#

def get_top_factors(disease, patient_data, age_val=None, gender_val=None, bmi_val=None,
                    systolic_val=None, diastolic_val=None, smoking_val=None, activity_val=None):
    """
    Returns top 3 patient-centered factors that contribute to disease risk.
    Only includes factors that are relevant for the given disease.
    """
    factors = []

    # --- Diabetes ---
    if disease == "Diabetes":
        if bmi_val and bmi_val > 25:
            factors.append("High BMI")
        if not patient_data["diabetes"].get("dpf_unknown", 0) and patient_data["diabetes"].get("dpf", 0) > 1.0:
            factors.append("Family history of diabetes")
        if not patient_data["diabetes"].get("avg_glucose_unknown", 0) and patient_data["diabetes"].get("avg_glucose", 0) > 140:
            factors.append("High average glucose")
        if not patient_data["diabetes"].get("skin_thickness_unknown", 0) and patient_data["diabetes"].get("skin_thickness", 0) > 30:
            factors.append("High skin thickness")
        if smoking_val is not None and smoking_val != 0:  # Skip "Non-smoker" (value 0)
            smoking_status = next((key for key, value in smoking_map.items() if value == smoking_val), "Unknown")
            factors.append(f"{smoking_status}")


    # --- IHD ---
    elif disease == "IHD":
        if not patient_data["ihd"].get("cholesterol_unknown", 0) and patient_data["ihd"].get("cholesterol", 0) > 240:
            factors.append("High cholesterol")
        if systolic_val and systolic_val > 140:
            factors.append("High systolic BP")
        if diastolic_val and diastolic_val > 90:
            factors.append("High diastolic BP")
        if smoking_val is not None and smoking_val != 0:  # Skip "Non-smoker" (value 0)
            smoking_status = next((key for key, value in smoking_map.items() if value == smoking_val), "Unknown")
            factors.append(f"{smoking_status}")
        if activity_val is not None and activity_val < 2:  # assuming 0=low,1=moderate,2=high
            factors.append("Low physical activity")
        if not patient_data["ihd"].get("exercise_angina_unknown", 0) and patient_data["ihd"].get("exercise_angina", 0) > 0:
            factors.append("Exercise-induced angina")

    # --- Stroke ---
    elif disease == "Stroke":
        if systolic_val and systolic_val > 130:
            factors.append("High systolic BP")
        if diastolic_val and diastolic_val > 80:
            factors.append("High diastolic BP")
        if bmi_val and bmi_val > 30:
            factors.append("High BMI")
        if not patient_data["stroke"].get("previous_tia_unknown", 0) and patient_data["stroke"].get("previous_tia", 0) > 0:
            factors.append("Previous TIA / mini-stroke")
        if smoking_val is not None and smoking_val != 0:  # Skip "Non-smoker" (value 0)
            smoking_status = next((key for key, value in smoking_map.items() if value == smoking_val), "Unknown")
            factors.append(f"{smoking_status}")

    # --- COVID-19 ---
    elif disease == "COVID-19":
        covid_data = patient_data.get("covid", {})

        # Symptoms
        if covid_data.get("fever", 0):
            factors.append("Fever")
        if covid_data.get("dry_cough", 0):
            factors.append("Dry cough")
        if covid_data.get("sore_throat", 0):
            factors.append("Sore throat")
        if covid_data.get("fatigue", 0):
            factors.append("Fatigue")
        if covid_data.get("headache", 0):
            factors.append("Headache")
        if covid_data.get("shortness_of_breath", 0):
            factors.append("Shortness of breath")
        if covid_data.get("loss_of_smell", 0):
            factors.append("Loss of smell")
        if covid_data.get("loss_of_taste", 0):
            factors.append("Loss of taste")
        if covid_data.get("chest_pain", 0):
            factors.append("Chest pain")

        # Comorbidities
        if covid_data.get("comorbidity", 0):
            factors.append("Comorbidity present")

        # Age-based risk
        if age_val and age_val > 60:
            factors.append("Advanced age")

    # Return **top 3 factors** or default message if empty
    return factors[:3] if factors else ["No significant patient-centered factors identified."]

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
# --- PDF Generation and Export ---
#---------------------------------------------------------------------------------------------#

patient_info = {
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "systolic": systolic,
    "diastolic": diastolic,
    "smoking": smoking_encoded,
    "activity": activity_encoded,
    "diabetes": disease_inputs.get("diabetes", {}),
    "ihd": disease_inputs.get("ihd", {}),
    "stroke": disease_inputs.get("stroke", {}),
    "covid": disease_inputs.get("covid", {})
}

# Disease descriptions
disease_descriptions = {
    "IHD": "Ischemic Heart Disease: Reduced blood supply to the heart, can lead to chest pain or heart attack.",
    "Diabetes": "Diabetes: High blood sugar levels due to insulin issues.",
    "Stroke": "Stroke: Interruption of blood flow to the brain causing potential long-term damage.",
    "COVID-19": "COVID-19: Viral infection that can affect respiratory and other systems."
}

def normalize_disease_name(name: str) -> str:
    """
    Removes ' Risk' suffix or trailing whitespace to match disease_descriptions keys.
    """
    return name.replace(" Risk", "").strip()

def save_pdf(results_dict, patient_info, top_factors_dict):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Patient Health Risk Report", ln=True, align="C")
    pdf.ln(5)

    # Patient info
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%d-%m-%y')}", ln=True)
    pdf.cell(0, 8, f"Age: {patient_info.get('age', 'N/A')} | Gender: {patient_info.get('gender', 'N/A')} | BMI: {bmi if bmi is not None and not np.isnan(bmi) else 'N/A'}", ln=True)
    pdf.cell(0, 8, f"Smoking: {patient_info.get('smoking', 'N/A')} | Alcohol: {patient_info.get('alcohol', 'N/A')} | Physical Activity: {patient_info.get('activity', 'N/A')}", ln=True)
    # Add risk of disease
    risk_of = []

    # Check if any disease is flagged as moderate or higher risk
    for disease, risk in results.items():
        category, _ = interpret_risk(risk)  # Get the category based on risk
        if category in ["Moderate", "High", "Very High"]:
            risk_of.append(disease)

    if risk_of:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Risk of: {', '.join(risk_of)}", ln=True)
    else:
        pdf.cell(0, 8, "No significant risks identified", ln=True)

    pdf.set_font("Arial", '', 12)    
    pdf.cell(0, 8, f"All inputs are patient-reported unless otherwise stated.", ln=True)

    # Risk categories with tips
    risk_categories = [
        (0.0, 0.1, "Low", "Maintain your healthy lifestyle."),
        (0.1, 0.2, "Moderate", "Consider improving diet, exercise, and regular check-ups."),
        (0.2, 1.0, "High", "Seek advice from a healthcare professional.")
    ]

    # Disease sections
    for disease, risk in results_dict.items():

        # Divider
        y = pdf.get_y()
        pdf.set_draw_color(180, 180, 180)
        pdf.line(10, y, 200, y)
        pdf.ln(4)

        # Determine risk category
        category_text, tip_text = "Unknown", ""
        for lower, upper, category, tip in risk_categories:
            if lower <= risk < upper:
                category_text, tip_text = category, tip
                break

        # Disease + Risk
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(120, 8, disease)
        pdf.cell(0, 8, f"Risk: {risk*100:.1f}% ({category_text})", ln=True)

        # Top 3 patient-centered factors
        factors = top_factors_dict.get(disease, ["No significant patient-centered factors identified."])
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 6, "Top contributing factors: " + ", ".join(factors))
        pdf.ln(2)

        # Disease description
        description_text = disease_descriptions.get(disease, "No description available.")
        pdf.set_font("Arial", 'I', 11)
        pdf.multi_cell(0, 6, description_text)

        # Tip
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 6, f"Tip: {tip_text}")     
   
        pdf.ln(3)

    # Divider line above patient concerns
    y = pdf.get_y()
    pdf.set_draw_color(180, 180, 180)  # light grey
    pdf.line(10, y, 200, y)
    pdf.ln(3)
    
    # Patient concerns / questions
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 6, "Patient concerns or questions:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, "______________________________________________________________\n" * 3) # change * 3 to change how many lines

    # Suggested follow-up
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 6, "Suggested follow-up:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, "[ ] 3 months   [ ] 6 months   [ ] 12 months", ln=True)

    # Disclaimer
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 5, "Disclaimer: This report is for informational purposes only and is not a medical diagnosis. \nPredictions are based on pre-trained models and the patient information provided. For any missing or unknown inputs, the models use average values from the training data, which may reduce the accuracy or reliability of the predictions. Please consult a licensed healthcare professional for personalized advice.")

    # Output PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes

# ---------------------------------------------------------------------------------------------
# --- Generate PDF & Download Button ---
# ---------------------------------------------------------------------------------------------

# --- Generate patient-specific top factors dynamically ---

top_factors_dynamic = {}
for disease in results.keys():
    top_factors_dynamic[disease] = get_top_factors(
        disease,
        patient_info,
        age_val=age,
        gender_val=gender,
        bmi_val=bmi,
        systolic_val=systolic,
        diastolic_val=diastolic,
        smoking_val=smoking_encoded,
        activity_val=activity_encoded
    )

if results:
    # Gather patient info
    patient_info = {
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "systolic": systolic,
    "diastolic": diastolic,
    "smoking": smoking_status,
    "alcohol": alcohol_consumption,
    "activity": physical_activity,
    "diabetes": disease_inputs.get("diabetes", {}),
    "ihd": {"cholesterol": 200, **disease_inputs.get("ihd", {})},  # allow default
    "covid": disease_inputs.get("covid", {}),
    }

    # Generate PDF bytes
    pdf_bytes = save_pdf(results, patient_info, top_factors_dynamic)

    # Streamlit download button
    st.download_button(
        label="Download Report as PDF",
        data=pdf_bytes,
        file_name="health_risk_report.pdf",
        mime="application/pdf"
    )

# ---------------------------------------------------------------------------------------------
# --- Patient Feedback Survey ---
# ---------------------------------------------------------------------------------------------

st.subheader("Patient Feedback")

# Track feedback submission status in session_state
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False  # Initialize submission state

with st.form(key="feedback_form"):
    # Feedback Input: Rating Scale (e.g., 1-5)
    feedback_rating = st.slider("Rate your experience (1 = Poor, 5 = Excellent)", 1, 5, 3)
    
    # Feedback Input: Free-text comments
    feedback_comments = st.text_area("Leave any additional comments or suggestions.")
    
    # Submit button
    submit_feedback = st.form_submit_button(
        label="Submit Feedback",
        disabled=st.session_state.feedback_submitted  # Disable button if feedback has been submitted
    )
    
    if submit_feedback:
        # Collect feedback and prepare data to send to your email
        feedback_data = {
            "Rating": feedback_rating,
            "Comments": feedback_comments,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Directly use email credentials
        SENDER_EMAIL = st.secrets["email"]["sender"]
        RECEIVER_EMAIL = st.secrets["email"]["receiver"]
        SENDER_PASSWORD = st.secrets["email"]["password"]
        SMTP_SERVER = st.secrets["email"]["smtp_server"]
        SMTP_PORT = st.secrets["email"]["smtp_port"]
        
        # Create the email message
        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = RECEIVER_EMAIL
        message["Subject"] = f"Patient Feedback - {feedback_data['Date']}"
        
        # Prepare the email body (feedback)
        body = f"""
        Feedback submitted on: {feedback_data['Date']}
        Rating: {feedback_data['Rating']}
        Comments:
        {feedback_data['Comments']}
        """
        
        # Attach the body to the email
        message.attach(MIMEText(body, "plain"))

        # Send email via SMTP
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()  # Secure the connection
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
            st.success("Feedback submitted successfully! Thank you for your input.")
            st.session_state.feedback_submitted = True  # Update the state to disable the button
        except Exception as e:
            st.error(f"Error sending feedback: {e}")