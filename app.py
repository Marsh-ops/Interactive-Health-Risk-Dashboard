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

diabetes_model = joblib.load(os.path.join(MODEL_PATH, "diabetes_model_calibrated.pkl"))
ihd_model = joblib.load(os.path.join(MODEL_PATH, "ihd_model_calibrated.pkl"))
stroke_model = joblib.load(os.path.join(MODEL_PATH, "stroke_model_calibrated.pkl"))
covid_model = joblib.load(os.path.join(MODEL_PATH, "covid_model_calibrated.pkl"))

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

def neutral_impute(input_df, neutral_values, feature_means):
    """
    Fill NaNs using neutral values where defined, otherwise fallback to feature means.
    """
    filled = input_df.copy()

    for col in filled.columns:
        if col in neutral_values:
            filled[col] = filled[col].fillna(neutral_values[col])
        else:
            filled[col] = filled[col].fillna(feature_means.get(col, 0))

    return filled

def safe_predict(model, input_df, feature_means, neutral_values=None):
    """
    Safe prediction with optional neutral imputation.
    """
    # Ensure correct feature order
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=np.nan)

    if neutral_values:
        input_df = neutral_impute(input_df, neutral_values, feature_means)
    else:
        for col in model.feature_names_in_:
            input_df[col] = input_df[col].fillna(feature_means[col])

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

IHD_NEUTRAL_VALUES = {
    "age": 45,                    # mid-adult, not elderly
    "sex": 0,                     # female baseline (lower risk)
    "chest_pain_type": 0,         # none
    "resting_bp_s": 120,          # normal BP
    "cholesterol": 180,           # healthy
    "exercise_angina": 0,         # no
    "st_slope": 2,                # upsloping (normal)
    "oldpeak": 0.0,               # none
    "max_heart_rate": 170,        # normal
    "fasting_blood_sugar": 0,     # normal
    "resting_ecg": 0              # normal
}
# ⚠️ These are not medical advice, but model-safe baselines.

RISK_BANDS = [
    (0.0, 0.1, "Low", "Maintain your healthy lifestyle."),
    (0.1, 0.2, "Moderate", "Consider improving diet, exercise, and regular check-ups."),
    (0.2, 0.4, "High", "Consult a healthcare professional to review risk factors."),
    (0.4, 1.0, "Very High", "Immediate medical consultation is strongly recommended.")
]

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
    3. Provide as much additional information as possible for accuracy when prompted
    4. View your personalised risk assessment for the selected disease/s
    """
)

st.markdown("### Select conditions to assess:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    show_diabetes = st.checkbox("🩸 Diabetes")

with col2:
    show_ihd = st.checkbox("❤️ Ischaemic Heart Disease (IHD)")

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

    st_slope_na = st.checkbox("ST slope during exercise: Unknown")

    if st_slope_na:
        st_slope = np.nan
    else:
        st_slope_label = st.selectbox(
            "ST slope during exercise",
            ["Upsloping (normal)", "Flat", "Downsloping (abnormal)"]
        )
        st_slope_map = {
            "Upsloping (normal)": 2,
            "Flat": 1,
            "Downsloping (abnormal)": 0
        }
        st_slope = st_slope_map[st_slope_label]

    st_slope_unknown = int(st_slope_na)

    oldpeak_na = st.checkbox("ST depression (Oldpeak): Unknown")

    oldpeak = (
        np.nan if oldpeak_na
        else st.number_input(
            "ST depression during exercise (Oldpeak)",
            min_value=0.0,
            max_value=6.0,
            value=0.0,
            step=0.1
        )
    )

    oldpeak_unknown = int(oldpeak_na)

    max_hr_na = st.checkbox("Maximum heart rate: Unknown")

    max_heart_rate = (
        np.nan if max_hr_na
        else st.number_input(
            "Maximum heart rate achieved",
            min_value=60,
            max_value=220,
            value=170
        )
    )

    max_hr_unknown = int(max_hr_na)

    disease_inputs["ihd"] = {
        "chest_pain": chest_pain,
        "chest_pain_unknown": chest_pain_unknown,
        "exercise_angina": exercise_angina,
        "exercise_angina_unknown": exercise_angina_unknown,
        "cholesterol": cholesterol,
        "cholesterol_unknown": cholesterol_unknown,
        "st_slope": st_slope,
        "st_slope_unknown": st_slope_unknown,
        "oldpeak": oldpeak,
        "oldpeak_unknown": oldpeak_unknown,
        "max_heart_rate": max_heart_rate,
        "max_hr_unknown": max_hr_unknown
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
        results["Diabetes"] = diabetes_model.predict_proba(X_diabetes)[:, 1][0]

    # -------------------- IHD --------------------
    if show_ihd:
        ihd_features = {col: np.nan for col in ihd_model.feature_names_in_}

        ihd_features.update({
            "age": age if not age_unknown else np.nan,
            "sex": gender_binary,
            "chest_pain_type": disease_inputs.get("ihd", {}).get("chest_pain", np.nan),
            "resting_bp_s": systolic if not systolic_unknown else np.nan,
            "cholesterol": disease_inputs.get("ihd", {}).get("cholesterol", np.nan),
            "exercise_angina": disease_inputs.get("ihd", {}).get("exercise_angina", np.nan),
            "st_slope": disease_inputs.get("ihd", {}).get("st_slope", np.nan),
            "oldpeak": disease_inputs.get("ihd", {}).get("oldpeak", np.nan),
            "max_heart_rate": disease_inputs.get("ihd", {}).get("max_heart_rate", np.nan)
        })

        X_ihd = pd.DataFrame([ihd_features], columns=ihd_model.feature_names_in_)
        X_ihd_filled = X_ihd.fillna(ihd_feature_means)

        results["IHD"] = ihd_model.predict_proba(X_ihd_filled)[:, 1][0]

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
        results["Stroke"] = stroke_model.predict_proba(X_stroke)[:, 1][0]

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

        X_covid = pd.DataFrame([covid_features], columns=covid_model.feature_names_in_)
        results["COVID-19"] = covid_model.predict_proba(X_covid)[:, 1][0]

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

def get_top_factors(disease, patient_data, 
                    age_val=None, gender_val=None, bmi_val=None,
                    systolic_val=None, diastolic_val=None, smoking_val=None, activity_val=None):
    """
    Returns top 3 patient-centered factors that contribute to disease risk.
    Sidebar values (generic) take precedence; additional info fills gaps.
    """
    factors = []

    # --- Diabetes ---
    if disease == "Diabetes":
        stroke = patient_data.get("diabetes", {})
        bmi = bmi_val if bmi_val is not None else patient_data["diabetes"].get("bmi", np.nan)
        dpf = patient_data["diabetes"].get("dpf", np.nan)
        glucose = patient_data["diabetes"].get("avg_glucose", np.nan)
        skin = patient_data["diabetes"].get("skin_thickness", np.nan)
        smoke = smoking_val if smoking_val is not None else patient_data["diabetes"].get("smoking_status", 0)

        if not np.isnan(bmi) and bmi > 25:
            factors.append("High BMI")
        if not np.isnan(dpf) and dpf > 1.0:
            factors.append("Family history of diabetes")
        if not np.isnan(glucose) and glucose > 140:
            factors.append("High average glucose")
        if not np.isnan(skin) and skin > 30:
            factors.append("High skin thickness")
        if smoke != 0:
            factors.append(next((k for k,v in smoking_map.items() if v==smoke), "Smoker"))

    # --- IHD ---
    elif disease == "IHD":
        ihd = patient_data.get("ihd", {})

        chol = ihd.get("cholesterol", np.nan)
        sys_bp = systolic_val if systolic_val is not None else np.nan
        smoke = smoking_val if smoking_val is not None else 0
        activity = activity_val if activity_val is not None else 2
        max_hr = ihd.get("max_heart_rate", np.nan)
        chest = ihd.get("chest_pain", np.nan)
        ex_angina = ihd.get("exercise_angina", np.nan)

        if not np.isnan(chol) and chol > 240:
            factors.append("High cholesterol")
        if not np.isnan(sys_bp) and sys_bp > 140:
            factors.append("High systolic blood pressure")
        if smoke != 0:
            factors.append("Smoking")
        if activity < 2:
            factors.append("Low physical activity")
        if not np.isnan(ex_angina) and ex_angina == 1:
            factors.append("Exercise-induced angina")
        if not np.isnan(max_hr) and max_hr > 150:
            factors.append("High heart rate")
        if not np.isnan(chest) and chest > 0:
            factors.append("Chest pain")

    # --- Stroke ---
    elif disease == "Stroke":
        stroke = patient_data.get("stroke", {})

        sys_bp = systolic_val if systolic_val is not None else stroke.get("systolic", np.nan)
        dia_bp = diastolic_val if diastolic_val is not None else stroke.get("diastolic", np.nan)
        bmi = bmi_val if bmi_val is not None else stroke.get("bmi", np.nan)
        smoke = smoking_val if smoking_val is not None else stroke.get("smoking_status", 0)
        prev_tia = stroke.get("previous_tia", np.nan)

        if not np.isnan(sys_bp) and sys_bp > 130:
            factors.append("High systolic BP")
        if not np.isnan(dia_bp) and dia_bp > 80:
            factors.append("High diastolic BP")
        if not np.isnan(bmi) and bmi > 30:
            factors.append("High BMI")
        if not np.isnan(prev_tia) and prev_tia > 0:
            factors.append("Previous TIA / mini-stroke")
        if smoke != 0:
            factors.append(next((k for k,v in smoking_map.items() if v==smoke), "Smoker"))

    # --- COVID-19 ---
    elif disease == "COVID-19":
        covid_data = patient_data.get("covid", {})

        age = age_val if age_val is not None else covid_data.get("age", np.nan)
        gender = gender_val if gender_val is not None else covid_data.get("gender", np.nan)

        # Symptoms
        for symptom in ["fever", "dry_cough", "sore_throat", "fatigue", "headache",
                        "shortness_of_breath", "loss_of_smell", "loss_of_taste", "chest_pain"]:
            if covid_data.get(symptom, 0):
                factors.append(symptom.replace("_", " ").title())

        # Comorbidities
        if covid_data.get("comorbidity", 0):
            factors.append("Comorbidity present")

        # Age-based risk
        if not np.isnan(age) and age > 60:
            factors.append("Advanced age")

    # Return top 3 factors or default message
    return factors[:3] if factors else ["No significant patient-centered factors identified."]

def interpret_risk(risk):
    category_text, tip_text = "Unknown", ""

    for lower, upper, category, tip in RISK_BANDS:
        if lower <= risk < upper:
            category_text, tip_text = category, tip
            break

    return category_text, tip_text

#---------------------------------------------------------------------------------------------#
# --- PDF Generation and Export ---
#---------------------------------------------------------------------------------------------#

patient_info = {
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "systolic": systolic,
    "diastolic": diastolic,
    "smoking_encoded": smoking_encoded,
    "smoking_label": smoking_status,
    "alcohol": alcohol_consumption,
    "activity_encoded": activity_encoded,
    "activity_label": physical_activity,
    "diabetes": disease_inputs.get("diabetes", {}),
    "ihd": disease_inputs.get("ihd", {}),
    "stroke": disease_inputs.get("stroke", {}),
    "covid": disease_inputs.get("covid", {})
}

# Disease descriptions
disease_descriptions = {
    "IHD": "Disease description: Reduced blood supply to the heart, can lead to chest pain or heart attack.",
    "Diabetes": "Disease description: High blood sugar levels due to insulin issues.",
    "Stroke": "Disease description: Interruption of blood flow to the brain causing potential long-term damage.",
    "COVID-19": "Disease description: Viral infection that can affect respiratory and other systems."
}

# def normalize_disease_name(name: str) -> str:
  #  """
   # Removes ' Risk' suffix or trailing whitespace to match disease_descriptions keys.
    #"""
    #return name.replace(" Risk", "").strip()

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
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 6, description_text)

        # Tip
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 6, f"Tip: {tip_text}")     
   
        pdf.ln(3)

        # --- Add IHD-specific disclaimer ---
        if disease == "IHD":
            pdf.set_font("Arial", 'I', 10)
            pdf.multi_cell(
                0,
                5,
                "Note: IHD risk predictions may be elevated if additional patient information is unavailable. Please interpret results in consultation with a healthcare professional."
            )

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
    pdf.multi_cell(
    0,
    5,
    "Disclaimer: This report is for informational and educational purposes only and is not a medical diagnosis.\n"
    "Predictions are generated using pre-trained machine learning models based on patient-reported inputs.\n"
    "while other models use average values derived from their training data. "
    "This may reduce accuracy and reliability. Please consult a licensed healthcare professional for medical advice."
    )

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

    if results:  # Only show download if we have prediction results
        pdf_bytes = save_pdf(results, patient_info, top_factors_dynamic)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
            file_name="health_risk_report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Please calculate risk first to generate the PDF report.")

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