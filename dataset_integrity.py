import json
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Project paths (update as needed)
# -----------------------------
BASE_DIR = r"E:\Software Engineerng\Projects\Portfolio\Interactive Health Risk Dashboard"
DATA_DIR = BASE_DIR + r"\datasets"
MODEL_DIR = BASE_DIR + r"\models"

CONFIG = {
    "Diabetes": {
        "dataset": DATA_DIR + r"\diabetes.csv",
        "model": MODEL_DIR + r"\diabetes_model.pkl",
        "feature_means": MODEL_DIR + r"\diabetes_model_feature_means.json",
        "target": "Outcome",
        "features": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    },
    "IHD": {
        "dataset": DATA_DIR + r"\ihd.csv",
        "model": MODEL_DIR + r"\ihd_model.pkl",
        "feature_means": MODEL_DIR + r"\ihd_model_feature_means.json",
        "target": "target",
        "features": ['age', 'sex', 'chest_pain_type', 'resting_bp_s', 'cholesterol', 
                     'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 
                     'exercise_angina', 'oldpeak', 'st_slope']
    },
    "Stroke": {
        "dataset": DATA_DIR + r"\stroke.csv",
        "model": MODEL_DIR + r"\stroke_model.pkl",
        "feature_means": MODEL_DIR + r"\stroke_model_feature_means.json",
        "target": "stroke",
        "features": ['id', 'gender', 'age', 'hypertension', 'heart_disease', 
                     'ever_married', 'work_type', 'Residence_type', 
                     'avg_glucose_level', 'bmi', 'smoking_status']
    },
    "Covid": {
        "dataset": DATA_DIR + r"\covid.csv",
        "model": MODEL_DIR + r"\covid_model.pkl",
        "feature_means": MODEL_DIR + r"\covid_model_feature_means.json",
        "target": "covid_result",
        "features": ['patient_id', 'age', 'gender', 'fever', 'dry_cough', 'sore_throat', 
                     'fatigue', 'headache', 'shortness_of_breath', 'loss_of_smell', 
                     'loss_of_taste', 'oxygen_level', 'body_temperature', 'comorbidity', 
                     'travel_history', 'contact_with_patient', 'chest_pain']
    }
}

# -----------------------------
# Audit function
# -----------------------------
def audit_model(name, dataset_path, model_path, feature_means_path, target_col, expected_features):
    print(f"\n\n================ AUDITING {name.upper()} =================\n")

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)

    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found in dataset. Columns: {df.columns.tolist()}")
        return

    model = joblib.load(model_path)

    with open(feature_means_path, "r") as f:
        feature_means = json.load(f)

    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(df[target_col].value_counts(normalize=True))

    # Feature alignment
    missing_features = set(expected_features) - set(df.columns)
    extra_features = set(df.columns) - set(expected_features) - {target_col}
    if missing_features:
        print("⚠️ Missing features:", missing_features)
    if extra_features:
        print("ℹ️ Extra features:", extra_features)

    # Preprocess
    df_proc = df.copy()

    # Fill missing values
    for col, mean in feature_means.items():
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(mean)

    # Encode categorical columns using feature_means mapping if available
    for col in expected_features:
        if col in df_proc.columns and df_proc[col].dtype == object:
            mapping = feature_means.get(col, None)
            if isinstance(mapping, dict):
                df_proc[col] = df_proc[col].map(mapping)
            else:
                # fallback: convert to category codes
                df_proc[col] = df_proc[col].astype("category").cat.codes

    X = df_proc[expected_features]
    y_true = df_proc[target_col]

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    print("\nPerformance Metrics:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Probability analysis
    print("\nProbability analysis:")
    print("Min :", np.min(y_prob))
    print("Max :", np.max(y_prob))
    print("Mean:", np.mean(y_prob))
    print("Std :", np.std(y_prob))

    # Dataset usability
    dataset_ok = True
    if missing_features or y_true.nunique() < 2:
        dataset_ok = False
    print("Dataset usable:", "✅ YES" if dataset_ok else "❌ NO")

    # Bias detection
    prevalence = y_true.mean()
    mean_prob = np.mean(y_prob)
    biased = mean_prob > prevalence + 0.15
    print("Model biased:", "YES 🚨" if biased else "NO ✅")

    # OOD detection
    z_scores = []
    for col in expected_features:
        if col in df.columns and df[col].dtype != object:
            z = abs(df[col].mean() - feature_means[col]) / (np.std(df[col]) + 1e-6)
            z_scores.append(z)
    mean_z = np.mean(z_scores) if z_scores else 0
    if mean_z > 2.5:
        ood = "VERY"
    elif mean_z > 1.5:
        ood = "MILD"
    else:
        ood = "NO"
    print("OOD detected:", ood)

    # Calibration check
    needs_calibration = (np.std(y_prob) < 0.1 or abs(mean_prob - prevalence) > 0.15)
    print("Calibration needed:", "ALMOST CERTAINLY ⚠️" if needs_calibration else "NO")

    # Public testing safety
    if not dataset_ok or ood == "VERY":
        safety = "❌ NOT SAFE"
    elif biased or needs_calibration:
        safety = "⚠️ LIMITED / INTERNAL ONLY"
    else:
        safety = "✅ SAFE FOR PILOT"
    print("Public testing safety:", safety)
    print("\n==============================================")

# -----------------------------
# Run all audits
# -----------------------------
if __name__ == "__main__":
    for name, cfg in CONFIG.items():
        audit_model(
            name=name,
            dataset_path=cfg["dataset"],
            model_path=cfg["model"],
            feature_means_path=cfg["feature_means"],
            target_col=cfg["target"],
            expected_features=cfg["features"]
        )

    print("\n✅ All audits complete.")