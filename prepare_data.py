#---------------------------------------------------------------------------------------------#
# --- Dependencies ---
#---------------------------------------------------------------------------------------------#

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

#---------------------------------------------------------------------------------------------#
# --- Step 0: Paths ---
#---------------------------------------------------------------------------------------------#

data_path = r"E:\Software Engineerng\Projects\Portfolio\Interactive Health Risk Dashboard\datasets"
model_path = r"E:\Software Engineerng\Projects\Portfolio\Interactive Health Risk Dashboard\models"

os.makedirs(model_path, exist_ok=True)  # ensure model folder exists

#---------------------------------------------------------------------------------------------#
# --- Step 1: Load datasets ---
#---------------------------------------------------------------------------------------------#

diabetes_df = pd.read_csv(os.path.join(data_path, "diabetes.csv"))
ihd_df = pd.read_csv(os.path.join(data_path, "heartdisease.csv"))
stroke_df = pd.read_csv(os.path.join(data_path, "stroke.csv"))
covid_df = pd.read_csv(os.path.join(data_path, "covid-19.csv"))

#---------------------------------------------------------------------------------------------#
# --- Step 2a: Handle missing values ---
#---------------------------------------------------------------------------------------------#

diabetes_df = diabetes_df.fillna(diabetes_df.mean())
ihd_df = ihd_df.fillna(ihd_df.mean())
stroke_df = stroke_df.dropna()  # keep only complete rows
covid_df = covid_df.fillna(0)

#---------------------------------------------------------------------------------------------#
# --- Step 2b: Encode categorical variables ---
#---------------------------------------------------------------------------------------------#

def encode_categorical(df, columns):
    le = LabelEncoder()
    for col in columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df

diabetes_df = encode_categorical(diabetes_df, ['Gender'])
ihd_df = encode_categorical(ihd_df, ['sex', 'cp'])
stroke_df = encode_categorical(stroke_df, ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
covid_df = encode_categorical(covid_df, ['gender', 'vaccination_status'])

#---------------------------------------------------------------------------------------------#
# --- Step 2c: Select features and target ---
#---------------------------------------------------------------------------------------------#

X_diabetes = diabetes_df.drop("Outcome", axis=1)
y_diabetes = diabetes_df["Outcome"].astype(int)

X_ihd = ihd_df.drop("target", axis=1)
y_ihd = ihd_df["target"].astype(int)

X_stroke = stroke_df.drop("stroke", axis=1)
y_stroke = stroke_df["stroke"].astype(int)

X_covid = covid_df.drop("mortality", axis=1)
y_covid = covid_df["mortality"].astype(int)

# Optional: check shapes
print("Diabetes X,y shapes:", X_diabetes.shape, y_diabetes.shape)
print("IHD X,y shapes:", X_ihd.shape, y_ihd.shape)
print("Stroke X,y shapes:", X_stroke.shape, y_stroke.shape)
print("COVID X,y shapes:", X_covid.shape, y_covid.shape)

#---------------------------------------------------------------------------------------------#
# --- Step 3: Train and Save Models ---
#---------------------------------------------------------------------------------------------#

def train_and_save_model(X, y, model_name, n_estimators=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Safe RandomForest settings for large datasets
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=1)
    
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    
    # Save to model_path
    joblib.dump(model, os.path.join(model_path, f"{model_name}.pkl"))
    
    return model

# Train models (use fewer trees if system hangs)
diabetes_model = train_and_save_model(X_diabetes, y_diabetes, "diabetes_model", n_estimators=50)
ihd_model = train_and_save_model(X_ihd, y_ihd, "ihd_model", n_estimators=50)
stroke_model = train_and_save_model(X_stroke, y_stroke, "stroke_model", n_estimators=50)
print("Training COVID model...")
covid_model = train_and_save_model(X_covid, y_covid, "covid_model", n_estimators=50)
print("COVID model trained successfully!")

print("All models trained and saved successfully!")

# Now you can safely print feature names
print("Diabetes model features:", diabetes_model.feature_names_in_)
print("IHD model features:", ihd_model.feature_names_in_)
print("Stroke model features:", stroke_model.feature_names_in_)
print("COVID model features:", covid_model.feature_names_in_)

# Save feature means for later use in the Streamlit app
def save_feature_means(X, model_name):
    feature_means = X.mean().to_dict()
    with open(os.path.join(model_path, f"{model_name}_feature_means.json"), "w") as f:
        json.dump(feature_means, f)

save_feature_means(X_diabetes, "diabetes")
save_feature_means(X_ihd, "ihd")
save_feature_means(X_stroke, "stroke")
save_feature_means(X_covid, "covid")