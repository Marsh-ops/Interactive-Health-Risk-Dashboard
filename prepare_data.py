#---------------------------------------------------------------------------------------------#
# --- Dependencies ---
#---------------------------------------------------------------------------------------------#

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

#---------------------------------------------------------------------------------------------#
# --- Step 1: Load datasets ---
#---------------------------------------------------------------------------------------------#

data_path = r"E:\Software Engineerng\Projects\Portfolio\Interactive Health Risk Dashboard\datasets"

diabetes_df = pd.read_csv(os.path.join(data_path, "diabetes.csv"))
ihd_df = pd.read_csv(os.path.join(data_path, "heartdisease.csv"))
stroke_df = pd.read_csv(os.path.join(data_path, "stroke.csv"))
covid_df = pd.read_csv(os.path.join(data_path, "covid-19.csv"))

#---------------------------------------------------------------------------------------------#
# --- Step 2a: Handle missing values ---
#---------------------------------------------------------------------------------------------#

diabetes_df = diabetes_df.fillna(diabetes_df.mean())
ihd_df = ihd_df.fillna(ihd_df.mean())
stroke_df = stroke_df.dropna()
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

# Apply encoding
diabetes_df = encode_categorical(diabetes_df, ['Gender'])
ihd_df = encode_categorical(ihd_df, ['sex', 'cp'])
stroke_df = encode_categorical(stroke_df, ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
covid_df = encode_categorical(covid_df, ['gender', 'vaccination_status'])  # updated for actual COVID columns

#---------------------------------------------------------------------------------------------#
# --- Step 2c: Select features and target ---
#---------------------------------------------------------------------------------------------#

X_diabetes = diabetes_df.drop("Outcome", axis=1)
y_diabetes = diabetes_df["Outcome"]

X_ihd = ihd_df.drop("target", axis=1)
y_ihd = ihd_df["target"]

X_stroke = stroke_df.drop("stroke", axis=1)
y_stroke = stroke_df["stroke"]

X_covid = covid_df.drop("mortality", axis=1)  # updated for actual target column
y_covid = covid_df["mortality"]

#---------------------------------------------------------------------------------------------#
# --- Optional: check shapes ---
#---------------------------------------------------------------------------------------------#

print("Diabetes X,y shapes:", X_diabetes.shape, y_diabetes.shape)
print("IHD X,y shapes:", X_ihd.shape, y_ihd.shape)
print("Stroke X,y shapes:", X_stroke.shape, y_stroke.shape)
print("COVID X,y shapes:", X_covid.shape, y_covid.shape)

#---------------------------------------------------------------------------------------------#
# --- Step 3: STrain and Save Models ---
#---------------------------------------------------------------------------------------------#

def train_and_save_model(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    joblib.dump(model, f"{model_name}.pkl")
    return model

# Train models for all datasets
diabetes_model = train_and_save_model(X_diabetes, y_diabetes, "diabetes_model")
ihd_model = train_and_save_model(X_ihd, y_ihd, "ihd_model")
stroke_model = train_and_save_model(X_stroke, y_stroke, "stroke_model")
covid_model = train_and_save_model(X_covid, y_covid, "covid_model")
