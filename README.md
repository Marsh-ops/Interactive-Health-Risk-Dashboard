------------------------------------------------------------------------------
# Interactive Health Risk Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-orange)

------------------------------------------------------------------------------
## Overview


The **Interactive Health Risk Dashboard** is a Streamlit-based web application that provides personalized health risk assessments for multiple conditions, including:

- Diabetes  
- Ischemic Heart Disease (IHD)  
- Stroke  
- COVID-19  

It leverages **machine learning models** to estimate probabilities of these conditions based on user-provided health metrics and disease-specific additional information.

> **Disclaimer:** This tool is for awareness purposes only. It is **not a medical diagnosis tool**. Always consult a licensed medical professional for official medical advice.

------------------------------------------------------------------------------
## Features


- User-friendly interface to input health metrics (BMI, blood pressure, heart rate, smoking, alcohol, activity level)  
- Disease-specific additional information with optional “N/A or Unknown” defaults  
- Real-time risk calculation using trained ML models  
- Risk visualization and downloadable PDF reports  
- General health tips based on risk categories  

------------------------------------------------------------------------------
## Risk Categories


| Risk %       | Category      | Suggested Action |
|--------------|--------------|----------------|
| 0–20%        | Low          | Maintain healthy lifestyle |
| 20–50%       | Moderate     | Monitor health, consider check-ups |
| 50–80%       | High         | Consult healthcare professional |
| 80–100%      | Very High    | Seek immediate medical advice |

------------------------------------------------------------------------------
## Installation & Setup

```bash
1. **Clone the repository:**

git clone https://github.com/yourusername/Interactive-Health-Risk-Dashboard.git
cd Interactive-Health-Risk-Dashboard

2. ** Create a virtual environment (optional but recommended):**

python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

3. **Install dependencies:**

pip install -r requirements.txt

4. Run the app locally:

streamlit run app.py
