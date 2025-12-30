from pathlib import Path
import joblib
import streamlit as st
import pandas as pd

# -------------------------------
# Load model ONCE (CRITICAL FIX)
# -------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "stroke_logistic_model.joblib"
    data = joblib.load(model_path)
    return data["model"], data["scaler"]

model, scaler = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("Stroke Prediction System")
st.write("Enter patient details to predict stroke risk")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "children", "Never_worked"]
)
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox(
    "Smoking Status",
    ["formerly smoked", "never smoked", "smokes"]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Stroke"):
    input_data = {
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": 1 if ever_married == "Yes" else 0,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,

        "gender_Male": 1 if gender == "Male" else 0,
        "gender_Other": 1 if gender == "Other" else 0,

        "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
        "work_type_Private": 1 if work_type == "Private" else 0,
        "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
        "work_type_children": 1 if work_type == "children" else 0,

        "Residence_type_Urban": 1 if residence_type == "Urban" else 0,

        "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
        "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
        "smoking_status_smokes": 1 if smoking_status == "smokes" else 0
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")
