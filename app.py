import joblib
import pandas as pd
import gradio as gr
from pathlib import Path

# -------------------------------
# Load model ONCE
# -------------------------------
model_path = Path(__file__).parent / "stroke_logistic_model.joblib"
data = joblib.load(model_path)

model = data["model"]
scaler = data["scaler"]

# -------------------------------
# Prediction function
# -------------------------------
def predict_stroke(
    age, hypertension, heart_disease, avg_glucose_level, bmi,
    gender, ever_married, work_type, residence_type, smoking_status
):
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
    prediction = model.predict(input_scaled)[0]

    return "⚠️ High Risk of Stroke" if prediction == 1 else "✅ Low Risk of Stroke"

# -------------------------------
# Gradio UI
# -------------------------------
app = gr.Interface(
    fn=predict_stroke,
    inputs=[
        gr.Number(label="Age", value=30),
        gr.Radio(["No", "Yes"], label="Hypertension"),
        gr.Radio(["No", "Yes"], label="Heart Disease"),
        gr.Number(label="Average Glucose Level", value=100),
        gr.Number(label="BMI", value=25),

        gr.Radio(["Male", "Female", "Other"], label="Gender"),
        gr.Radio(["No", "Yes"], label="Ever Married"),
        gr.Dropdown(
            ["Private", "Self-employed", "children", "Never_worked"],
            label="Work Type"
        ),
        gr.Radio(["Urban", "Rural"], label="Residence Type"),
        gr.Dropdown(
            ["formerly smoked", "never smoked", "smokes"],
            label="Smoking Status"
        ),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Stroke Prediction System",
    description="Enter patient details to predict stroke risk"
)

if __name__ == "__main__":
    app.launch()
