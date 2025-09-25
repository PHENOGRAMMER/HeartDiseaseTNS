import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("HeartDisease_Ensemble.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction")
st.write("Enter patient details below to get the heart disease risk prediction.")

# 1️⃣ Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=35)
sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0
cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3, value=2)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=110)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=180)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0
restecg = st.number_input("Resting ECG Results (0–2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=179)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0
st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.number_input("ST Slope", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thalassemia (1–3)", min_value=1, max_value=3, value=2)

# 2️⃣ Feature Engineering (same as training)
age_chol = age * chol
thalach_age = thal / (age + 1)

# 3️⃣ Prepare input array (ensure SAME order as training)
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, st_depression, st_slope, ca, thal, age_chol, thalach_age]])

# 4️⃣ Scale input
input_scaled = scaler.transform(input_data)

# 5️⃣ Predict
pred = model.predict(input_scaled)[0]
risk_proba = model.predict_proba(input_scaled)[0][1]  # probability of heart disease

# 6️⃣ Display result
if st.button("Predict"):
    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({risk_proba*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({risk_proba*100:.2f}%)")
