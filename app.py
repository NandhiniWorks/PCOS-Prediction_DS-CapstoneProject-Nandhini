import streamlit as st
import pandas as pd
import joblib

model = joblib.load("pcos_logistic_model.pkl")
scaler = joblib.load("pcos_scaler.pkl")

st.title("PCOS Prediction System")
st.write("Enter the patient's details below.")

cycle = st.selectbox("Cycle", ["Regular", "Irregular"])
beta_hcg_ii = st.number_input("Beta-HCG-II", value=0.0)
hair_growth = st.selectbox("Hair Growth", ["No", "Yes"])
follicle_left = st.number_input("Follicle Left", min_value=0.0, value=0.0)
follicle_right = st.number_input("Follicle Right", min_value=0.0, value=0.0)

if st.button("Predict"):

    cycle_value = 1 if cycle == "Irregular" else 0
    hair_value = 1 if hair_growth == "Yes" else 0

    new_patient = pd.DataFrame({
        "Cycle(R/I)": [cycle_value],
        "Beta-HCG-II": [beta_hcg_ii],
        "Hair Growth": [hair_value],
        "Follicle Left": [follicle_left],
        "Follicle Right": [follicle_right]
    })

    new_patient_scaled = scaler.transform(new_patient)

    prediction = model.predict(new_patient_scaled)

    if prediction[0] == 1:
        st.error("PCOS predicted")
    else:
        st.success("No PCOS predicted")

st.caption("This tool provides a machine-learning prediction and is not a medical diagnosis.")