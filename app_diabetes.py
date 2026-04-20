import streamlit as st
import pandas as pd
import joblib
# Load the trained model
logis = joblib.load('diabetes_model.pkl')
# Title of the app
st.title("Diabetes Prediction")
# Get user input
st.header("Input Features")
pregnancies = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
diabetes_pedigree = st.number_input("Diabetes Pedigree")
age = st.number_input("Age")
# Prepare the data for prediction
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigree': [diabetes_pedigree],
    'Age': [age]
})
 # Make prediction
if st.button("Predict"):
    prediction = logis.predict(input_data)
    st.write(f"Predicted Diabetes Result: {prediction[0]}")
