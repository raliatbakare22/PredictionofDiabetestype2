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

    # Get probabilities for both classes
    prediction_probability = logis.predict_proba(input_data)

    # Probability of diabetes = class 1
    probability = prediction_probability[0][1]

    st.write(f"Predicted Diabetes Result: {prediction[0]}")

    if prediction[0] == 1:

        st.markdown(
            f'''
            <h4 style="color:red; background-color:#000; font-size:20px;">
            The model predicts you <strong>have diabetes</strong>
            with a probability of {probability:.2f}.
            </h4>
            ''',
            unsafe_allow_html=True
        )

        st.status("Model Prediction Completed")

    else:

        st.markdown(
            f'''
            <h4 style="color:green; background-color:#000; font-size:20px;">
            The model predicts you <strong>do not have diabetes</strong>
            with a probability of {1 - probability:.2f}.
            </h4>
            ''',
            unsafe_allow_html=True
        )

        st.status("Model Prediction Completed")