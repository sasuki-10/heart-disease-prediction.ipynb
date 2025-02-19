import pickle
import numpy as np
import streamlit as st

def load_model():
    with open("heart_disease_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def predict_heart_disease(alcohol, smoking, past_history, family_history, pulse):
    model = load_model()
    user_input = np.array([[alcohol, smoking, past_history, family_history, pulse]])
    prediction = model.predict(user_input)
    return "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Answer the following questions to predict your heart disease risk:")

alcohol = st.radio("Do you consume alcohol?", [1, 0])
smoking = st.radio("Do you smoke?", [1, 0])
past_history = st.radio("Do you have a past history of heart disease?", [1, 0])
family_history = st.radio("Is there a family history of heart disease?", [1, 0])
pulse = st.number_input("Enter your pulse rate:", min_value=40, max_value=200, step=1)

if st.button("Predict"):
    result = predict_heart_disease(alcohol, smoking, past_history, family_history, pulse)
    st.subheader("Prediction:")
    st.write(result)

# Run the app using: streamlit run script_name.py
