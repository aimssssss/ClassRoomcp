import streamlit as st
import joblib
import pandas as pd

# Load the brain we created in Step 2
model = joblib.load('classroom_comfort_model.pkl')

st.title("🇳🇬 Classroom Comfort Predictor")

# Creating the sliders
temp = st.slider("Temperature (°C)", 20, 45, 30)
humidity = st.slider("Humidity (%)", 10, 100, 60)
co2 = st.slider("CO2 Level", 400, 1500, 700)
light = st.slider("Light Intensity", 100, 500, 300)

if st.button("Predict Comfort"):
    # Match the columns from your friend's code
    features = pd.DataFrame([[temp, humidity, co2, light]], 
                            columns=["Temperature", "Humidity", "CO2", "Light"])

    prediction = model.predict(features)[0]
    labels = {0: "Uncomfortable", 1: "Neutral", 2: "Comfortable"}

    st.success(f"Result: {labels[prediction]}")