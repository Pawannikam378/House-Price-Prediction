import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('linear_model.pkl')

st.set_page_config(page_title="Home Price Prediction", page_icon="🏠")
st.title("Home Price Prediction App")
st.write("Enter House Details Below")

# Accept Inputs
income = st.number_input("Average Area Income")
house_age = st.number_input("Average Area House Age")
rooms = st.number_input("Average Area Number of Rooms")
bedrooms = st.number_input("Average Area Number of Bedrooms")
population = st.number_input("Average Area Population")

# Predict
if st.button("Predict Price"):
    input_data = np.array([[income, house_age, rooms, bedrooms, population]])
    prediction = model.predict(input_data)

    st.success(f"Prediction: ${prediction[0]}")