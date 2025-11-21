import streamlit as st
import requests
import json

st.title("🚗 Used Car Price Predictor")

# Load dropdown options
with open('options.json', 'r') as f:
    options = json.load(f)

# UI
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Select Car Brand", options['companies'])
    car_name = st.selectbox("Select Model", options['models'])
    fuel_type = st.selectbox("Select Fuel Type", options['fuel_types'])

with col2:
    year = st.number_input("Manufacturing Year", min_value=1995, max_value=2019, value=2018)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)

# Predict Button
if st.button("Predict Price"):

    payload = {
        "company": company,
        "car_name": car_name,
        "year": year,
        "km_driven": km_driven,
        "fuel_type": fuel_type
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            price = response.json()['predicted_price']
            st.success(f"Estimated Price: ₹ {price:,.2f}")
        else:
            st.error(response.json())

    except Exception as e:
        st.error("Backend not running or some error occurred.")
        st.write(e)
