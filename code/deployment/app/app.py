import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://api:8000")
INPUT_SPECS = [
    ("koi_period", "Orbital period (days)", 9.75, 0.0001),
    ("koi_duration", "Transit duration (hours)", 3.79, 0.0001),
    ("koi_depth", "Transit depth (ppm)", 421.10, 0.0),
    ("koi_prad", "Planet radius (Earth radii)", 2.39, 0.0001),
    ("koi_teq", "Equilibrium temperature (K)", 878.0, 0.0001),
    ("koi_insol", "Insolation flux", 141.60, 0.0),
    ("koi_model_snr", "Signal-to-noise ratio", 23.0, 0.0),
    ("koi_steff", "Stellar effective temperature (K)", 5767.0, 0.0001),
    ("koi_slogg", "Stellar surface gravity", 4.44, 0.0001),
    ("koi_srad", "Stellar radius (Solar radii)", 1.0, 0.0001),
    ("koi_kepmag", "Kepler magnitude", 14.52, 0.0001),
]


st.set_page_config(page_title="Exoplanet Classifier", page_icon="🪐")
st.title("Kepler Exoplanet Classifier")
st.write("Enter the observed values to classify an exoplanet candidate.")

with st.form("prediction_form"):
    columns = st.columns(2)
    payload = {}
    for index, (name, label, value, minimum) in enumerate(INPUT_SPECS):
        with columns[index % 2]:
            payload[name] = st.number_input(label, min_value=minimum, value=value, format="%.4f")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.success(f"Predicted disposition: {result['prediction']}")
        st.json(result["probabilities"])
    except requests.RequestException:
        st.error("The API is unavailable. Please try again.")
