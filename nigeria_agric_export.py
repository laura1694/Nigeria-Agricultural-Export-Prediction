# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

# App configuration
st.set_page_config(
    page_title="Nigeria Agricultural Export Prediction",
    layout="wide"
)

# =========================
# 🎯 HEADER
# =========================
st.title("🌾 Nigeria Agricultural Export Prediction App")
st.markdown("This app predicts whether an agricultural export is High Value or Low Value.")
st.markdown("---")

# =========================
# 🤖 LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        model = jb.load("NAM.pkl")  # <-- make sure this file exists
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# =========================
# 🎛️ SIDEBAR INPUTS
# =========================
with st.sidebar:
    st.header("📊 Input Features")

    year = st.slider("Year", 2000, 2025, 2020)

    export_value = st.number_input(
        "Export Value", min_value=0.0, max_value=1000000.0, value=1000.0
    )

    product = st.selectbox(
        "Product",
        ["Cocoa", "Cashew", "Sesame", "Palm Oil", "Rubber"]
    )

    destination = st.selectbox(
        "Destination Country",
        ["USA", "UK", "China", "India", "Germany"]
    )

# =========================
# 🔢 ENCODING (IMPORTANT)
# =========================

# MUST MATCH TRAINING ENCODING
product_map = {
    "Cocoa": 0,
    "Cashew": 1,
    "Sesame": 2,
    "Palm Oil": 3,
    "Rubber": 4
}

country_map = {
    "USA": 0,
    "UK": 1,
    "China": 2,
    "India": 3,
    "Germany": 4
}

product_encoded = product_map[product]
country_encoded = country_map[destination]

# =========================
# 📦 FEATURES ARRAY
# =========================
features = np.array([[
    export_value,
    year,
    product_encoded,
    country_encoded
]])

# =========================
# 🔮 PREDICTION
# =========================
if st.sidebar.button("Predict"):

    if model is None:
        st.error("Model not loaded properly.")
    else:
        try:
            prediction = model.predict(features)[0]

            st.markdown("## 📢 Prediction Result")

            if prediction == 1:
                st.error("🚨 HIGH VALUE EXPORT")
                st.write("This export is classified as high value.")
            else:
                st.success("✅ LOW VALUE EXPORT")
                st.write("This export is classified as low value.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")