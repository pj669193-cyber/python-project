import streamlit as st
import numpy as np
import joblib

model = joblib.load("data/fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to check if it's fraudulent.")

st.sidebar.header("Transaction Input")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
time = st.sidebar.number_input("Time (seconds elapsed)", min_value=0.0, value=50000.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**V1 - V28 Features**")
v_features = []
for i in range(1, 29):
    v = st.sidebar.slider(f"V{i}", -5.0, 5.0, 0.0)
    v_features.append(v)

if st.button("🔍 Detect Fraud"):
    amount_scaled = (amount - 88.35) / 250.12
    time_scaled = (time - 94813) / 47488
    features = np.array([[time_scaled, *v_features, amount_scaled]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.r(f"🚨 FRAUDULENT! (Confidence: {probability*100:.1f}%)")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud prob: {probability*100:.1f}%)")

    st.metric("Fraud Probability", f"{probability*100:.2f}%")
