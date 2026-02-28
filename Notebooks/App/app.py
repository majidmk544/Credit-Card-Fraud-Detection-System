import streamlit as st
import numpy as np
import joblib

# =====================
# LOAD MODELS & SCALER
# =====================
model_lr = joblib.load("model_lr.pkl")     # Logistic Regression (Stage 1)
model_xgb = joblib.load("model_xgb.pkl")   # XGBoost (Stage 2)
scaler = joblib.load("scaler.pkl")         # StandardScaler used in training

# Tuned thresholds
T1 = 0.05
T2 = 0.45

# Required features
FEATURES = ['V17', 'V14', 'V12', 'V10', 'V3', 'V16', 'V7', 'V11', 'V4', 'V18', 'Amount']

# =====================
# PREDICTION FUNCTION
# =====================
def predict_dual_model(features_list):
    # features_list is already in correct order
    features_raw = features_list.copy()
    
    # Scale only Amount (last column)
    amount_scaled = scaler.transform([[features_raw[-1]]])[0][0]
    features_raw[-1] = amount_scaled
    
    X = np.array([features_raw])
    
    lr_prob = model_lr.predict_proba(X)[:, 1][0]
    xgb_prob = model_xgb.predict_proba(X)[:, 1][0]
    
    pred = (
        (lr_prob >= T1) and
        (xgb_prob >= T2)
    )
    return pred, lr_prob, xgb_prob



# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>💳 Fraud Detection App</h1>
    <p style='text-align: center; font-size: 18px;'>
    Enter transaction details below and find out if it’s <b>Fraud</b> or <b>Legit</b>.
    </p>
    """,
    unsafe_allow_html=True
)

# Input fields in 2 columns
col1, col2 = st.columns(2)
user_inputs = []

for i, feature in enumerate(FEATURES):
    if i % 2 == 0:
        val = col1.number_input(f"{feature}", value=0.0, format="%.4f")
    else:
        val = col2.number_input(f"{feature}", value=0.0, format="%.4f")
    user_inputs.append(val)

# Predict button
if st.button("🚀 Predict", use_container_width=True):
    pred, lr_prob, xgb_prob = predict_dual_model(user_inputs)

    if pred == 1:
        st.markdown(
            f"""
            <div style='background-color: #ffe6e6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: red;'>🚨 FRAUD DETECTED!</h2>
            <p><b>LR Probability:</b> {lr_prob:.2%} | <b>XGB Probability:</b> {xgb_prob:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='background-color: #e6ffe6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: green;'>✅ Legit Transaction</h2>
            <p><b>LR Probability:</b> {lr_prob:.2%} | <b>XGB Probability:</b> {xgb_prob:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
