# streamlit_app.py

import streamlit as st
import pandas as pd
import sqlite3
import joblib
import json
import os
from datetime import datetime

# ===============================
# CONFIG & FILE PATHS
# ===============================
MODEL_FILE = "xgb_churn_model.pkl"
FEATURES_FILE = "features.json"
THRESHOLD_FILE = "threshold.json"

st.set_page_config(page_title="Churn AI Strategy Pro", layout="wide")

# ===============================
# LOAD MODEL & RESOURCES
# ===============================
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_FILE):
        st.error(f"{MODEL_FILE} not found. Train your model first!")
        st.stop()
    model = joblib.load(MODEL_FILE)

    with open(FEATURES_FILE, "r") as f:
        features = json.load(f)

    threshold = 0.5
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r") as f:
            threshold = json.load(f).get("threshold", 0.5)

    import shap
    explainer = shap.TreeExplainer(model)

    return model, features, threshold, explainer

model, ALL_FEATURES, OPTIMAL_THRESHOLD, explainer = load_resources()

# ===============================
# DATABASE (SAFE MIGRATION)
# ===============================
def init_db():
    conn = sqlite3.connect("churn_app.db", check_same_thread=False)
    c = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    # Predictions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            churn_probability REAL,
            risk TEXT,
            created_at DATETIME,
            total_charges REAL,
            notes TEXT
        )
    """)

    # Safe migration: add missing columns if table exists but schema changed
    c.execute("PRAGMA table_info(predictions)")
    existing_cols = [row[1] for row in c.fetchall()]
    if "notes" not in existing_cols:
        c.execute("ALTER TABLE predictions ADD COLUMN notes TEXT")
    if "total_charges" not in existing_cols:
        c.execute("ALTER TABLE predictions ADD COLUMN total_charges REAL")

    conn.commit()
    return conn

conn = init_db()
db_cursor = conn.cursor()

# ===============================
# AUTHENTICATION
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.title("Authentication")
    mode = st.sidebar.radio("Mode", ["Login", "Sign Up"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if mode == "Sign Up":
        if st.sidebar.button("Register"):
            try:
                db_cursor.execute(
                    "INSERT INTO users (username, password) VALUES (?,?)",
                    (username, password)
                )
                conn.commit()
                st.sidebar.success("Registered! Switch to Login.")
            except:
                st.sidebar.error("Username already exists.")
    else:
        if st.sidebar.button("Login"):
            db_cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, password)
            )
            if db_cursor.fetchone():
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
    st.stop()

# ===============================
# HEADER
# ===============================
st.sidebar.success(f"Logged in as: {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.title("📊 Telecom Customer Churn – Prescriptive AI")
st.markdown("---")

# ===============================
# CUSTOMER INPUT FORM
# ===============================
st.subheader("👤 Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=10.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

with col3:
    total_charges = tenure * monthly
    st.metric("Total Charges ($)", round(total_charges, 2))

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("🚀 Run AI Analysis"):

    # Map inputs to model features
    input_dict = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total_charges,
        "gender_Male": 1 if gender=="Male" else 0,
        "Partner_Yes": 1 if partner=="Yes" else 0,
        "Dependents_Yes": 1 if dependents=="Yes" else 0,
        "PhoneService_Yes": 1,
        "MultipleLines_No phone service": 0,
        "MultipleLines_Yes": 0,
        "InternetService_Fiber optic": 1 if internet=="Fiber optic" else 0,
        "InternetService_No": 1 if internet=="No" else 0,
        "OnlineSecurity_Yes": 1 if online_security=="Yes" else 0,
        "OnlineSecurity_No internet service": 1 if online_security=="No internet service" else 0,
        "OnlineBackup_Yes": 1 if online_backup=="Yes" else 0,
        "OnlineBackup_No internet service": 1 if online_backup=="No internet service" else 0,
        "DeviceProtection_Yes": 1 if device_protection=="Yes" else 0,
        "DeviceProtection_No internet service": 1 if device_protection=="No internet service" else 0,
        "TechSupport_Yes": 1 if tech_support=="Yes" else 0,
        "TechSupport_No internet service": 1 if tech_support=="No internet service" else 0,
        "StreamingTV_Yes": 1 if streaming_tv=="Yes" else 0,
        "StreamingTV_No internet service": 1 if streaming_tv=="No internet service" else 0,
        "StreamingMovies_Yes": 1 if streaming_movies=="Yes" else 0,
        "StreamingMovies_No internet service": 1 if streaming_movies=="No internet service" else 0,
        "Contract_One year": 1 if contract=="One year" else 0,
        "Contract_Two year": 1 if contract=="Two year" else 0,
        "PaperlessBilling_Yes": 1 if paperless=="Yes" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method=="Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method=="Mailed check" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method=="Credit card" else 0
    }

    df_final = pd.DataFrame([input_dict]).reindex(columns=ALL_FEATURES, fill_value=0)

    # Predict churn probability
    proba = model.predict_proba(df_final)[0, 1]
    risk_level = "🔴 High Risk" if proba>=0.7 else "🟡 Medium Risk" if proba>=0.4 else "🟢 Low Risk"

    # Save prediction to DB (audit trail)
    db_cursor.execute("""
        INSERT INTO predictions (username, churn_probability, risk, created_at, total_charges, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (st.session_state.username, proba, risk_level, datetime.now(), total_charges, "Predicted churn"))
    conn.commit()

    # Display metrics
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{proba*100:.2f}%")
    col2.metric("Risk Level", risk_level)

    # ===============================
    # SHAP TABLE
    # ===============================
    st.subheader("🔍 SHAP Feature Impact (Descending)")
    shap_values = explainer.shap_values(df_final)[0]
    shap_df = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Impact": shap_values,
        "Your Value": df_final.iloc[0].values
    })
    shap_df["AbsImpact"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values("AbsImpact", ascending=False).drop(columns="AbsImpact")
    st.dataframe(shap_df.style.background_gradient(cmap="RdYlGn_r", subset=["Impact"]), use_container_width=True)

    # ===============================
    # STRATEGY SIMULATOR
    # ===============================
    st.subheader("🧪 Retention Strategy Simulator")
    strategies = [
        ("10% Discount", {"MonthlyCharges": monthly*0.9}),
        ("25% Discount", {"MonthlyCharges": monthly*0.75}),
        ("Upgrade Contract to 1 Year", {"Contract_One year":1, "Contract_Two year":0}),
        ("Add Online Security", {"OnlineSecurity_Yes":1}),
        ("Add Tech Support", {"TechSupport_Yes":1}),
        ("Discount + Security", {"MonthlyCharges":monthly*0.85, "OnlineSecurity_Yes":1})
    ]

    results = []
    for name, change in strategies:
        sim = df_final.copy()
        for k,v in change.items():
            if k in sim.columns:
                sim[k] = v
        new_prob = model.predict_proba(sim)[0,1]
        results.append({
            "Strategy": name,
            "New Probability (%)": round(new_prob*100,2),
            "Improvement (%)": round((proba-new_prob)*100,2)
        })

    st.table(pd.DataFrame(results).sort_values("New Probability (%)"))
    best_action = results[0]["Strategy"]
    st.success(f"🤖 AI Recommendation: **{best_action}**")
