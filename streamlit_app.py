import streamlit as st
import pandas as pd
import joblib
import json
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="ChurnAI Pro | Strategic Retention",
    page_icon="📊",
    layout="wide"
)

# ===============================
# RESOURCE LOADER
# ===============================
MODEL_FILE = "xgb_churn_model.pkl"
FEATURES_FILE = "features.json"

@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_FILE):
        st.error("🚨 Model files missing! Ensure xgb_churn_model.pkl and features.json are in the repo.")
        st.stop()
    
    model = joblib.load(MODEL_FILE)
    with open(FEATURES_FILE, "r") as f:
        features = json.load(f)
    
    import shap
    # Using TreeExplainer for XGBoost speed and stability
    explainer = shap.TreeExplainer(model)
    return model, features, explainer

model, ALL_FEATURES, explainer = load_resources()

# ===============================
# MAIN DASHBOARD UI
# ===============================
st.title("📊 Customer Retention Intelligence")
st.markdown("Immediate churn risk assessment and prescriptive analytics.")

# Single container for the main app logic
with st.container():
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("👤 Profile")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 5)
        with c2:
            st.subheader("🔌 Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        with c3:
            st.subheader("💳 Billing & Contract")
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            total_charges_est = tenure * monthly

        run_analysis = st.form_submit_button("Analyze Report")

    if run_analysis:
        # Mapping for the model features
        PAYMENT_MAP = {
            "Electronic check": "PaymentMethod_Electronic check",
            "Mailed check": "PaymentMethod_Mailed check",
            "Bank transfer": "PaymentMethod_Bank transfer (automatic)",
            "Credit card": "PaymentMethod_Credit card (automatic)"
        }

        # Construct Input Dictionary (matching Telco Dataset structure)
        input_dict = {
            "SeniorCitizen": 1 if senior=="Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges_est,
            "gender_Male": 1 if gender=="Male" else 0,
            "Partner_Yes": 1 if partner=="Yes" else 0,
            "Dependents_Yes": 1 if dependents=="Yes" else 0,
            "PhoneService_Yes": 1 if phone=="Yes" else 0,
            "PhoneService_No": 1 if phone=="No" else 0,
            "MultipleLines_No": 1 if multiple=="No" else 0,
            "MultipleLines_No phone service": 1 if multiple=="No phone service" else 0,
            "MultipleLines_Yes": 1 if multiple=="Yes" else 0,
            "InternetService_DSL": 1 if internet=="DSL" else 0,
            "InternetService_Fiber optic": 1 if internet=="Fiber optic" else 0,
            "InternetService_No": 1 if internet=="No" else 0,
            "OnlineSecurity_No internet service": 1 if security=="No internet service" else 0,
            "OnlineSecurity_Yes": 1 if security=="Yes" else 0,
            "OnlineBackup_No internet service": 1 if backup=="No internet service" else 0,
            "OnlineBackup_Yes": 1 if backup=="Yes" else 0,
            "DeviceProtection_No internet service": 1 if device=="No internet service" else 0,
            "DeviceProtection_Yes": 1 if device=="Yes" else 0,
            "TechSupport_No internet service": 1 if tech=="No internet service" else 0,
            "TechSupport_Yes": 1 if tech=="Yes" else 0,
            "StreamingTV_No internet service": 1 if tv=="No internet service" else 0,
            "StreamingTV_Yes": 1 if tv=="Yes" else 0,
            "StreamingMovies_No internet service": 1 if movies=="No internet service" else 0,
            "StreamingMovies_Yes": 1 if movies=="Yes" else 0,
            "Contract_Month-to-month": 1 if contract=="Month-to-month" else 0,
            "Contract_One year": 1 if contract=="One year" else 0,
            "Contract_Two year": 1 if contract=="Two year" else 0,
            "PaperlessBilling_Yes": 1 if paperless=="Yes" else 0,
        }
        
        # Apply payment mapping
        for key in PAYMENT_MAP.values(): input_dict[key]=0
        input_dict[PAYMENT_MAP[payment]]=1

        # Transform to DataFrame
        df_final = pd.DataFrame([input_dict]).reindex(columns=ALL_FEATURES, fill_value=0)
        
        # Prediction
        proba = float(model.predict_proba(df_final)[0,1])
        risk = "🔴 High" if proba >= 0.7 else "🟡 Medium" if proba >= 0.4 else "🟢 Low"

        # RESULTS SECTION
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Risk", f"{proba*100:.1f}%")
        m2.metric("Assessment", risk)
        m3.metric("Projected Value", f"${total_charges_est:,.2f}")

        # SHAP EXPLANATION (Absolute Magnitude Ranking)
        st.subheader(" AI Feature Attribution (SHAP)")
        shap_values = explainer(df_final)
        shap_v = shap_values.values[0]
        shap_df = pd.DataFrame({"Feature": ALL_FEATURES, "Impact Score": shap_v})
        shap_df = shap_df.reindex(shap_df["Impact Score"].abs().sort_values(ascending=False).index)
        
        st.dataframe(
            shap_df.style.background_gradient(cmap="RdYlGn_r", subset=["Impact Score"]), 
            use_container_width=True
        )

        # RETENTION SIMULATOR
        st.subheader(" Prescriptive Retention Simulator")
        scenarios = [
            ("Switch to 2-Year Contract", {"Contract_Two year":1, "Contract_Month-to-month":0}),
            ("Switch to Annual Contract", {"Contract_One year":1, "Contract_Month-to-month":0}),
            ("Auto-pay (Credit Card)", {"PaymentMethod_Credit card (automatic)":1, "PaymentMethod_Electronic check":0}),
            ("Apply 15% Loyalty Discount", {"MonthlyCharges": monthly*0.85}),
            ("Downgrade Fiber to DSL", {
                "InternetService_Fiber optic":0, 
                "InternetService_DSL":1, 
                "InternetService_No":0, 
                "MonthlyCharges":max(20, monthly-20)
            }),
            ("Full Strategy: 2yr + Auto-Pay + 20% Discount", {
                "Contract_Two year":1, 
                "Contract_Month-to-month":0, 
                "PaymentMethod_Credit card (automatic)":1, 
                "MonthlyCharges":monthly*0.8
            })
        ]
        
        sim_results = []
        for name, changes in scenarios:
            sim_df = df_final.copy()
            # Reset groups to ensure mutual exclusivity
            sim_df[[c for c in sim_df.columns if c.startswith("Contract_")]] = 0
            sim_df[[c for c in sim_df.columns if c.startswith("PaymentMethod_")]] = 0
            sim_df[[c for c in sim_df.columns if c.startswith("InternetService_")]] = 0
            
            for col, val in changes.items():
                if col in sim_df.columns: sim_df[col] = val
            
            sim_df["TotalCharges"] = tenure * sim_df["MonthlyCharges"].values[0]
            new_p = float(model.predict_proba(sim_df)[0,1])
            reduction = max(0, (proba - new_p) * 100)
            sim_results.append({
                "Action": name, 
                "New Risk (%)": f"{new_p*100:.1f}%", 
                "Reduction Impact": f"{reduction:.1f}%"
            })
        
        st.table(pd.DataFrame(sim_results).sort_values("Reduction Impact", ascending=False))
