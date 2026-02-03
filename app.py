import pandas as pd
import joblib
import json
import logging

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logging.info("App initialized")

# -----------------------------
# File paths
# -----------------------------
model_path = r"C:\Users\zoraw\Desktop\Agentic Customer Segmentatio\xgb_churn_model.pkl"
features_path = r"C:\Users\zoraw\Desktop\Agentic Customer Segmentatio\features.json"
threshold_path = r"C:\Users\zoraw\Desktop\Agentic Customer Segmentatio\threshold.json"

# -----------------------------
# Load model & feature info
# -----------------------------
model = joblib.load(model_path)

with open(features_path, "r") as f:
    feature_columns = json.load(f)

with open(threshold_path, "r") as f:
    threshold = json.load(f)["threshold"]

# -----------------------------
# Allowed categories for input validation
# -----------------------------
ALLOWED_CATEGORIES = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
    "Electronic check",
    "Mailed check",
    "Bank transfer",
    "Credit card"
]

}

# -----------------------------
# Prediction function
# -----------------------------
def predict_churn(new_data: pd.DataFrame):
    # Validate categorical inputs
    for col, allowed in ALLOWED_CATEGORIES.items():
        if col in new_data.columns:
            invalid = ~new_data[col].isin(allowed)
            if invalid.any():
                raise ValueError(
                    f"Invalid value(s) in column '{col}': {new_data.loc[invalid, col].tolist()}"
                )

    # Ensure all expected columns exist
    X_new = pd.get_dummies(new_data)
    for col in feature_columns:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[feature_columns]  # order columns correctly

    # Predict probability
    proba = model.predict_proba(X_new)[:, 1]

    # Convert using threshold
    pred = (proba >= threshold).astype(int)

    # Compute risk level
    def risk_bucket(p):
        if p >= 0.8:
            return "High Risk"
        elif p >= 0.5:
            return "Medium Risk"
        else:
            return "Low Risk"

    risk_levels = [risk_bucket(p) for p in proba]

    return pd.DataFrame({
        "Churn_Probability": proba,
        "Churn_Prediction": pred,
        "Risk_Level": risk_levels
    })
