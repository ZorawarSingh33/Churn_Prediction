import pandas as pd
from app import predict_churn  # Import your app.py function

# -----------------------------
# Example new customer data
# -----------------------------
new_customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 10,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer",
    "MonthlyCharges": 55.0,
    "TotalCharges": 3300.0
}])

# -----------------------------
# Run prediction
# -----------------------------
result = predict_churn(new_customer)

# -----------------------------
# Risk bucket logic
# -----------------------------
def risk_bucket(p):
    if p >= 0.8:
        return "High Risk"
    elif p >= 0.5:
        return "Medium Risk"
    else:
        return "Low Risk"

result["risk_level"] = result["Churn_Probability"].apply(risk_bucket)

# -----------------------------
# Format final output
# -----------------------------
final_result = {
    "churn_probability": round(float(result["Churn_Probability"].iloc[0]), 3),
    "churn_prediction": int(result["Churn_Prediction"].iloc[0]),
    "risk_level": result["risk_level"].iloc[0]
}

# -----------------------------
# Print final output
# -----------------------------
print(final_result)
