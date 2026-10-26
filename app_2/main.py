"""Lab 1 FastAPI skeleton: serves the model you saved from the notebook.

Run it by hand (this is Round 1 of the race):
    uvicorn app.main:app --port 8000
Then send predictions with curl (see sample_request.json).
"""
import joblib
import pandas as pd
from fastapi import FastAPI

MODEL_PATH = "model.joblib"   # where the notebook saves the model (yes, hard-coded on purpose)
FEATURES = ["tenure_months", "monthly_fee", "contract", "internet_service", "support_calls"]

app = FastAPI(title="AI381 Lab 1 churn API")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: dict):
    X = pd.DataFrame([customer])[FEATURES]
    p = float(model.predict_proba(X)[0, 1])
    return {"churn_probability": round(p, 3)}
