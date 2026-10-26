"""Refactored FastAPI service for AI381 churn prediction (src/app/main.py).

Run from repository root:
    uvicorn src.app.main:app --port 8000
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
import pydantic
from pydantic import BaseModel, Field

PYDANTIC_V2 = int(pydantic.__version__.split(".")[0]) >= 2
if PYDANTIC_V2:
    from pydantic import ConfigDict, model_validator
else:
    from pydantic import root_validator

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
CANONICAL_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "contract",
    "internet_service",
    "support_calls",
]


def _normalize_charges_and_fee(values: Any) -> Any:
    if not isinstance(values, dict):
        return values
    data = dict(values)
    charges = data.get("monthly_charges")
    fee = data.get("monthly_fee")
    if charges is None and fee is None:
        raise ValueError("Either 'monthly_charges' or 'monthly_fee' must be provided.")
    if charges is None:
        data["monthly_charges"] = fee
    if fee is None:
        data["monthly_fee"] = charges
    return data


class CustomerRequest(BaseModel):
    """Pydantic input validation schema for /predict.

    Accepts either `monthly_charges` (v1) or `monthly_fee` (v2) and validates
    ranges and allowed categorical values before inference.
    """

    tenure_months: int = Field(..., ge=0, le=1200, description="Customer tenure in months (>= 0)")
    monthly_charges: Optional[float] = Field(None, gt=0, le=10000, description="Monthly charge in USD (> 0)")
    monthly_fee: Optional[float] = Field(None, gt=0, le=10000, description="Monthly fee alias for v2 data (> 0)")
    contract: Literal["month-to-month", "one-year", "two-year"] = Field(
        ..., description="Contract type"
    )
    internet_service: Literal["dsl", "fiber", "none"] = Field(
        ..., description="Internet service tier"
    )
    support_calls: int = Field(..., ge=0, le=100, description="Number of support calls (>= 0)")

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

        @model_validator(mode="before")
        @classmethod
        def validate_charges_or_fee(cls, values: Any) -> Any:
            return _normalize_charges_and_fee(values)
    else:
        class Config:
            extra = "forbid"

        @root_validator(pre=True)
        def validate_charges_or_fee(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            return _normalize_charges_and_fee(values)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    dataset: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


def load_model_and_metadata(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Run `python -m src.train.train` first."
        )
    loaded_model = joblib.load(model_path)
    metadata = getattr(loaded_model, "metadata_", None)
    if metadata is None:
        meta_path = model_path.with_suffix(".metadata.json")
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            metadata = {"model_version": "unknown", "features": CANONICAL_FEATURES}
    return loaded_model, metadata


app = FastAPI(title="AI381 Churn Prediction API", version="2.0.0")
model, model_metadata = load_model_and_metadata(MODEL_PATH)


def _expected_features() -> list:
    """Inspect the loaded pipeline or metadata to determine required column names."""
    if "features" in model_metadata:
        return list(model_metadata["features"])
    try:
        prep = dict(model.steps)["prep"]
        cols = []
        for _, _, c in prep.transformers:
            cols.extend(c)
        return cols
    except Exception:
        return CANONICAL_FEATURES


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=str(model_metadata.get("model_version", "unknown")),
        dataset=model_metadata.get("dataset"),
        metrics=model_metadata.get("metrics"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerRequest) -> PredictionResponse:
    features = _expected_features()
    payload = {
        "tenure_months": customer.tenure_months,
        "monthly_charges": customer.monthly_charges,
        "monthly_fee": customer.monthly_fee,
        "contract": customer.contract,
        "internet_service": customer.internet_service,
        "support_calls": customer.support_calls,
    }
    try:
        X = pd.DataFrame([payload])[features]
        p = float(model.predict_proba(X)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return PredictionResponse(
        churn_probability=round(p, 3),
        churn_prediction=int(p > 0.5),
        model_version=str(model_metadata.get("model_version", "unknown")),
    )