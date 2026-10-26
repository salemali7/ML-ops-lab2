"""Pytest suite for AI381 Churn Prediction API and training pipeline (tests/test_api.py).

Covers:
1. API contract (/health and /predict response structure and types)
2. Model loads (pipeline steps, predict_proba interface, metadata, and missing file error)
3. Input schema (Pydantic 422 validation on missing/invalid/out-of-range/extra fields)
4. Edge case (boundary customer with 0 tenure/0 support calls and extreme maximum values)
5. Deterministic prediction (exact repeatable probability across calls and v1/v2 aliases)
"""
from pathlib import Path
import sys

# Ensure repository root is on sys.path even when invoked via plain `pytest`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.train.train import train_and_evaluate  # noqa: E402

# Ensure model.joblib exists before importing src.app.main
MODEL_PATH = Path("model.joblib")
DATA_PATH = Path("churn_v2.csv") if Path("churn_v2.csv").exists() else Path("churn_v1.csv")
if not MODEL_PATH.exists() and DATA_PATH.exists():
    train_and_evaluate(data_path=DATA_PATH, model_out=MODEL_PATH)

from src.app.main import app, load_model_and_metadata  # noqa: E402

client = TestClient(app)


def test_api_contract():
    """1. API contract: /health and /predict return HTTP 200 and match response schemas."""
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_body = health_resp.json()
    assert health_body["status"] == "ok"
    assert isinstance(health_body["model_version"], str) and len(health_body["model_version"]) > 0

    predict_resp = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charges": 75.5,
            "contract": "one-year",
            "internet_service": "dsl",
            "support_calls": 1,
        },
    )
    assert predict_resp.status_code == 200
    pred_body = predict_resp.json()
    assert set(pred_body.keys()) == {"churn_probability", "churn_prediction", "model_version"}
    assert isinstance(pred_body["churn_probability"], float)
    assert 0.0 <= pred_body["churn_probability"] <= 1.0
    assert pred_body["churn_prediction"] in (0, 1)
    assert pred_body["model_version"] == health_body["model_version"]


def test_model_loads(tmp_path: Path):
    """2. Model loads: serialized pipeline loads with expected steps, metadata, and error handling."""
    loaded_model, metadata = load_model_and_metadata(MODEL_PATH)
    assert hasattr(loaded_model, "predict_proba")
    step_names = [name for name, _ in loaded_model.steps]
    assert "prep" in step_names and "clf" in step_names
    assert "model_version" in metadata
    assert "features" in metadata and len(metadata["features"]) == 5

    # Verify missing model artifact raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_model_and_metadata(tmp_path / "nonexistent_model.joblib")


@pytest.mark.parametrize(
    "invalid_payload",
    [
        # Missing both monthly_charges and monthly_fee
        {
            "tenure_months": 5,
            "contract": "month-to-month",
            "internet_service": "fiber",
            "support_calls": 3,
        },
        # Negative tenure_months (violates ge=0)
        {
            "tenure_months": -1,
            "monthly_charges": 89.9,
            "contract": "month-to-month",
            "internet_service": "fiber",
            "support_calls": 3,
        },
        # Non-positive monthly_charges (violates gt=0)
        {
            "tenure_months": 5,
            "monthly_charges": 0.0,
            "contract": "month-to-month",
            "internet_service": "fiber",
            "support_calls": 3,
        },
        # Unsupported contract category
        {
            "tenure_months": 5,
            "monthly_charges": 89.9,
            "contract": "weekly",
            "internet_service": "fiber",
            "support_calls": 3,
        },
        # Unknown extra field (violates extra="forbid")
        {
            "tenure_months": 5,
            "monthly_charges": 89.9,
            "contract": "month-to-month",
            "internet_service": "fiber",
            "support_calls": 3,
            "unexpected_field": "malicious",
        },
    ],
)
def test_input_schema_validation(invalid_payload: dict):
    """3. Input schema: invalid types, ranges, categories, or extra fields return HTTP 422."""
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_edge_case_zero_tenure_and_extreme_values():
    """4. Edge case: brand-new customer (tenure=0, calls=0) vs high-risk boundary customer."""
    new_customer_low_risk = {
        "tenure_months": 0,
        "monthly_charges": 12.0,
        "contract": "two-year",
        "internet_service": "none",
        "support_calls": 0,
    }
    high_risk_boundary = {
        "tenure_months": 0,
        "monthly_charges": 130.0,
        "contract": "month-to-month",
        "internet_service": "fiber",
        "support_calls": 10,
    }

    resp_low = client.post("/predict", json=new_customer_low_risk)
    resp_high = client.post("/predict", json=high_risk_boundary)

    assert resp_low.status_code == 200
    assert resp_high.status_code == 200

    p_low = resp_low.json()["churn_probability"]
    p_high = resp_high.json()["churn_probability"]

    assert 0.0 < p_low < 0.5
    assert 0.5 < p_high <= 1.0
    assert p_high > p_low


def test_deterministic_prediction():
    """5. Deterministic prediction: identical requests (and v1/v2 alias keys) return identical output."""
    payload_v1 = {
        "tenure_months": 5,
        "monthly_charges": 89.9,
        "contract": "month-to-month",
        "internet_service": "fiber",
        "support_calls": 3,
    }
    payload_v2 = {
        "tenure_months": 5,
        "monthly_fee": 89.9,
        "contract": "month-to-month",
        "internet_service": "fiber",
        "support_calls": 3,
    }

    r1 = client.post("/predict", json=payload_v1)
    r2 = client.post("/predict", json=payload_v1)
    r3 = client.post("/predict", json=payload_v2)

    assert r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 200
    assert r1.json() == r2.json() == r3.json()
    assert r1.json()["churn_probability"] == 0.895
    assert r1.json()["churn_prediction"] == 1
