"""Refactored training script for AI381 churn model (src/train/train.py).

Usage:
    python -m src.train.train --data churn_v2.csv --model-out model.joblib
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CANONICAL_NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL_FEATURES = ["contract", "internet_service"]
TARGET_COLUMN = "churn"
COLUMN_ALIASES = {
    "monthly_fee": "monthly_charges",
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file for dataset/artifact provenance."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_data(data_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load dataset, normalize known column renames, and validate required columns."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Normalize known column aliases (e.g. monthly_fee -> monthly_charges)
    rename_map = {
        alias: canonical
        for alias, canonical in COLUMN_ALIASES.items()
        if alias in df.columns and canonical not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = CANONICAL_NUMERIC + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {data_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df, CANONICAL_NUMERIC + CATEGORICAL_FEATURES


def build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Construct preprocessing + logistic regression pipeline."""
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_and_evaluate(
    data_path: Path,
    model_out: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train churn model, evaluate on stratified holdout split, and save model + metadata."""
    df, features = load_and_validate_data(data_path)
    X = df[features]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_pipeline(CANONICAL_NUMERIC, CATEGORICAL_FEATURES)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    f1 = round(float(f1_score(y_test, proba > 0.5)), 3)
    auc = round(float(roc_auc_score(y_test, proba)), 3)

    data_sha256 = compute_sha256(data_path)
    model_version = f"{data_path.stem}-{data_sha256[:8]}"
    metadata = {
        "model_version": model_version,
        "dataset": data_path.name,
        "dataset_rows": int(len(df)),
        "dataset_sha256": data_sha256,
        "features": features,
        "metrics": {"f1": f1, "roc_auc": auc},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Attach metadata directly to pipeline and save sidecar JSON
    model.metadata_ = metadata
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    metadata["model_sha256"] = compute_sha256(model_out)
    model.metadata_ = metadata
    joblib.dump(model, model_out)

    meta_path = model_out.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Dataset      : {data_path.name} ({len(df)} rows)")
    print(f"Model version: {model_version}")
    print(f"F1           : {f1}")
    print(f"AUC          : {auc}")
    print(f"Saved model  : {model_out} (metadata: {meta_path})")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI381 churn prediction model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("churn_v2.csv") if Path("churn_v2.csv").exists() else Path("churn_v1.csv"),
        help="Path to training CSV file (churn_v1.csv or churn_v2.csv).",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("model.joblib"),
        help="Output path for serialized joblib model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(data_path=args.data, model_out=args.model_out)