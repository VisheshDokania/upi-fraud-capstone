"""Train and evaluate the standalone UPI-like synthetic transaction baseline."""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             precision_recall_curve, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "notebooks" / "upi_synthetic_report" / "upi_synthetic_transactions.csv"
DEFAULT_REPORT_DIR = ROOT / "notebooks" / "upi_synthetic_report"
DEFAULT_MODEL = ROOT / "models" / "upi_synthetic_model.pkl"
TARGET = "is_fraud"
EXCLUDED = {
    "transaction_id", "timestamp", "sender_vpa", "receiver_vpa", "fraud_typology", TARGET,
}
CATEGORICAL = [
    "transaction_type", "initiation_mode", "merchant_mcc", "transaction_status",
    "sender_psp", "receiver_psp",
]


def prepare_features(frame: pd.DataFrame):
    """Derive time/provider categories without exposing account IDs to the classifier."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    if data["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid or missing values")
    data["transaction_hour_utc"] = data["timestamp"].dt.hour.astype("int8")
    data["transaction_dayofweek_utc"] = data["timestamp"].dt.dayofweek.astype("int8")
    for source, target in (("sender_vpa", "sender_psp"), ("receiver_vpa", "receiver_psp")):
        if source not in data:
            raise ValueError(f"required VPA field is missing: {source}")
        data[target] = (
            data[source].astype("string").str.rsplit("@", n=1).str[-1].fillna("Unknown")
        )
    features = data.drop(columns=[column for column in EXCLUDED if column in data.columns])
    missing_categories = sorted(set(CATEGORICAL) - set(features.columns))
    if missing_categories:
        raise ValueError(f"required categorical fields are missing: {missing_categories}")
    for column in CATEGORICAL:
        features[column] = features[column].astype("string").fillna("Unknown").astype(object)
    numeric = [column for column in features.columns if column not in CATEGORICAL]
    for column in numeric:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features, numeric


def best_validation_threshold(y_true, probabilities):
    """Choose an F1 threshold using validation labels only."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[int(np.argmax(f1))])


def evaluate_split(name, y_true, probabilities, threshold):
    predicted = probabilities >= threshold
    return {
        "period": name,
        "rows": int(len(y_true)),
        "fraud_rows": int(np.sum(y_true)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "threshold": float(threshold),
    }


def train(data_path=DEFAULT_DATA, report_dir=DEFAULT_REPORT_DIR, model_path=DEFAULT_MODEL):
    data_path = Path(data_path)
    report_dir = Path(report_dir)
    model_path = Path(model_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found at {data_path}; run "
            "python -m upi_synthetic.generate_data first."
        )
    data = pd.read_csv(data_path)
    required = {TARGET, "timestamp", "sender_vpa", "receiver_vpa"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype("int8")
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only binary labels 0 and 1")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)
    features, numeric = prepare_features(data)
    labels = data[TARGET].to_numpy(dtype="int8")
    row_count = len(data)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    if train_end < 1 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError("dataset is too small for chronological train/validation/test splits")

    X_train, y_train = features.iloc[:train_end], labels[:train_end]
    X_val, y_val = features.iloc[train_end:validation_end], labels[train_end:validation_end]
    X_test, y_test = features.iloc[validation_end:], labels[validation_end:]
    for split_name, target in (("train", y_train), ("validation", y_val), ("test", y_test)):
        if np.unique(target).size < 2:
            raise ValueError(
                f"{split_name} split has only one label class; generate more rows or use another seed"
            )

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL),
        ("numeric", numeric_pipeline, numeric),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2_000, random_state=42,
        )),
    ])

    started = time.perf_counter()
    model.fit(X_train, y_train)
    validation_probabilities = model.predict_proba(X_val)[:, 1]
    threshold = best_validation_threshold(y_val, validation_probabilities)
    validation_metrics = evaluate_split(
        "validation", y_val, validation_probabilities, threshold
    )
    # The fixed chronological test partition is scored once, after threshold selection.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_split("test", y_test, test_probabilities, threshold)
    elapsed = round(time.perf_counter() - started, 2)

    report_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([validation_metrics, test_metrics])
    metrics.to_csv(report_dir / "upi_synthetic_metrics.csv", index=False)
    transformed_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    coefficient_frame = pd.DataFrame({
        "feature": transformed_names,
        "log_odds_coefficient": coefficients,
    })
    coefficient_frame["absolute_coefficient"] = coefficient_frame["log_odds_coefficient"].abs()
    coefficient_frame.sort_values("absolute_coefficient", ascending=False).to_csv(
        report_dir / "upi_synthetic_coefficients.csv", index=False
    )
    bundle = {
        "name": "upi_synthetic_logistic_regression",
        "pipeline": model,
        "features": list(features.columns),
        "categorical_features": list(CATEGORICAL),
        "threshold": threshold,
        "threshold_selection": "validation F1",
        "dataset_kind": "synthetic UPI-like scenarios; not observed UPI transactions",
        "explainability": "standardized logistic-regression coefficients",
        "seed": 42,
    }
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    print(f"Model: {bundle['name']}")
    print(f"Rows: {row_count:,} | chronological split: 70% train / 15% validation / 15% test")
    print(f"Categoricals one-hot encoded: {', '.join(CATEGORICAL)}")
    print(f"Validation-selected F1 threshold: {threshold:.6f}")
    print(f"Validation metrics: {validation_metrics}")
    print(f"Fixed test metrics: {test_metrics}")
    print(f"Training and evaluation seconds: {elapsed:.2f}")
    print(f"Metrics and coefficient table: {report_dir}")
    print(f"New model artifact: {model_path}")
    print("All metrics are synthetic scenario results, not real UPI performance.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    train(args.data, args.report_dir, args.model_out)


if __name__ == "__main__":
    main()
