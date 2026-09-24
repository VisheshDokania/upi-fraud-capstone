"""
Week 6b - Save the SMS and phishing-URL classifiers for the API (new file).
Week 6 evaluated them but never saved the models. This script:
  - drops duplicate texts first (the phishing-URL set has many exact duplicates;
    with a random split the same URL can land in train and test, inflating scores)
  - re-evaluates on a clean 80/20 split and prints the honest numbers
  - refits on all data and saves sklearn Pipelines to models/

Run from the project root:
    python scripts/06b_export_nlp_models.py
"""
import importlib.util
import gc
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from app.text_features import Cleaner, clean_text, clean_url  # noqa: E402

MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "notebooks" / "nlp_module_report"
spec = importlib.util.spec_from_file_location("week6", HERE / "06_nlp_phishing_module.py")
week6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(week6)


def make_pipeline(kind):
    return Pipeline([("clean", Cleaner(kind)),
                     ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])


def fit_and_save(df, kind, fname):
    before = len(df)
    text_cleaner = clean_url if kind == "url" else clean_text
    df = df[["text", "label"]].copy()
    df["clean_text"] = df["text"].fillna("").astype(str).map(text_cleaner)
    label_counts = df.groupby("clean_text", sort=False)["label"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    conflict_count = len(conflicting)
    if conflicting:
        df = df.loc[~df["clean_text"].isin(conflicting)]
    df = df.drop_duplicates("clean_text").reset_index(drop=True)
    after = len(df)
    print(f"{kind}: {before:,} rows; removed {conflict_count:,} conflicting cleaned texts; "
          f"{after:,} rows remain after deduplication")
    Xtr, Xte, ytr, yte = train_test_split(df["text"].astype(str), df["label"], test_size=0.2,
                                          stratify=df["label"], random_state=42)
    evaluation_model = make_pipeline(kind).fit(Xtr, ytr)
    p = evaluation_model.predict_proba(Xte)[:, 1]
    pr_auc = average_precision_score(yte, p)
    f1 = f1_score(yte, p >= 0.5)
    print(f"  cleaned-text test PR-AUC {pr_auc:.4f}  F1@0.5 {f1:.4f}")
    del Xtr, Xte, ytr, yte, evaluation_model, p
    gc.collect()
    full_model = make_pipeline(kind).fit(df["text"].astype(str), df["label"])
    joblib.dump(full_model, MODEL_DIR / fname)
    del full_model, df
    gc.collect()
    return {
        "model": kind,
        "rows_before": before,
        "conflicting_clean_texts_removed": conflict_count,
        "rows_after_dedupe": after,
        "test_pr_auc": pr_auc,
        "test_f1_at_0_5": f1,
    }


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(fit_and_save(week6.load_sms_spam(), "sms", "sms_spam.joblib"))
    rows.append(fit_and_save(week6.load_phishing_urls(), "url", "phishing_url.joblib"))
    pd.DataFrame(rows).to_csv(OUT_DIR / "nlp_export_summary.csv", index=False)
    print("Saved models/sms_spam.joblib and models/phishing_url.joblib")
