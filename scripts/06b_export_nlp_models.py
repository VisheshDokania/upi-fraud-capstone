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
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from app.text_features import Cleaner  # noqa: E402

MODEL_DIR = ROOT / "models"
spec = importlib.util.spec_from_file_location("week6", HERE / "06_nlp_phishing_module.py")
week6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(week6)


def make_pipeline(kind):
    return Pipeline([("clean", Cleaner(kind)),
                     ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])


def fit_and_save(df, kind, fname):
    before = len(df)
    df = df.drop_duplicates("text")
    print(f"{kind}: {before:,} rows -> {len(df):,} after removing duplicate texts")
    Xtr, Xte, ytr, yte = train_test_split(df["text"].astype(str), df["label"], test_size=0.2,
                                          stratify=df["label"], random_state=42)
    p = make_pipeline(kind).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    print(f"  dedup test PR-AUC {average_precision_score(yte, p):.4f}  F1@0.5 {f1_score(yte, p >= 0.5):.4f}")
    joblib.dump(make_pipeline(kind).fit(df["text"].astype(str), df["label"]), MODEL_DIR / fname)


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    fit_and_save(week6.load_sms_spam(), "sms", "sms_spam.joblib")
    fit_and_save(week6.load_phishing_urls(), "url", "phishing_url.joblib")
    print("Saved models/sms_spam.joblib and models/phishing_url.joblib")
