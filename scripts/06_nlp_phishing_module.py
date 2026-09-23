"""
Week 6 — NLP Phishing / SMS Fraud Module (local version)
=============================================================
Run:
    cd scripts
    python 06_nlp_phishing_module.py
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "notebooks" / "nlp_module_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_sms_spam():
    fpath = DATA_DIR / "sms_spam" / "spam.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} not found — run 01_data_acquisition.py first.")
    df = pd.read_csv(fpath, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = (df["label"] == "spam").astype(int)
    return df


def load_phishing_urls():
    fpath = DATA_DIR / "phishing_urls" / "phishing_site_urls.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} not found — run 01_data_acquisition.py first.")
    df = pd.read_csv(fpath)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"url": "text", "label": "label"})
    df["label"] = (df["label"].astype(str).str.lower() == "bad").astype(int)
    return df[["text", "label"]]


def clean_text(s):
    """For SMS/message text — a URL inside a message is noise relative to
    the surrounding words, so collapsing it to one token is correct here."""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_url(s):
    """For the phishing-URL dataset, the ENTIRE input is a URL — unlike
    clean_text() above, we must NOT strip URL structure, since that IS
    the signal (suspicious TLDs, hyphenated subdomains, login/verify
    keywords in the path). Split on URL delimiters instead of erasing them."""
    s = str(s).lower()
    s = re.sub(r"https?://", " ", s)
    s = re.sub(r"[/\.\-_?=&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def train_module(df, name, cleaner=clean_text):
    df = df.copy()
    df["clean_text"] = df["text"].astype(str).apply(cleaner)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train_vec, y_train)

    proba = clf.predict_proba(X_test_vec)[:, 1]
    pred = clf.predict(X_test_vec)
    report = classification_report(y_test, pred, output_dict=True)
    auc, f1 = roc_auc_score(y_test, proba), f1_score(y_test, pred)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {auc:.4f}  F1: {f1:.4f}")
    print(classification_report(y_test, pred, target_names=["legit", "suspicious"]))
    pd.DataFrame(report).T.to_csv(OUT_DIR / f"{name}_classification_report.csv")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_terms = feature_names[np.argsort(coefs)[-15:]][::-1]
    print(f"Top terms flagging '{name}' as suspicious: {list(top_terms)}")

    return {"module": name, "roc_auc": auc, "f1": f1, "top_terms": list(top_terms)}


if __name__ == "__main__":
    import json
    results = []
    sms_df = load_sms_spam()
    print(f"Loaded real SMS data: {len(sms_df):,} messages, {sms_df.label.mean()*100:.1f}% spam")
    results.append(train_module(sms_df, "sms_spam", cleaner=clean_text))
    del sms_df

    url_df = load_phishing_urls()
    print(f"Loaded real URL data: {len(url_df):,} URLs, {url_df.label.mean()*100:.1f}% phishing")
    results.append(train_module(url_df, "phishing_urls", cleaner=clean_url))
    del url_df

    with open(OUT_DIR / "nlp_module_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll outputs written to {OUT_DIR}/")
    print("Script exiting now — all memory released.")
