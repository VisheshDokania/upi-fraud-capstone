"""
Week 1 — Data Acquisition & Integrity Audit (local / VS Code version)
=========================================================================
Run this from the VS Code integrated terminal, NOT as notebook cells --
each dataset is downloaded, verified, and its row-count check is done
inside download_and_verify(), so nothing large stays in memory after
this script exits. This is the actual fix for the Colab RAM crash: a
notebook keeps every variable alive across cells indefinitely, a script
releases everything the moment it finishes.

Setup (one-time):
    1. Get your Kaggle API key: kaggle.com/settings -> "Create New Token"
       (downloads kaggle.json -- open it, you need the two fields inside)
    2. cp .env.example .env
       Fill in KAGGLE_USERNAME and KAGGLE_KEY in .env
    3. pip install -r requirements.txt

Run:
    cd scripts
    python 01_data_acquisition.py
"""

import os
import subprocess
import zipfile
import gc
from pathlib import Path
from dotenv import load_dotenv

# --- credentials from .env, not Colab secrets ---
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
    raise RuntimeError(
        "KAGGLE_USERNAME / KAGGLE_KEY not found. Copy .env.example to .env "
        "and fill in your real Kaggle API credentials first."
    )
# the kaggle CLI reads these exact env var names
os.environ["KAGGLE_USERNAME"] = os.environ["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_KEY"]

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASETS = {
    "ieee_cis": {
        "kaggle_id": "c/ieee-fraud-detection",
        "kind": "competition",
        "expected_files": ["train_transaction.csv", "train_identity.csv"],
        "description": "Real Vesta e-commerce transactions, isFraud label",
    },
    "elliptic": {
        "kaggle_id": "ellipticco/elliptic-data-set",
        "kind": "dataset",
        # NOTE: this dataset extracts into a nested subfolder -- confirmed
        # from an actual run, not assumed.
        "expected_files": [
            "elliptic_bitcoin_dataset/elliptic_txs_features.csv",
            "elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
            "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
        ],
        "description": "Real Bitcoin transaction graph, licit/illicit/unknown labels",
    },
    "creditcard_fraud": {
        "kaggle_id": "mlg-ulb/creditcardfraud",
        "kind": "dataset",
        "expected_files": ["creditcard.csv"],
        "description": "Real anonymized European card transactions (ULB/Worldline)",
    },
    "paysim": {
        "kaggle_id": "ealaxi/paysim1",
        "kind": "dataset",
        "expected_files": ["PS_20174392719_1491204439457_log.csv"],
        "description": "Synthetic mobile-money simulator — kept as baseline only",
    },
    "sms_spam": {
        "kaggle_id": "uciml/sms-spam-collection-dataset",
        "kind": "dataset",
        "expected_files": ["spam.csv"],
        "description": "Real SMS ham/spam messages (UCI) — Week 6",
    },
    "phishing_urls": {
        "kaggle_id": "taruntiwarihp/phishing-site-urls",
        "kind": "dataset",
        "expected_files": ["phishing_site_urls.csv"],
        "description": "Real crawled legitimate vs phishing URLs — Week 6",
    },
}


def download_and_verify(name: str, cfg: dict) -> dict:
    """Downloads ONE dataset, verifies it, and returns a small dict report
    (not the data itself) -- so nothing large from this dataset stays
    referenced once the function returns and Python garbage-collects it."""
    target = DATA_DIR / name
    target.mkdir(exist_ok=True)
    print(f"\n=== {name} ===")
    print(f"Source: {cfg['description']}")

    if cfg["kind"] == "competition":
        comp_name = cfg["kaggle_id"].split("/")[-1]
        cmd = ["kaggle", "competitions", "download", "-c", comp_name, "-p", str(target)]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", cfg["kaggle_id"], "-p", str(target), "--unzip"]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: download command exited with code {result.returncode}")
        print(f"  stderr: {result.stderr[-500:]}")

    # competitions download as a zip that isn't auto-unzipped by --unzip
    # (that flag only exists for `datasets download`) -- extract manually
    for item in target.glob("*.zip"):
        print(f"  Extracting {item.name}...")
        with zipfile.ZipFile(item, "r") as zf:
            zf.extractall(target)
        item.unlink()

    report = {"dataset": name, "files_found": [], "files_missing": [], "row_counts": {}}
    for fname in cfg["expected_files"]:
        fpath = target / fname
        if fpath.exists():
            report["files_found"].append(fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                n = sum(1 for _ in f) - 1
            report["row_counts"][fname] = n
        else:
            report["files_missing"].append(fname)

    del result  # explicit, though it would be collected anyway at function exit
    gc.collect()
    return report


if __name__ == "__main__":
    import pandas as pd
    reports = []
    for name, cfg in DATASETS.items():
        reports.append(download_and_verify(name, cfg))

    print("\n" + "=" * 60)
    print("ACQUISITION AUDIT REPORT")
    print("=" * 60)
    for r in reports:
        print(f"\n{r['dataset']}:")
        print(f"  Found:   {r['files_found']}")
        print(f"  Missing: {r['files_missing']}")
        for fname, n in r["row_counts"].items():
            print(f"  {fname}: {n:,} rows")

    pd.DataFrame(reports).to_csv(DATA_DIR / "acquisition_audit.csv", index=False)
    print(f"\nAudit saved to {DATA_DIR / 'acquisition_audit.csv'}")
    print("\nScript exiting now -- all memory from this run is released.")
