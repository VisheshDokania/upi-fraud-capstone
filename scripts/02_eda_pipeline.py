"""
Week 2 — Exploratory Data Analysis (memory-conscious version)
==================================================================
Each dataset's EDA runs in its own function, and the dataframe is
explicitly deleted + garbage-collected before the next dataset loads.
This matters most for PaySim (6.3M rows, ~470MB as a CSV, more once
parsed into a DataFrame with object dtypes) -- if it stays alive in
memory alongside IEEE-CIS, Elliptic, and Credit-Card-Fraud, that's
very likely what exhausted Colab's RAM.

Run:
    cd scripts
    python 02_eda_pipeline.py
"""

import gc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "notebooks" / "eda_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

report_lines = ["# EDA Summary Report\n"]


def class_imbalance_plot(y: pd.Series, name: str):
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    counts.plot(kind="bar", ax=ax, color=["#2E8B78", "#C0392B"])
    ax.set_title(f"{name}: class balance")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_class_balance.png", dpi=120)
    plt.close(fig)
    rate = counts.min() / counts.sum() * 100
    return rate, counts.to_dict()


def missingness_report(df: pd.DataFrame, name: str, top_n: int = 15):
    miss = df.isna().mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(6, 4))
    miss.plot(kind="barh", ax=ax, color="#0F2C4C")
    ax.set_title(f"{name}: top-{top_n} missing-value columns")
    ax.set_xlabel("fraction missing")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_missingness.png", dpi=120)
    plt.close(fig)
    return miss


def eda_ieee_cis():
    fpath = DATA_DIR / "ieee_cis" / "train_transaction.csv"
    if not fpath.exists():
        report_lines.append("## IEEE-CIS\n\n*Not downloaded yet.*\n")
        return
    df = pd.read_csv(fpath)
    rate, counts = class_imbalance_plot(df["isFraud"], "ieee_cis")
    miss = missingness_report(df, "ieee_cis")
    report_lines.append("## IEEE-CIS Fraud Detection\n")
    report_lines.append(f"- Rows: {len(df):,}, Columns: {df.shape[1]}")
    report_lines.append(f"- Fraud rate: {rate:.3f}% ({counts})")
    report_lines.append(f"- Top missing columns: {list(miss.index[:5])}")
    report_lines.append(f"- Mean TransactionAmt (fraud vs legit): "
                         f"{df[df.isFraud==1].TransactionAmt.mean():.2f} vs "
                         f"{df[df.isFraud==0].TransactionAmt.mean():.2f}\n")
    del df  # explicit -- this dataframe is the largest single object here
    gc.collect()


def eda_elliptic():
    ELLIPTIC_DIR = DATA_DIR / "elliptic" / "elliptic_bitcoin_dataset"
    feat_path = ELLIPTIC_DIR / "elliptic_txs_features.csv"
    class_path = ELLIPTIC_DIR / "elliptic_txs_classes.csv"
    edge_path = ELLIPTIC_DIR / "elliptic_txs_edgelist.csv"
    if not (feat_path.exists() and class_path.exists()):
        report_lines.append("## Elliptic\n\n*Not downloaded yet.*\n")
        return

    features = pd.read_csv(feat_path, header=None)
    classes = pd.read_csv(class_path)
    edges = pd.read_csv(edge_path)
    features.columns = ["txId", "timeStep"] + [f"f{i}" for i in range(features.shape[1] - 2)]
    merged = features.merge(classes, on="txId", how="left")

    labeled = merged[merged["class"] != "unknown"]
    rate, counts = class_imbalance_plot(labeled["class"], "elliptic")

    report_lines.append("## Elliptic (Bitcoin transaction graph)\n")
    report_lines.append(f"- Total nodes: {len(merged):,}, Edges: {len(edges):,}")
    report_lines.append(f"- Labeled nodes: {len(labeled):,} ({len(labeled)/len(merged)*100:.1f}% of total)")
    report_lines.append(f"- Illicit rate among labeled: {rate:.2f}% ({counts})")
    report_lines.append(f"- Time steps: {merged['timeStep'].nunique()}\n")

    del features, classes, edges, merged, labeled
    gc.collect()


def eda_creditcard():
    fpath = DATA_DIR / "creditcard_fraud" / "creditcard.csv"
    if not fpath.exists():
        report_lines.append("## Credit Card Fraud\n\n*Not downloaded yet.*\n")
        return
    df = pd.read_csv(fpath)
    rate, counts = class_imbalance_plot(df["Class"], "creditcard")
    report_lines.append("## Credit Card Fraud (ULB)\n")
    report_lines.append(f"- Rows: {len(df):,}")
    report_lines.append(f"- Fraud rate: {rate:.4f}% ({counts})\n")
    del df
    gc.collect()


def eda_paysim():
    """
    PaySim is 6.3M rows -- the most likely single cause of the Colab RAM
    exhaustion if it was still resident alongside the other three
    datasets. Reading only the columns actually needed, and specifying
    dtypes up front instead of letting pandas infer them, cuts memory
    roughly in half compared to a naive pd.read_csv().
    """
    fpath = DATA_DIR / "paysim" / "PS_20174392719_1491204439457_log.csv"
    if not fpath.exists():
        report_lines.append("## PaySim\n\n*Not downloaded yet.*\n")
        return

    dtypes = {
        "step": "int32", "type": "category", "amount": "float32",
        "nameOrig": "string", "oldbalanceOrg": "float32", "newbalanceOrig": "float32",
        "nameDest": "string", "oldbalanceDest": "float32", "newbalanceDest": "float32",
        "isFraud": "int8", "isFlaggedFraud": "int8",
    }
    df = pd.read_csv(fpath, dtype=dtypes)
    rate, counts = class_imbalance_plot(df["isFraud"], "paysim")
    report_lines.append("## PaySim (synthetic baseline)\n")
    report_lines.append(f"- Rows: {len(df):,}")
    report_lines.append(f"- Fraud rate: {rate:.3f}% ({counts})")
    report_lines.append(f"- Transaction types: {df['type'].value_counts().to_dict()}\n")
    del df
    gc.collect()


if __name__ == "__main__":
    sns.set_style("whitegrid")
    eda_ieee_cis()
    eda_elliptic()
    eda_creditcard()
    eda_paysim()

    with open(OUT_DIR / "eda_summary.md", "w") as f:
        f.write("\n".join(report_lines))

    print(f"EDA report written to {OUT_DIR / 'eda_summary.md'}")
    print(f"Plots written to {OUT_DIR}/")
