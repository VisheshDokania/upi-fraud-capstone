"""Export report tables and 300-DPI figures from verified run CSVs only."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
OUT = ROOT / "docs" / "report_assets"
DPI = 300


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required result CSV is missing: {path}")
    return pd.read_csv(path)


def write_table(name, frame, columns, source):
    table = frame.loc[:, columns].copy()
    table.to_csv(OUT / f"{name}.csv", index=False)
    headers = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [headers, separator]
    for record in table.itertuples(index=False, name=None):
        cells = []
        for value in record:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    markdown = (
        f"# {name.replace('_', ' ').title()}\n\n"
        f"Source CSV: `{source.relative_to(ROOT).as_posix()}`.\n\n"
        + "\n".join(rows)
        + "\n"
    )
    (OUT / f"{name}.md").write_text(markdown, encoding="utf-8")
    return table


def finish_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def grouped_pr_figure(frame, model_col, validation_col, test_col, title, filename,
                      val_label="Validation", test_label="Test"):
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, frame[validation_col], width, label=val_label, color="#3973AC")
    ax.bar(x + width / 2, frame[test_col], width, label=test_label, color="#E1812C")
    ax.set_xticks(x, frame[model_col], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC")
    ax.set_title(title)
    ax.legend()
    finish_figure(fig, filename)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab_path = NB / "tabular_timesplit_report" / "timesplit_summary.csv"
    graph_path = NB / "graph_valsplit_report" / "graph_valsplit_summary.csv"
    fusion_path = NB / "fusion_report" / "fusion_summary.csv"
    nlp_path = NB / "nlp_module_report" / "nlp_export_summary.csv"
    week3_path = NB / "ablation_report" / "ablation_summary.csv"
    week5_path = NB / "graph_ablation_report" / "graph_ablation_summary.csv"
    latency_path = ROOT / "docs" / "latency_summary.csv"

    tab = read_csv(tab_path)
    graph = read_csv(graph_path)
    fusion = read_csv(fusion_path)
    nlp = read_csv(nlp_path)
    week3 = read_csv(week3_path).rename(columns={"Unnamed: 0": "model"})
    week5 = read_csv(week5_path).rename(columns={"Unnamed: 0": "model"})
    latency = read_csv(latency_path)

    write_table("week3_shuffled_cv", week3,
                ["model", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall"],
                week3_path)
    write_table("tabular_timesplit", tab,
                ["model", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_f1",
                 "test_precision", "test_recall"], tab_path)
    write_table("week5_graph", week5,
                ["model", "roc_auc", "f1", "precision", "recall"], week5_path)
    write_table("graph_valsplit", graph,
                ["model", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_f1",
                 "test_precision", "test_recall"], graph_path)
    write_table("fusion_test", fusion,
                ["model", "test_pr_auc", "test_roc_auc", "test_f1", "test_precision",
                 "test_recall", "threshold"], fusion_path)
    write_table("nlp_cleaned_text", nlp,
                ["model", "rows_before", "conflicting_clean_texts_removed",
                 "rows_after_dedupe", "test_pr_auc", "test_f1_at_0_5"], nlp_path)
    write_table("api_latency", latency,
                ["group", "request_count", "p50_ms", "p95_ms", "mean_ms",
                 "min_ms", "max_ms", "non_2xx_count"], latency_path)

    grouped_pr_figure(tab, "model", "val_pr_auc", "test_pr_auc",
                      "IEEE-CIS time-split PR-AUC", "tabular_pr_auc.png")
    grouped_pr_figure(graph, "model", "val_pr_auc", "test_pr_auc",
                      "Elliptic graph PR-AUC", "graph_pr_auc.png")

    fusion_sorted = fusion.sort_values("test_pr_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(fusion_sorted["model"], fusion_sorted["test_pr_auc"], color="#4C956C")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test PR-AUC")
    ax.set_title("Elliptic fusion variants (descriptive test comparison)")
    ax.tick_params(axis="x", labelrotation=22)
    finish_figure(fig, "fusion_test_pr_auc.png")

    x = np.arange(len(nlp))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, nlp["test_pr_auc"], width, label="PR-AUC", color="#3973AC")
    ax.bar(x + width / 2, nlp["test_f1_at_0_5"], width, label="F1 at 0.5", color="#E1812C")
    ax.set_xticks(x, nlp["model"])
    ax.set_ylim(0, 1)
    ax.set_title("Cleaned-text model metrics")
    ax.legend()
    finish_figure(fig, "nlp_test_metrics.png")

    latency_plot = latency.loc[
        latency["group"].isin(["transaction", "sms", "url", "graph"])
    ].copy()
    x = np.arange(len(latency_plot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, latency_plot["p50_ms"], width, label="p50", color="#3973AC")
    ax.bar(x + width / 2, latency_plot["p95_ms"], width, label="p95", color="#E1812C")
    ax.set_xticks(x, latency_plot["group"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Local API scoring latency")
    ax.legend()
    finish_figure(fig, "api_latency.png")

    curve_files = {
        "xgboost": NB / "tabular_timesplit_report" / "pr_curve_xgboost.csv",
        "lightgbm": NB / "tabular_timesplit_report" / "pr_curve_lightgbm.csv",
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, path in curve_files.items():
        curve = read_csv(path)
        score = tab.loc[tab["model"] == model, "test_pr_auc"].iloc[0]
        ax.plot(curve["recall"], curve["precision"], label=f"{model} (AP={score:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="IEEE-CIS time-split test PR curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    finish_figure(fig, "tabular_test_pr_curves.png")

    (OUT / "README.md").write_text(
        "# Report assets\n\n"
        "Generated by `scripts/09_make_report_assets.py` from the cited run CSVs. "
        "CSV tables preserve source precision; Markdown tables are rounded for display. "
        "Figures are saved at 300 DPI. The fusion chart is a descriptive test comparison, "
        "not a test-based selection rule.\n",
        encoding="utf-8",
    )
    print(f"Report tables and figures written to {OUT}")


if __name__ == "__main__":
    main()
