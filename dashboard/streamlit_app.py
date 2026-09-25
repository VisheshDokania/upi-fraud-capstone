"""Local Streamlit workbench for the existing fraud scoring API."""
from __future__ import annotations

import os
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("FRAUD_API", "http://127.0.0.1:8000").rstrip("/")
ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
DEMO_PATH = NB / "tabular_timesplit_report" / "demo_transactions.csv"
TABULAR_RESULTS = NB / "tabular_timesplit_report" / "timesplit_summary.csv"
GRAPH_RESULTS = NB / "graph_valsplit_report" / "graph_valsplit_summary.csv"
FUSION_RESULTS = NB / "fusion_report" / "fusion_summary.csv"
NLP_RESULTS = NB / "nlp_module_report" / "nlp_export_summary.csv"

st.set_page_config(
    page_title="UPI Fraud Intelligence | Capstone",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #172635; --muted: #657482; --line: #dbe2e7; --paper: #f4f6f7;
      --panel: #ffffff; --navy: #102b3d; --blue: #22617a;
      --low: #187457; --low-bg: #e9f5ef; --mid: #9a6700; --mid-bg: #fff4dc;
      --high: #b43732; --high-bg: #fff0ef;
    }
    html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { letter-spacing: -0.035em; color: var(--ink); }
    h1 { font-size: 2.05rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.38rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.08rem !important; font-weight: 700 !important; }
    p, label, li { color: var(--ink); }
    .eyebrow { color: var(--blue); font: 500 .73rem Consolas, monospace; letter-spacing: .12em; text-transform: uppercase; }
    .hero { background: var(--navy); color: #f4f7f8; padding: 1.55rem 1.8rem; border-radius: 8px;
      border-left: 4px solid #70a6ad; margin: .6rem 0 1.15rem; }
    .hero-title { color: #fff; font-size: 1.62rem; font-weight: 800; letter-spacing: -.035em; margin: .2rem 0 .25rem; }
    .hero-copy { color: #c1cdd3; font-size: .94rem; margin: 0; max-width: 880px; }
    .hero-kicker { color: #9fc8ca; font: 500 .7rem Consolas, monospace; letter-spacing: .14em; text-transform: uppercase; }
    .section-label { color: var(--muted); font: 500 .7rem Consolas, monospace; letter-spacing: .1em; text-transform: uppercase; margin: .2rem 0 .65rem; }
    .decision { border: 1px solid var(--line); border-left: 4px solid var(--blue); border-radius: 7px;
      background: var(--panel); padding: 1.1rem 1.25rem; margin: .4rem 0 .8rem; }
    .decision.low { border-left-color: var(--low); }
    .decision.medium { border-left-color: #c18413; }
    .decision.high { border-left-color: var(--high); }
    .decision-label { color: var(--muted); font: 500 .68rem Consolas, monospace; letter-spacing: .11em; text-transform: uppercase; }
    .decision-value { font-size: 1.55rem; font-weight: 800; letter-spacing: -.035em; margin-top: .3rem; }
    .risk-pill { display: inline-block; border-radius: 4px; padding: .22rem .5rem; font: 600 .72rem Consolas, monospace; letter-spacing: .07em; }
    .risk-pill.low { color: var(--low); background: var(--low-bg); }
    .risk-pill.medium { color: var(--mid); background: var(--mid-bg); }
    .risk-pill.high { color: var(--high); background: var(--high-bg); }
    .mono { font-family: Consolas, monospace; }
    .small-note { color: var(--muted); font-size: .82rem; line-height: 1.55; }
    .factor-up { color: var(--high); font: 600 .76rem Consolas, monospace; }
    .factor-down { color: var(--low); font: 600 .76rem Consolas, monospace; }
    .step { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: .8rem .9rem; min-height: 88px; }
    .step-n { color: var(--blue); font: 500 .68rem Consolas, monospace; }
    .step-t { color: var(--ink); font-weight: 700; margin-top: .35rem; font-size: .9rem; }
    .step-d { color: var(--muted); font-size: .77rem; margin-top: .18rem; line-height: 1.4; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: .8rem 1rem; }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { color: var(--ink); font-weight: 750; }
    div[data-testid="stTabs"] button { font-weight: 700; }
    .stButton > button, .stFormSubmitButton > button { border-radius: 5px; border: 1px solid var(--navy);
      background: var(--navy); color: #fff; font-weight: 700; padding: .55rem 1rem; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { border-color: var(--blue); background: var(--blue); color: #fff; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); border-radius: 7px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo():
    if not DEMO_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DEMO_PATH)


@st.cache_data(show_spinner=False)
def load_results(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def api_json(method, route, **kwargs):
    response = requests.request(method, f"{API}{route}", timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def api_error_message(exc):
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            return str(exc.response.json().get("detail", exc.response.text))
        except ValueError:
            return exc.response.text
    return str(exc)


def clean_features(row):
    features = {}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        features[key] = value
    return features


def display_value(value):
    if value is None or pd.isna(value):
        return "Not available"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def feature_name(name):
    # Keep IEEE-CIS identifiers visible: their detailed semantics are not defined here.
    if name == "TransactionAmt":
        return "TransactionAmt · transaction amount"
    return name.replace("_", " ")


def risk_recommendation(risk):
    if risk == "high":
        return "The configured alert threshold was reached. Manual review and additional verification are recommended before approval. This demo does not initiate either action."
    if risk == "medium":
        return "The score is near the configured alert threshold. Review the transaction details under your normal process; this demo does not initiate verification."
    return "The score is below the configured alert threshold. This is not a guarantee of safety; continue standard controls."


def render_shap_explanation(result, inputs):
    drivers = result.get("top_drivers") or []
    if not drivers:
        error = result.get("shap_error")
        if error:
            st.warning(f"A local SHAP explanation was unavailable: {error}")
        else:
            st.info("The API did not return feature contributions for this result.")
        return

    factors = pd.DataFrame(drivers)
    factors["shap"] = pd.to_numeric(factors["shap"], errors="coerce")
    factors = factors.dropna(subset=["shap"]).sort_values("shap", key=np.abs, ascending=False)
    if factors.empty:
        st.info("The API returned no usable SHAP contributions.")
        return

    st.markdown("#### Top contributing features")
    st.caption(
        "Signed TreeSHAP contributions for this prediction, ordered by absolute impact. "
        "Positive values move the model output toward fraud; negative values move it toward legitimate. "
        "Units are model-output units, not percentages or causal effects."
    )
    names = [feature_name(name) for name in factors["feature"]]
    colors = ["#b43732" if value > 0 else "#187457" if value < 0 else "#7d8991"
              for value in factors["shap"]]
    fig = go.Figure(go.Bar(
        x=factors["shap"], y=names, orientation="h", marker_color=colors,
        customdata=[display_value(inputs.get(name)) for name in factors["feature"]],
        hovertemplate="%{y}<br>SHAP: %{x:.4f}<br>Input: %{customdata}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#7d8991", line_width=1)
    fig.update_layout(
        height=max(240, 47 * len(factors)), margin=dict(l=8, r=12, t=12, b=22),
        xaxis_title="SHAP contribution (model-output units)", yaxis_title="",
        showlegend=False, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#172635", size=12),
        xaxis=dict(showgrid=True, gridcolor="#e8edf0", zeroline=False),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    positive = factors.loc[factors["shap"] > 0]
    negative = factors.loc[factors["shap"] < 0]
    toward_fraud, toward_legit = st.columns(2)
    with toward_fraud:
        st.markdown('<div class="section-label">Moved score toward fraud</div>', unsafe_allow_html=True)
        if positive.empty:
            st.markdown("No positive contribution among the API's top features.")
        for item in positive.itertuples(index=False):
            value = display_value(inputs.get(item.feature))
            st.markdown(
                f'<div class="factor-up">+ {escape(feature_name(item.feature))} · +{item.shap:.4f}</div>'
                f'<div class="small-note">Input value: {escape(value)}</div>',
                unsafe_allow_html=True,
            )
    with toward_legit:
        st.markdown('<div class="section-label">Moved score toward legitimate</div>', unsafe_allow_html=True)
        if negative.empty:
            st.markdown("No negative contribution among the API's top features.")
        for item in negative.itertuples(index=False):
            value = display_value(inputs.get(item.feature))
            st.markdown(
                f'<div class="factor-down">− {escape(feature_name(item.feature))} · {item.shap:.4f}</div>'
                f'<div class="small-note">Input value: {escape(value)}</div>',
                unsafe_allow_html=True,
            )


def get_risk_class(risk):
    return risk if risk in {"low", "medium", "high"} else "unknown"


def record_analysis(entry):
    history = st.session_state.setdefault("analysis_history", [])
    history.insert(0, entry)
    del history[20:]
    counts = st.session_state.setdefault(
        "analysis_counts", {"total": 0, "alerts": 0, "low": 0, "medium": 0, "high": 0}
    )
    counts["total"] += 1
    counts["alerts"] += int(entry["prediction"] == "Alert")
    if entry["risk"] in {"low", "medium", "high"}:
        counts[entry["risk"]] += 1


def render_transaction_result():
    analysis = st.session_state.get("last_transaction_analysis")
    if not analysis:
        return
    result = analysis["result"]
    inputs = analysis["inputs"]
    risk = get_risk_class(result.get("risk"))
    if risk == "unknown":
        st.error("The API returned an unrecognized risk band; no classification is displayed.")
        return
    probability = float(result["fraud_probability"])
    threshold = float(result["threshold"])
    alert = probability >= threshold
    label = "Fraud alert" if alert else "No fraud alert"

    st.markdown("---")
    st.markdown('<div class="eyebrow">Transaction analysis</div>', unsafe_allow_html=True)
    score_col, why_col = st.columns([0.8, 1.6], gap="large")
    with score_col:
        st.markdown(
            f'<div class="decision {risk}">'
            f'<div class="decision-label">Fraud risk</div>'
            f'<div class="decision-value">{risk.upper()} '
            f'<span class="risk-pill {risk}">{probability:.1%} score</span></div>'
            f'<div class="small-note" style="margin-top:.55rem">Prediction: {label}</div>'
            f'</div>', unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0), text=f"Estimated model fraud probability · {probability:.1%}")
        st.caption(f"Alert threshold: {threshold:.1%} · selected on the validation period.")
        st.markdown("**Recommended action**")
        st.markdown(f'<div class="small-note">{risk_recommendation(risk)}</div>', unsafe_allow_html=True)
    with why_col:
        st.markdown("### Why did the model make this decision?")
        render_shap_explanation(result, inputs)

    st.markdown("### Transaction details")
    details = [
        ("Amount", display_value(inputs.get("TransactionAmt"))),
        ("ProductCD", display_value(inputs.get("ProductCD"))),
        ("Card4", display_value(inputs.get("card4"))),
        ("Card6", display_value(inputs.get("card6"))),
        ("DeviceType", display_value(inputs.get("DeviceType"))),
        ("Dataset time field", display_value(analysis.get("transaction_dt"))),
    ]
    detail_cols = st.columns(len(details))
    for col, (label, value) in zip(detail_cols, details):
        with col:
            st.metric(label, value)
    st.caption("TransactionDT is shown as dataset context only; it is excluded from the fitted model's 431 inputs.")
    if analysis.get("reference_label") is not None:
        st.caption(
            f"Reference label in the held-out demonstration row: "
            f"{'fraud' if int(analysis['reference_label']) == 1 else 'legitimate'}. "
            "This label was not sent to the scoring API."
        )
    st.caption(f"Model: {result.get('model', 'unknown')} · top five returned contributions are shown.")


def render_transaction_tab(demo, api_online):
    if demo.empty:
        st.error("The held-out demo sample is missing. Run scripts/03b_tabular_timesplit.py to create it.")
        return
    st.markdown("### Analyze a transaction")
    st.markdown(
        '<div class="small-note">Choose a held-out IEEE-CIS demonstration row, or adjust its supported fields. '
        'The fitted model receives the available feature fields from that row.</div>',
        unsafe_allow_html=True,
    )
    if not api_online:
        st.error("Scoring API unavailable. Start it from the project root with `python -m uvicorn app.api:app --host 127.0.0.1 --port 8000`.")

    def option_label(index):
        row = demo.loc[index]
        return f"Transaction {row.get('TransactionID', index)} · amount {display_value(row.get('TransactionAmt'))}"

    selected = st.selectbox("Held-out sample transaction", demo.index.tolist(), format_func=option_label)
    source = demo.loc[selected]
    with st.form("transaction_analysis_form"):
        amount = st.number_input(
            "Transaction amount", min_value=0.0, value=float(source.get("TransactionAmt", 0) or 0),
            step=1.0, format="%.2f", key=f"amount_{selected}",
        )
        with st.expander("Edit additional available inputs", expanded=False):
            st.caption("Choices are drawn from this demo sample. Unchanged fields retain the selected row's values.")
            override_values = {"ProductCD": "Product / transaction class", "card4": "Card network", "card6": "Card type", "DeviceType": "Device type"}
            overrides = {}
            override_cols = st.columns(2)
            for position, (column, label) in enumerate(override_values.items()):
                if column not in demo.columns:
                    continue
                values = sorted(demo[column].dropna().astype(str).unique().tolist())
                current = source.get(column)
                if not values:
                    continue
                default = str(current) if pd.notna(current) and str(current) in values else values[0]
                with override_cols[position % 2]:
                    overrides[column] = st.selectbox(
                        label, values, index=values.index(default), key=f"{column}_{selected}"
                    )
        submitted = st.form_submit_button("Analyze transaction", disabled=not api_online)

    if submitted:
        model_columns = [column for column in demo.columns
                         if column not in {"isFraud", "TransactionID", "TransactionDT"}]
        model_row = source[model_columns].copy()
        if "TransactionAmt" in model_row.index:
            model_row["TransactionAmt"] = amount
        for column, value in overrides.items():
            if column in model_row.index:
                model_row[column] = value
        payload_features = clean_features(model_row)
        try:
            response = api_json("POST", "/score/transaction", json={"features": payload_features})
            analysis = {
                "result": response,
                "inputs": payload_features,
                "transaction_id": display_value(source.get("TransactionID")),
                "transaction_dt": source.get("TransactionDT"),
                "reference_label": source.get("isFraud"),
            }
            st.session_state["last_transaction_analysis"] = analysis
            record_analysis({
                "transaction": analysis["transaction_id"], "risk": response.get("risk", "unknown"),
                "score": float(response.get("fraud_probability", 0)),
                "prediction": "Alert" if float(response.get("fraud_probability", 0)) >= float(response.get("threshold", 1)) else "Below threshold",
                "amount": amount,
            })
            st.rerun()
        except (requests.RequestException, ValueError, KeyError) as exc:
            st.error(f"Transaction scoring failed: {api_error_message(exc)}")
    render_transaction_result()


def render_text_tab(api_online):
    st.markdown("### Message and URL screening")
    st.markdown(
        '<div class="small-note">Independent text classifiers score suspicious content. Their current API returns a '
        'probability and risk band; it does not expose per-token explanations.</div>', unsafe_allow_html=True,
    )
    kind = st.radio("Input type", ["sms", "url"], horizontal=True, format_func=lambda x: x.upper())
    placeholder = "Paste a payment-request message or URL"
    text = st.text_area("Content to screen", placeholder=placeholder, height=120)
    if st.button("Screen content", disabled=not api_online, key="score_text"):
        if not text.strip():
            st.warning("Enter a message or URL before screening.")
        else:
            try:
                result = api_json("POST", "/score/text", json={"text": text, "kind": kind})
                risk = get_risk_class(result.get("risk"))
                probability = float(result["suspicious_probability"])
                left, right = st.columns([1, 2])
                with left:
                    st.markdown(
                        f'<div class="decision {risk}"><div class="decision-label">Suspicion risk</div>'
                        f'<div class="decision-value">{risk.upper()}</div></div>', unsafe_allow_html=True,
                    )
                    st.metric("Suspicious probability", f"{probability:.1%}")
                    st.progress(min(max(probability, 0), 1))
                with right:
                    st.info("Text-level explanation is not available from the current model/API; no word-level reasons are inferred.")
                    st.caption("Displayed text risk bands follow the API's fixed 0.50 threshold (medium begins at half that value); this is distinct from the tabular model's validation-selected threshold.")
            except (requests.RequestException, ValueError, KeyError) as exc:
                st.error(f"Text scoring failed: {api_error_message(exc)}")


def render_graph_tab(api_online):
    st.markdown("### Elliptic graph risk")
    st.markdown(
        '<div class="small-note">This view uses the saved fusion output and graph edges. '
        'Neighbour scores describe linked nodes; they are not causal explanations.</div>', unsafe_allow_html=True,
    )
    node = st.number_input("Elliptic node index", min_value=0, value=0, step=1)
    if st.button("Inspect node", disabled=not api_online, key="score_graph"):
        try:
            result = api_json("GET", f"/graph/node/{int(node)}")
            probability = float(result["fused_risk"])
            left, right = st.columns([1, 2])
            with left:
                st.metric(f"Node {int(node)} · fused risk score", f"{probability:.1%}")
                st.progress(min(max(probability, 0), 1))
            with right:
                neighbors = pd.DataFrame(result.get("neighbours", []))
                if not neighbors.empty:
                    neighbors = neighbors.sort_values("fused_risk", ascending=False)
                    st.dataframe(neighbors, hide_index=True, width="stretch")
                else:
                    st.info("No neighboring nodes were returned for this node.")
            st.caption("Graph view does not provide a per-node GNNExplainer result in the current API.")
        except (requests.RequestException, ValueError, KeyError) as exc:
            st.error(f"Graph lookup failed: {api_error_message(exc)}")


def render_session_overview():
    history = st.session_state.get("analysis_history", [])
    counts = st.session_state.get(
        "analysis_counts", {"total": 0, "alerts": 0, "low": 0, "medium": 0, "high": 0}
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analyses this session", counts["total"])
    c2.metric("Alert threshold reached", counts["alerts"])
    c3.metric("Medium risk", counts["medium"])
    c4.metric("Low risk", counts["low"])
    st.caption("Session-only counts from transaction analyses submitted in this browser session; not stored or aggregated across users.")
    if history:
        st.markdown("#### Recent transaction analyses")
        recent = pd.DataFrame(history)
        recent = recent.rename(columns={"transaction": "Demo transaction", "risk": "Risk band",
                                        "score": "Model score", "prediction": "Threshold result",
                                        "amount": "Amount"})
        recent["Model score"] = recent["Model score"].map(lambda p: f"{p:.1%}")
        st.dataframe(recent, hide_index=True, width="stretch")


def render_model_tab(api_online, health):
    st.markdown("### Model card")
    tabular = load_results(str(TABULAR_RESULTS))
    demo = load_demo()
    if not tabular.empty:
        selected = tabular.sort_values("val_pr_auc", ascending=False).iloc[0]
        model_name = str(selected["model"])
        threshold = float(selected["threshold"])
        model_metrics = selected
    else:
        model_name, threshold, model_metrics = "Unavailable", None, None

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### Tabular transaction model")
        st.markdown(
            f"**Model**  \n{model_name.title()} Classifier  \n"
            "**Task**  \nBinary fraud classification  \n"
            f"**Input**  \nIEEE-CIS tabular fields ({max(0, len(demo.columns) - 3)} model features from the demo schema)  \n"
            "**Output**  \nFraud probability, risk band, threshold decision  \n"
            "**Explainability**  \nTreeSHAP for the selected transaction; signed top-five contributions  \n"
            f"**Threshold**  \n{threshold:.4f} selected on validation by F1" if threshold is not None else "Model bundle or evaluation summary unavailable."
        )
        if model_metrics is not None:
            st.caption("Evaluation values below are copied from the time-split summary CSV. Test-period metrics are descriptive; model selection uses validation PR-AUC.")
            metrics = pd.DataFrame([
                {"Period": "Validation", "PR-AUC": model_metrics.get("val_pr_auc"),
                 "ROC-AUC": model_metrics.get("val_roc_auc"), "F1": model_metrics.get("val_f1")},
                {"Period": "Fixed test", "PR-AUC": model_metrics.get("test_pr_auc"),
                 "ROC-AUC": model_metrics.get("test_roc_auc"), "F1": model_metrics.get("test_f1")},
            ])
            st.dataframe(metrics.set_index("Period").round(4), width="stretch")
        st.markdown("#### Other verified model results")
        graph = load_results(str(GRAPH_RESULTS))
        fusion = load_results(str(FUSION_RESULTS))
        nlp = load_results(str(NLP_RESULTS))
        if not graph.empty:
            with st.expander("Graph model · validation-selected results"):
                graph_columns = [column for column in
                                 ("model", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_f1")
                                 if column in graph.columns]
                st.caption("Source: notebooks/graph_valsplit_report/graph_valsplit_summary.csv")
                st.dataframe(graph[graph_columns].round(4), hide_index=True, width="stretch")
        if not fusion.empty:
            with st.expander("Elliptic fusion · fixed test comparison"):
                fusion_columns = [column for column in
                                  ("model", "test_pr_auc", "test_roc_auc", "test_f1", "threshold")
                                  if column in fusion.columns]
                st.caption("Source: notebooks/fusion_report/fusion_summary.csv")
                st.dataframe(fusion[fusion_columns].round(4), hide_index=True, width="stretch")
        if not nlp.empty:
            with st.expander("SMS and URL models"):
                nlp_columns = [column for column in
                               ("model", "test_pr_auc", "test_f1_at_0_5", "rows_after_dedupe")
                               if column in nlp.columns]
                st.caption("Source: notebooks/nlp_module_report/nlp_export_summary.csv")
                st.dataframe(nlp[nlp_columns].round(4), hide_index=True, width="stretch")
    with right:
        st.markdown("#### Data and task boundaries")
        st.markdown(
            "- Tabular metrics use the IEEE-CIS time-based evaluation.\n"
            "- Public transaction-level UPI data is unavailable; this project does not claim UPI validation.\n"
            "- The SMS and URL models return scores only; no token explanation is exposed.\n"
            "- The graph API returns fused node and neighbor scores; case-specific GNNExplainer output is not connected here.\n"
            "- Session activity is temporary and is not saved to a database."
        )
        loaded = health.get("models_loaded", []) if health else []
        st.markdown("**API model availability**")
        st.write(" · ".join(loaded) if loaded else "API status unavailable")
        st.markdown("**How a transaction is scored**")
        steps = [
            ("01", "Input", "Select a held-out sample row and edit supported fields."),
            ("02", "Feature mapping", "The API aligns fields to the trained model schema."),
            ("03", "Score", "LightGBM returns a fraud probability."),
            ("04", "Explain", "TreeSHAP reports the strongest signed local contributions."),
            ("05", "Decision", "The validation-selected threshold determines whether an alert is raised."),
        ]
        for number, title, description in steps:
            st.markdown(
                f'<div class="step"><div class="step-n">{number}</div>'
                f'<div class="step-t">{title}</div><div class="step-d">{description}</div></div>',
                unsafe_allow_html=True,
            )


def main():
    demo = load_demo()
    try:
        health = api_json("GET", "/health")
        api_online = health.get("status") == "ok"
    except (requests.RequestException, ValueError):
        health, api_online = {}, False

    loaded_models = ", ".join(health.get("models_loaded", [])) if api_online else "API offline"
    st.markdown('<div class="eyebrow">Fraud intelligence · capstone demonstration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Digital payments · tabular + graph + text</div>'
        '<div class="hero-title">UPI Fraud Detection & Explainability</div>'
        '<p class="hero-copy">A local research prototype for transaction risk scoring. Inspect the model score, '
        'the validation threshold, and the feature contributions behind each tabular decision.</p></div>',
        unsafe_allow_html=True,
    )
    status_col, context_col = st.columns([1, 3])
    with status_col:
        status_text = "API ONLINE" if api_online else "API OFFLINE"
        st.markdown(
            f'<span class="risk-pill {"low" if api_online else "high"}" style="font-size:.73rem">'
            f'{status_text}</span> <span class="small-note">{escape(loaded_models)}</span>', unsafe_allow_html=True,
        )
    with context_col:
        st.markdown(
            '<div class="small-note">Research evaluation uses IEEE-CIS, Elliptic and public text datasets. '
            'Public transaction-level UPI fraud data is unavailable; demo predictions are not UPI performance claims.</div>',
            unsafe_allow_html=True,
        )

    render_session_overview()
    transaction_tab, text_tab, graph_tab, model_tab = st.tabs(
        ["Transaction workbench", "SMS / URL screening", "Graph risk", "Model & evaluation"]
    )
    with transaction_tab:
        render_transaction_tab(demo, api_online)
    with text_tab:
        render_text_tab(api_online)
    with graph_tab:
        render_graph_tab(api_online)
    with model_tab:
        render_model_tab(api_online, health)


if __name__ == "__main__":
    main()
