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
      color-scheme: dark;
      --ink: #e8eff3; --muted: #a8bbc7; --line: #324957; --paper: #0b1720;
      --panel: #122430; --navy: #1b3c4d; --blue: #67afc2;
      --low: #72d0a6; --low-bg: #123b32; --mid: #f3c76b; --mid-bg: #44361c;
      --high: #ff938d; --high-bg: #492622;
    }
    html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
      color: var(--ink) !important;
      caret-color: var(--ink);
      background-color: var(--panel) !important;
      border-color: var(--line) !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
      color: var(--muted) !important;
      opacity: 1;
    }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { letter-spacing: -0.035em; color: var(--ink); }
    h1 { font-size: 2.05rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.38rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.08rem !important; font-weight: 700 !important; }
    p, label, li { color: var(--ink); }
    a { color: var(--blue); }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted); }
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
    .step { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: .8rem .9rem; min-height: 88px; }
    .step-n { color: var(--blue); font: 500 .68rem Consolas, monospace; }
    .step-t { color: var(--ink); font-weight: 700; margin-top: .35rem; font-size: .9rem; }
    .step-d { color: var(--muted); font-size: .77rem; margin-top: .18rem; line-height: 1.4; }
    div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: .9rem 1rem; }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { color: var(--muted) !important; }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { color: var(--ink) !important; font-weight: 750; }
    div[data-testid="stTabs"] button { font-weight: 700; color: var(--muted) !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--ink) !important; }
    .stButton > button, .stFormSubmitButton > button { border-radius: 5px; border: 1px solid var(--navy);
      background: var(--navy); color: #fff; font-weight: 700; padding: .55rem 1rem; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { border-color: var(--blue); background: var(--blue); color: #fff; }
    .stButton > button, .stFormSubmitButton > button,
    .stButton > button *, .stFormSubmitButton > button * {
      color: #f4f7f8 !important;
      -webkit-text-fill-color: #f4f7f8 !important;
    }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
    [data-testid="stTextInput"] label, [data-testid="stTextArea"] label,
    [data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label,
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label,
    [data-testid="stSlider"] label, [data-testid="stMultiSelect"] label,
    [data-testid="stDateInput"] label, [data-testid="stTimeInput"] label {
      color: var(--ink) !important;
    }
    [data-testid="stWidgetLabel"] { margin-bottom: .35rem; }
    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stMultiSelect"] [data-baseweb="select"],
    [data-testid="stDateInput"] [data-baseweb="input"],
    [data-testid="stTimeInput"] [data-baseweb="input"] {
      background-color: var(--panel) !important;
      border-color: var(--line) !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stMultiSelect"] [data-baseweb="select"] *,
    [data-testid="stDateInput"] [data-baseweb="input"] *,
    [data-testid="stTimeInput"] [data-baseweb="input"] * {
      color: var(--ink) !important;
      -webkit-text-fill-color: var(--ink) !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] svg,
    [data-testid="stMultiSelect"] [data-baseweb="select"] svg { fill: var(--muted); }
    [role="listbox"], [role="option"], [role="listbox"] * {
      color: var(--ink) !important;
      background-color: var(--panel) !important;
    }
    [data-testid="stRadio"] [role="radio"],
    [data-testid="stCheckbox"] [role="checkbox"] { border-color: var(--muted); }
    [data-testid="stExpander"] details {
      background-color: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
      color: var(--ink) !important;
    }
    [data-testid="stExpander"] summary { padding: .55rem .75rem; }
    [data-testid="stExpander"] summary:hover { background-color: #1a3341; }
    [data-testid="stAlert"], [data-testid="stAlert"] * {
      color: var(--ink) !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
      color: var(--ink) !important;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    [data-testid="stDataFrame"] canvas {
      color-scheme: dark;
    }
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [role="gridcell"] { color: var(--ink) !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--panel); border-color: var(--line); border-radius: 7px; }
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


UPI_MODEL_PATH = ROOT / "models" / "upi_synthetic_model.pkl"
UPI_COEFFICIENTS_PATH = NB / "upi_synthetic_report" / "upi_synthetic_coefficients.csv"
UPI_PRESETS = {
    "collect": {
        "amount": 18_000.0, "transaction_type": "P2P",
        "initiation_mode": "Collect Request", "merchant_mcc": None,
        "failed_pin_attempts": 1, "sender_psp": "oksbi", "receiver_psp": "okaxis",
        "transaction_status": "SUCCESS", "receiver_is_new": True,
        "receiver_age_hours": 2.0, "merchant_vpa_age_hours": 12.0,
        "merchant_transactions_1h": 0, "sender_transactions_10m": 1,
        "small_failed_transactions_1h": 0, "amount_to_sender_average": 8.0,
    },
    "qr": {
        "amount": 2_400.0, "transaction_type": "P2M", "initiation_mode": "Scan QR",
        "merchant_mcc": 5411.0, "failed_pin_attempts": 0,
        "sender_psp": "okhdfcbank", "receiver_psp": "ybl",
        "transaction_status": "SUCCESS", "receiver_is_new": True,
        "receiver_age_hours": 1.0, "merchant_vpa_age_hours": 3.0,
        "merchant_transactions_1h": 8, "sender_transactions_10m": 1,
        "small_failed_transactions_1h": 0, "amount_to_sender_average": 3.5,
    },
    "velocity": {
        "amount": 22_000.0, "transaction_type": "P2P",
        "initiation_mode": "Contact/Phone Number", "merchant_mcc": None,
        "failed_pin_attempts": 3, "sender_psp": "okicici", "receiver_psp": "oksbi",
        "transaction_status": "SUCCESS", "receiver_is_new": False,
        "receiver_age_hours": 4_000.0, "merchant_vpa_age_hours": 2_160.0,
        "merchant_transactions_1h": 0, "sender_transactions_10m": 4,
        "small_failed_transactions_1h": 3, "amount_to_sender_average": 18.0,
    },
}
UPI_FORM_DEFAULTS = {
    "amount": 250.0, "transaction_type": "P2P",
    "initiation_mode": "Contact/Phone Number", "merchant_mcc": None,
    "failed_pin_attempts": 0, "sender_psp": "okaxis", "receiver_psp": "ybl",
    "transaction_status": "SUCCESS", "receiver_is_new": False,
    "receiver_age_hours": 2_400.0, "merchant_vpa_age_hours": 2_160.0,
    "merchant_transactions_1h": 2, "sender_transactions_10m": 1,
    "small_failed_transactions_1h": 0, "amount_to_sender_average": 1.0,
}


@st.cache_resource(show_spinner=False)
def load_upi_artifacts():
    import pickle

    if not UPI_MODEL_PATH.exists():
        raise FileNotFoundError(
            "UPI synthetic model is missing. Run the UPI generator and training commands from the README."
        )
    if not UPI_COEFFICIENTS_PATH.exists():
        raise FileNotFoundError(
            "Exported UPI coefficient table is missing. Run the UPI training command from the README."
        )
    with UPI_MODEL_PATH.open("rb") as model_file:
        bundle = pickle.load(model_file)
    coefficients = pd.read_csv(UPI_COEFFICIENTS_PATH)
    required = {"feature", "log_odds_coefficient"}
    if not required.issubset(coefficients.columns):
        raise ValueError("UPI coefficient CSV does not have the expected columns")
    preprocessor = bundle["pipeline"].named_steps["preprocess"]
    model_features = preprocessor.get_feature_names_out()
    if set(model_features) != set(coefficients["feature"]):
        raise ValueError("UPI model and exported coefficient features do not match")
    return bundle, coefficients


def _upi_category_options(bundle, category):
    category_names = bundle["categorical_features"]
    category_index = category_names.index(category)
    encoded = bundle["pipeline"].named_steps["preprocess"].named_transformers_["categorical"]
    levels = encoded.named_steps["onehot"].categories_[category_index]
    return [str(level) for level in levels if str(level) != "Unknown"]


def _apply_upi_preset(name):
    for field, value in UPI_PRESETS[name].items():
        st.session_state[f"upi_{field}"] = value
    st.session_state["upi_loaded_preset"] = name


def _upi_feature_label(encoded_name, categorical_features):
    if encoded_name.startswith("numeric__"):
        raw_name = encoded_name.removeprefix("numeric__")
        labels = {
            "amount": "Transaction amount",
            "failed_pin_attempts": "Failed PIN attempts",
            "receiver_is_new": "New receiver indicator",
            "receiver_age_hours": "Receiver age (hours)",
            "merchant_vpa_age_hours": "Merchant VPA age (hours)",
            "merchant_transactions_1h": "Merchant transactions in one hour",
            "sender_transactions_10m": "Sender transactions in ten minutes",
            "small_failed_transactions_1h": "Small failed transactions in one hour",
            "amount_to_sender_average": "Amount / sender average",
            "transaction_hour_utc": "Transaction hour (UTC)",
            "transaction_dayofweek_utc": "Transaction weekday (UTC)",
        }
        return labels.get(raw_name, raw_name.replace("_", " "))
    raw_name = encoded_name.removeprefix("categorical__")
    for category in sorted(categorical_features, key=len, reverse=True):
        prefix = f"{category}_"
        if raw_name.startswith(prefix):
            value = raw_name.removeprefix(prefix)
            label = {
                "transaction_type": "Transaction type",
                "initiation_mode": "Initiation mode",
                "merchant_mcc": "Merchant MCC",
                "transaction_status": "Transaction status",
                "sender_psp": "Sender PSP",
                "receiver_psp": "Receiver PSP",
            }.get(category, category.replace("_", " "))
            return f"{label}: {value}"
    return raw_name.replace("_", " ")


def _upi_row_from_form(values):
    from upi_synthetic.train_model import prepare_features

    now = pd.Timestamp.now(tz="UTC")
    row = pd.DataFrame([{
        "transaction_id": "UPI-DEMO-INPUT",
        "timestamp": now,
        "amount": float(values["amount"]),
        "sender_vpa": f"demo.sender@{values['sender_psp']}",
        "receiver_vpa": f"demo.receiver@{values['receiver_psp']}",
        "transaction_type": values["transaction_type"],
        "initiation_mode": values["initiation_mode"],
        "merchant_mcc": values["merchant_mcc"] if values["transaction_type"] == "P2M" else None,
        "failed_pin_attempts": int(values["failed_pin_attempts"]),
        "transaction_status": values["transaction_status"],
        "receiver_is_new": int(values["receiver_is_new"]),
        "receiver_age_hours": float(values["receiver_age_hours"]),
        "merchant_vpa_age_hours": (
            float(values["merchant_vpa_age_hours"])
            if values["transaction_type"] == "P2M" else np.nan
        ),
        "merchant_transactions_1h": (
            int(values["merchant_transactions_1h"])
            if values["transaction_type"] == "P2M" else 0
        ),
        "sender_transactions_10m": int(values["sender_transactions_10m"]),
        "small_failed_transactions_1h": int(values["small_failed_transactions_1h"]),
        "amount_to_sender_average": float(values["amount_to_sender_average"]),
    }])
    features, _ = prepare_features(row)
    return features


def _score_upi_form(bundle, coefficients, values):
    features = _upi_row_from_form(values).reindex(columns=bundle["features"])
    pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(features)[:, 1][0])
    preprocessor = pipeline.named_steps["preprocess"]
    encoded = preprocessor.transform(features)
    encoded = encoded.toarray()[0] if hasattr(encoded, "toarray") else np.asarray(encoded)[0]
    names = preprocessor.get_feature_names_out()
    coefficient_map = coefficients.set_index("feature")["log_odds_coefficient"]
    weights = coefficient_map.reindex(names).to_numpy(dtype=float)
    contributions = encoded * weights
    active = np.flatnonzero(np.abs(contributions) > 1e-10)
    order = active[np.argsort(np.abs(contributions[active]))[::-1]][:8]
    drivers = pd.DataFrame({
        "feature": [_upi_feature_label(names[i], bundle["categorical_features"]) for i in order],
        "contribution_log_odds": contributions[order],
        "coefficient": weights[order],
    })
    return probability, drivers


def render_upi_live_scoring():
    st.info("Demonstration on synthetic UPI-like data modeling documented fraud typologies.")
    st.caption(
        "This separate synthetic model is not connected to the existing fraud API. "
        "Its prediction is a demonstration on generated scenarios, not real UPI performance."
    )
    try:
        bundle, coefficients = load_upi_artifacts()
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        st.error(str(exc))
        return

    sender_psps = _upi_category_options(bundle, "sender_psp")
    receiver_psps = _upi_category_options(bundle, "receiver_psp")
    mcc_values = _upi_category_options(bundle, "merchant_mcc")
    mcc_options = [None]
    for value in mcc_values:
        if value != "Unknown":
            try:
                mcc_options.append(float(value))
            except ValueError:
                continue
    for field, default in UPI_FORM_DEFAULTS.items():
        if field == "sender_psp" and default not in sender_psps:
            default = sender_psps[0]
        if field == "receiver_psp" and default not in receiver_psps:
            default = receiver_psps[0]
        st.session_state.setdefault(f"upi_{field}", default)

    st.markdown("### Load a scenario")
    example_columns = st.columns(3)
    examples = [
        ("collect", "Collect request scam"),
        ("qr", "QR code spoofing"),
        ("velocity", "Velocity attack"),
    ]
    for column, (preset_key, label) in zip(example_columns, examples):
        with column:
            st.button(
                label, key=f"upi_example_{preset_key}",
                on_click=_apply_upi_preset, args=(preset_key,), use_container_width=True,
            )
    loaded_preset = st.session_state.get("upi_loaded_preset")
    if loaded_preset:
        st.caption(f"Loaded scenario: {dict(examples)[loaded_preset]}. Review or edit its fields, then score it.")

    st.markdown("### UPI transaction fields")
    with st.form("upi_live_scoring_form"):
        primary_left, primary_right = st.columns(2)
        with primary_left:
            amount = st.number_input(
                "Amount", min_value=0.0, step=100.0, key="upi_amount",
            )
            transaction_type = st.selectbox(
                "Transaction type", ["P2P", "P2M"], key="upi_transaction_type",
            )
            initiation_mode = st.selectbox(
                "Initiation mode",
                ["Collect Request", "Scan QR", "Contact/Phone Number"],
                key="upi_initiation_mode",
            )
            merchant_mcc = st.selectbox(
                "Merchant MCC", mcc_options, key="upi_merchant_mcc",
                format_func=lambda value: "Not applicable / unknown" if value is None else str(int(value)),
            )
        with primary_right:
            failed_pin_attempts = st.number_input(
                "Failed PIN attempts before success", min_value=0, max_value=20,
                step=1, key="upi_failed_pin_attempts",
            )
            sender_psp = st.selectbox("Sender PSP", sender_psps, key="upi_sender_psp")
            receiver_psp = st.selectbox("Receiver PSP", receiver_psps, key="upi_receiver_psp")
        with st.expander("Scenario context features used by this model", expanded=False):
            context_left, context_right = st.columns(2)
            with context_left:
                transaction_status = st.selectbox(
                    "Transaction status", ["SUCCESS", "FAILED"], key="upi_transaction_status",
                )
                receiver_is_new = st.checkbox("Receiver VPA is new to sender", key="upi_receiver_is_new")
                receiver_age_hours = st.number_input(
                    "Receiver VPA age (hours)", min_value=0.0, step=1.0,
                    key="upi_receiver_age_hours",
                )
                merchant_vpa_age_hours = st.number_input(
                    "Merchant VPA age (hours)", min_value=0.0, step=1.0,
                    key="upi_merchant_vpa_age_hours",
                )
            with context_right:
                merchant_transactions_1h = st.number_input(
                    "Transactions to merchant in one hour", min_value=0, step=1,
                    key="upi_merchant_transactions_1h",
                )
                sender_transactions_10m = st.number_input(
                    "Sender transactions in ten minutes", min_value=0, step=1,
                    key="upi_sender_transactions_10m",
                )
                small_failed_transactions_1h = st.number_input(
                    "Small failed transactions in one hour", min_value=0, step=1,
                    key="upi_small_failed_transactions_1h",
                )
                amount_to_sender_average = st.number_input(
                    "Amount / sender's average amount", min_value=0.0, step=0.5,
                    key="upi_amount_to_sender_average",
                )
        submitted = st.form_submit_button("Score UPI transaction", use_container_width=True)

    if submitted:
        values = {
            "amount": amount, "transaction_type": transaction_type,
            "initiation_mode": initiation_mode, "merchant_mcc": merchant_mcc,
            "failed_pin_attempts": failed_pin_attempts,
            "sender_psp": sender_psp, "receiver_psp": receiver_psp,
            "transaction_status": transaction_status, "receiver_is_new": receiver_is_new,
            "receiver_age_hours": receiver_age_hours,
            "merchant_vpa_age_hours": merchant_vpa_age_hours,
            "merchant_transactions_1h": merchant_transactions_1h,
            "sender_transactions_10m": sender_transactions_10m,
            "small_failed_transactions_1h": small_failed_transactions_1h,
            "amount_to_sender_average": amount_to_sender_average,
        }
        try:
            probability, drivers = _score_upi_form(bundle, coefficients, values)
            threshold = float(bundle["threshold"])
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"UPI model scoring failed: {exc}")
            return

        fraud_alert = probability >= threshold
        st.markdown("---")
        st.markdown("### UPI model result")
        score_col, decision_col = st.columns([1, 2])
        with score_col:
            st.metric("Fraud probability", f"{probability:.1%}")
            st.progress(probability, text="Synthetic model fraud probability")
        with decision_col:
            st.markdown("**Prediction**")
            if fraud_alert:
                st.error(f"Fraud alert · score meets the validation threshold ({threshold:.1%}).")
            else:
                st.success(f"Below fraud alert threshold ({threshold:.1%}).")
            st.caption(
                f"Model: {bundle['name']} · decision threshold selected on validation. "
                "This is a synthetic scenario model, not a production UPI decision."
            )

        st.markdown("#### Features driving this score")
        st.caption(
            "Local additive terms calculated from the saved model's transformed inputs and exported coefficients. "
            "Positive log-odds contributions raise the fraud score; negative contributions lower it. "
            "These are model contributions, not causal claims."
        )
        if drivers.empty:
            st.info("No non-zero coefficient contributions were returned for this input.")
        else:
            colors = ["#b43732" if value > 0 else "#187457"
                      for value in drivers["contribution_log_odds"]]
            fig = go.Figure(go.Bar(
                x=drivers["contribution_log_odds"], y=drivers["feature"],
                orientation="h", marker_color=colors,
                customdata=drivers["coefficient"],
                hovertemplate=("%{y}<br>Contribution: %{x:.4f} log-odds"
                               "<br>Exported coefficient: %{customdata:.4f}<extra></extra>"),
            ))
            fig.add_vline(x=0, line_color="#7d8991", line_width=1)
            fig.update_layout(
                height=max(260, 42 * len(drivers)),
                margin=dict(l=8, r=12, t=8, b=24),
                xaxis_title="Contribution to model log-odds",
                yaxis_title="", showlegend=False,
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                font=dict(family="Segoe UI, Arial, sans-serif", color="#172635", size=12),
                xaxis=dict(showgrid=True, gridcolor="#e8edf0"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.dataframe(
                drivers.rename(columns={
                    "feature": "Feature", "contribution_log_odds": "Contribution (log-odds)",
                    "coefficient": "Exported coefficient",
                }).round(4), hide_index=True, width="stretch",
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
    transaction_tab, text_tab, graph_tab, model_tab, upi_tab = st.tabs(
        ["Transaction workbench", "SMS / URL screening", "Graph risk", "Model & evaluation",
         "UPI Live Scoring"]
    )
    with transaction_tab:
        render_transaction_tab(demo, api_online)
    with text_tab:
        render_text_tab(api_online)
    with graph_tab:
        render_graph_tab(api_online)
    with model_tab:
        render_model_tab(api_online, health)
    with upi_tab:
        render_upi_live_scoring()


if __name__ == "__main__":
    main()
