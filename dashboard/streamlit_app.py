"""
Week 9 - Demo dashboard (new file). Talks to the FastAPI service.

Run (two terminals, from the project root):
    uvicorn app.api:app --port 8000
    streamlit run dashboard/streamlit_app.py
"""
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("FRAUD_API", "http://127.0.0.1:8000")
ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
st.title("Real-time Digital Payment Fraud Detection")
try:
    st.caption(f"API: {API} - models loaded: {requests.get(API + '/health', timeout=30).json()['models_loaded']}")
except Exception:
    st.error("API not reachable. Start it with: uvicorn app.api:app --port 8000")

tab1, tab2, tab3, tab4 = st.tabs(["Transaction risk", "SMS / URL check", "Graph view", "Results"])

with tab1:
    demo_path = NB / "tabular_timesplit_report" / "demo_transactions.csv"
    if demo_path.exists():
        demo = pd.read_csv(demo_path)
        i = st.selectbox("Pick a real held-out test transaction", demo.index,
                         format_func=lambda k: f"#{demo.loc[k, 'TransactionID']} amt {demo.loc[k, 'TransactionAmt']:.2f} (true label: {'FRAUD' if demo.loc[k, 'isFraud'] else 'legit'})")
        feats = demo.drop(columns=["isFraud"]).iloc[int(i)].dropna().to_dict()
    else:
        st.info("Run scripts/03b_tabular_timesplit.py to create demo transactions.")
        feats = {}
    amt = st.number_input("TransactionAmt", value=float(feats.get("TransactionAmt", 100.0)))
    feats["TransactionAmt"] = amt
    if st.button("Score transaction"):
        r = requests.post(API + "/score/transaction", json={"features": feats}, timeout=30).json()
        if "fraud_probability" in r:
            st.metric("Fraud probability", f"{r['fraud_probability']:.1%}", r["risk"].upper())
            if r["top_drivers"]:
                st.bar_chart(pd.DataFrame(r["top_drivers"]).set_index("feature")["shap"])
                st.caption("SHAP: positive pushes toward fraud, negative toward legit")
        else:
            st.error(r)

with tab2:
    kind = st.radio("Type", ["sms", "url"], horizontal=True)
    text = st.text_area("Paste a payment-request SMS or a link",
                        "Your KYC is pending. Update now at http://bit.ly/upi-kyc to avoid account block")
    if st.button("Check"):
        r = requests.post(API + "/score/text", json={"text": text, "kind": kind}, timeout=30).json()
        st.metric("Suspicious probability", f"{r.get('suspicious_probability', 0):.1%}", r.get("risk", "").upper())

with tab3:
    idx = st.number_input("Elliptic node index", min_value=0, value=0, step=1)
    if st.button("Show neighbourhood"):
        r = requests.get(f"{API}/graph/node/{int(idx)}", timeout=30).json()
        if "fused_risk" in r:
            st.metric("Fused risk (tabular + graph)", f"{r['fused_risk']:.1%}")
            st.dataframe(pd.DataFrame(r["neighbours"]))
        else:
            st.error(r)

with tab4:
    for label, p in [("Tabular (time split)", NB / "tabular_timesplit_report" / "timesplit_summary.csv"),
                     ("Graph (validation split)", NB / "graph_valsplit_report" / "graph_valsplit_summary.csv"),
                     ("Fusion", NB / "fusion_report" / "fusion_summary.csv")]:
        if p.exists():
            st.subheader(label)
            st.dataframe(pd.read_csv(p).round(4))
    img = NB / "fusion_report" / "f1_by_time_step.png"
    if img.exists():
        st.image(str(img))
