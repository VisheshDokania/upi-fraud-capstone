"""
Week 9 - FastAPI scoring service (new file).

Endpoints
  GET  /health                 which models are loaded
  POST /score/transaction      {"features": {...IEEE-CIS columns...}} -> fraud probability + top SHAP drivers
  POST /score/text             {"text": "...", "kind": "sms"|"url"}   -> suspicious probability
  GET  /graph/node/{idx}       fused (tabular+graph) risk for an Elliptic node + its neighbours

Run from the project root (the folder that has scripts/, models/, notebooks/):
    uvicorn app.api:app --reload --port 8000
    open http://127.0.0.1:8000/docs  (interactive Swagger page - good for the demo)
"""
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scripts.common_eval import fix_string_categories

ROOT = Path(os.environ.get("UPI_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
MODEL_DIR = ROOT / "models"
FUSION_DIR = ROOT / "notebooks" / "fusion_report"
GRAPH_DIR = ROOT / "notebooks" / "graph_pipeline"

app = FastAPI(title="UPI / Digital Payment Fraud Detection API", version="1.0")
STATE: Dict[str, Any] = {}


def _load():
    STATE.clear()
    tab = MODEL_DIR / "tabular_winner.pkl"
    if tab.exists():
        with open(tab, "rb") as f:
            STATE["tabular"] = pickle.load(f)
    for key, fname in [("sms", "sms_spam.joblib"), ("url", "phishing_url.joblib")]:
        p = MODEL_DIR / fname
        if p.exists():
            import joblib
            STATE[key] = joblib.load(p)
    fused = FUSION_DIR / "fused_node_proba.npy"
    if fused.exists():
        STATE["fused"] = np.load(fused)
        STATE["edge_index"] = np.load(GRAPH_DIR / "elliptic_edge_index.npy")


@app.on_event("startup")
def startup():
    _load()


class TxIn(BaseModel):
    features: Dict[str, Any] = Field(..., description="IEEE-CIS feature name -> value; missing ones are treated as missing")


class TextIn(BaseModel):
    text: str
    kind: Literal["sms", "url"] = "sms"


def _risk_band(p, thr):
    return "high" if p >= thr else ("medium" if p >= thr / 2 else "low")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(k for k in STATE if k != "edge_index")}


@app.post("/score/transaction")
def score_transaction(tx: TxIn):
    b = STATE.get("tabular")
    if b is None:
        raise HTTPException(503, "tabular model not loaded - run scripts/03b_tabular_timesplit.py")
    row = pd.DataFrame([{c: tx.features.get(c, np.nan) for c in b["features"]}])
    for c in b["cat_cols"]:
        row[c] = pd.Categorical(row[c], categories=b["cat_levels"][c])
    m = b["model"]
    if b.get("needs_str_cats", False):
        row = fix_string_categories(row, b["cat_cols"])
    X = row
    p = float(m.predict_proba(X)[:, 1][0])
    drivers = []
    try:
        import shap
        sv = shap.TreeExplainer(m).shap_values(X)
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)[0]
        top = np.argsort(-np.abs(sv))[:5]
        drivers = [{"feature": b["features"][i], "shap": float(sv[i])} for i in top]
    except Exception:
        pass  # SHAP is optional for scoring
    return {"model": b["name"], "fraud_probability": p, "threshold": b["threshold"],
            "risk": _risk_band(p, b["threshold"]), "top_drivers": drivers}


@app.post("/score/text")
def score_text(t: TextIn):
    pipe = STATE.get(t.kind)
    if pipe is None:
        raise HTTPException(503, f"{t.kind} model not loaded - run scripts/06b_export_nlp_models.py")
    p = float(pipe.predict_proba([t.text])[:, 1][0])
    return {"kind": t.kind, "suspicious_probability": p, "risk": _risk_band(p, 0.5)}


@app.get("/graph/node/{idx}")
def graph_node(idx: int, max_neighbours: int = 25):
    fused = STATE.get("fused")
    if fused is None:
        raise HTTPException(503, "fusion output not found - run scripts/08_fusion_elliptic.py")
    if not 0 <= idx < len(fused):
        raise HTTPException(404, "node index out of range")
    ei = STATE["edge_index"]
    nb = np.unique(np.concatenate([ei[1][ei[0] == idx], ei[0][ei[1] == idx]]))[:max_neighbours]
    return {"node": idx, "fused_risk": float(fused[idx]),
            "neighbours": [{"node": int(n), "fused_risk": float(fused[n])} for n in nb]}
