"""Run from the project root:  pytest -q
Fast checks that need no real data and no torch."""
import subprocess
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.text_features import clean_text, clean_url  # noqa: E402
from common_eval import best_f1_threshold, evaluate, make_graph_masks  # noqa: E402


def test_clean_text_keeps_link_signal():
    assert "urltoken" in clean_text("Update KYC at http://bit.ly/x now")


def test_clean_url_keeps_structure():
    assert clean_url("http://paypal.com.login-verify.ru/webscr") == "paypal com login verify ru webscr"


def test_threshold_and_metrics():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.clip(y * 0.6 + rng.random(2000) * 0.5, 0, 1)
    thr = best_f1_threshold(y, p)
    r = evaluate(y, p, thr)
    assert 0 < thr < 1 and r["pr_auc"] > 0.5 and r["positives"] == y.sum()


@pytest.mark.parametrize("script", ["03b_tabular_timesplit.py", "08_fusion_elliptic.py"])
def test_smoke_scripts_run(script, tmp_path):
    pytest.importorskip("xgboost")
    args = [sys.executable, str(ROOT / "scripts" / script), "--smoke-test"]
    if script.startswith("03b"):
        args += ["--models", "xgboost"]
    r = subprocess.run(args, capture_output=True, text=True, cwd=ROOT / "scripts", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]


def test_api_health_without_models(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.setenv("UPI_PROJECT_ROOT", str(tmp_path))
    import importlib
    import app.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.post("/score/text", json={"text": "hi", "kind": "sms"}).status_code == 503


def test_api_coerces_null_numeric_input(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.setenv("UPI_PROJECT_ROOT", str(tmp_path))
    import importlib
    import app.api as api
    importlib.reload(api)

    class Model:
        def predict_proba(self, X):
            assert np.isnan(X.loc[0, "amount"])
            assert np.isnan(X.loc[0, "count"])
            return np.array([[0.1, 0.9]])

    class Explainer:
        def shap_values(self, X, check_additivity=False):
            return np.zeros((1, X.shape[1]))

    monkeypatch.setattr(api, "STATE", {})
    from fastapi.testclient import TestClient
    with TestClient(api.app) as client:
        api.STATE["tabular"] = {
            "name": "fake", "model": Model(), "features": ["amount", "count"],
            "cat_cols": [], "threshold": 0.5, "needs_str_cats": False,
        }
        api.STATE["shap_explainer"] = Explainer()
        response = client.post("/score/transaction", json={
            "features": {"amount": None, "count": "not-a-number"}
        })
    assert response.status_code == 200
    assert response.json()["fraud_probability"] == pytest.approx(0.9)
    assert response.json()["shap_error"] is None


def test_05b_graph_masks_keep_periods_disjoint():
    y = np.array([1, -1, 0, 1, 0, 1, 0])
    steps = np.array([29, 30, 32, 33, 34, 35, 49])
    masks = make_graph_masks(y, steps)
    assert np.flatnonzero(masks["train"]).tolist() == [0]
    assert np.flatnonzero(masks["val"]).tolist() == [2]
    assert np.flatnonzero(masks["threshold"]).tolist() == [3, 4]
    assert np.flatnonzero(masks["test"]).tolist() == [5, 6]
    assert not masks["val"][1]  # unlabelled node is excluded


def test_fusion_threshold_uses_reserved_threshold_period():
    spec = importlib.util.spec_from_file_location(
        "fusion_module", ROOT / "scripts" / "08_fusion_elliptic.py"
    )
    fusion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion)
    steps = np.array([30, 31, 32, 33, 34])
    y = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.95, 0.2, 0.7, 0.6, 0.55])
    masks = fusion.masks(y, steps)
    threshold = fusion.tune_threshold(y, scores, masks["threshold"])
    expected = best_f1_threshold(y[masks["threshold"]], scores[masks["threshold"]])
    fit_period_threshold = best_f1_threshold(y[masks["fuser_fit"]], scores[masks["fuser_fit"]])
    assert threshold == expected
    assert threshold != fit_period_threshold
