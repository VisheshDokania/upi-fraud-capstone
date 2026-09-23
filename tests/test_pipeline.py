"""Run from the project root:  pytest -q
Fast checks that need no real data and no torch."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.text_features import clean_text, clean_url  # noqa: E402
from common_eval import best_f1_threshold, evaluate  # noqa: E402


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
