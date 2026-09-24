# Three-minute demo script

## Before presenting

In two PowerShell terminals at the project root:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

```powershell
& .\.venv\Scripts\python.exe -m streamlit run dashboard\streamlit_app.py
```

Open the Streamlit URL shown in the second terminal. Keep the API terminal visible so request failures can be spotted quickly.

## Presentation

### 0:00–0:30 — Frame the problem

Explain that this capstone demonstrates real-time digital-payment risk scoring with tabular, graph, and text models. Public transaction-level UPI data is unavailable, so the experiments use the real IEEE-CIS, Elliptic, and SMS/phishing datasets described in the README; PaySim is synthetic and only a baseline.

### 0:30–1:10 — Transaction risk

Open **Transaction risk**, choose a held-out IEEE-CIS test transaction, and submit it. Point out the predicted risk and the top SHAP drivers. Change the transaction amount to show how the live request is rescored; explain that this is a dataset-backed demonstration, not a live UPI connection.

### 1:10–1:45 — SMS and URL screening

Open **SMS / URL check**. Score the prefilled payment-request message, then switch to URL and submit a suspicious-looking example. Explain that the text pipelines clean messages and URLs before TF-IDF classification.

### 1:45–2:20 — Graph view

Open **Graph view**, enter a node index, and show the fused risk and neighboring nodes. Explain that the graph branch uses Elliptic transaction links and that its score is combined with the node’s own features by the configured gate.

### 2:20–3:00 — Results and caveats

Open **Results** and point to the tabular time-split, graph validation-split, and fusion tables. Explain that thresholds are tuned on reserved validation periods and headline comparisons use PR-AUC. Close by noting that the strongest fusion test PR-AUC in this run came from tabular-only; the gated model remains the graph demo output, and the demo should not imply measured UPI performance.
