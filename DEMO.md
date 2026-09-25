# Real Time UPI Fraud Detection - local demo

The five-tab Streamlit site is a local research demonstration. IEEE-CIS card fraud, Elliptic Bitcoin, SMS and URL tasks are distinct proxy datasets. The fifth tab uses generated UPI-like records; none of these results establishes fraud accuracy on observed UPI transactions.

## Before presenting

From the project root, with the datasets and fitted artifacts prepared as explained in [README.md](README.md), open two PowerShell terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\streamlit_app.py
```

Check `http://127.0.0.1:8000/health` and the Streamlit URL printed by the second terminal. UPI Live Scoring needs the separate `models/upi_synthetic_model.pkl` and `notebooks/upi_synthetic_report/upi_synthetic_coefficients.csv`; generate/train them with the two commands in README before opening that tab. The API does not score UPI scenarios. If an artifact is missing, say so rather than narrating a result that is not on screen.

## Five-tab presentation route

1. **Transaction workbench:** Select held-out transaction `3459510` (amount 59.636) for a high-risk illustration if it is in the local demo sample. Show the alert threshold, top signed TreeSHAP terms and the temporary session-history table. Contrast with `3459432` (amount 33.261) if available. These are IEEE-CIS card records, not UPI transactions. Rescoring after you alter a supported field does not change the source data.
2. **SMS / URL screening:** Score a KYC scam message and a benign message. Enter a suspicious-looking URL as text (do not visit it). The 0.50 text risk cutoff is advisory and distinct from the transaction model's validation-selected threshold. False positives are possible.
3. **Graph risk:** Enter Elliptic node `144392` to inspect a saved fused risk and neighboring nodes if the local fusion artifacts exist. Compare nodes `0` and `2` if present. Edges describe Bitcoin transaction links; neighbor scores are not causal explanations or real-time UPI links.
4. **Model & evaluation:** Point to LightGBM's 431 feature inputs and validation-to-later-test change (PR-AUC 0.7046 to 0.5698). Show graph/fusion/text tables if their local summary CSVs are present. Tabular-only has higher held-out Elliptic PR-AUC (0.7906) than the degree-aware gate (0.7435); that is an honest negative result, not a reason to select a model on the test set.
5. **UPI Live Scoring:** Read the synthetic-data warning first. Load the collect-request preset, point to initiation mode, PSP handle categories, failed PIN attempts and other UPI-like fields, then score it locally in Streamlit. Explain that generated labels and features co-vary by construction; do not present its high internal score as observed UPI performance.

Keep the boundary simple: interactive demo is not a bank integration or an automatic fraud block. The project still needs permissioned UPI labels and future-period operational evaluation.
