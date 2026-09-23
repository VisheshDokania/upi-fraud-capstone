# Capstone report outline

1. Introduction - UPI growth and fraud (cite RBI annual report / NPCI stats with year), problem statement, objectives
2. Literature review - base paper (LightGBM + GraphSAGE gated ensemble, IEEE PuneCon 2025) + 12-15 papers: tabular GBDTs for fraud, Elliptic/Weber et al. 2019, GraphSAGE/GAT/GCN, SHAP, GNNExplainer, SMS/URL phishing detection. Table: paper | data | method | metric | gap
3. Datasets - IEEE-CIS (real), Elliptic (real), ULB credit card (real), PaySim (synthetic baseline), SMS Spam Collection, phishing URLs. Row counts from acquisition_audit.csv. Why UPI-level data is not public
4. Methodology - architecture diagram (tabular branch, graph branch, NLP side-channel, fusion, explainability, API)
5. Experimental setup - splits (time-based for IEEE-CIS, temporal for Elliptic), metrics (PR-AUC headline), threshold tuning on validation, hardware, seeds
6. Results
   - Table 1 tabular ablation (time split) + appendix: shuffled CV
   - Table 2 graph ablation (validation split)
   - Table 3 fusion: tabular-only vs graph-only vs avg/weighted/stacked/gated
   - Figure: F1 per Elliptic time step (dark-market shutdown dip)
   - Table 4 NLP modules (deduplicated)
   - Figures: SHAP global + one worked example; GNNExplainer subgraph
7. System demo - API endpoints, dashboard screenshots, latency per request
8. Discussion - what worked, when the graph helps, concept drift, limitations (no real UPI data, cross-domain datasets, offline not streaming)
9. Conclusion and future work - Kafka/Redis streaming (from base paper), federated learning across banks, Indian-language SMS
10. References (IEEE style) + Appendix (how to reproduce: commands, requirements)

## Likely viva questions
- Why is your data not UPI data? (no public source; base paper uses PaySim too)
- Why PR-AUC and not accuracy? (3.5% / ~10% positives)
- How did you avoid leakage? (time splits, validation-only tuning)
- Does the graph actually help over features alone? (Table 3)
- Why does performance drop after time step 43? (dark-market shutdown, concept drift)
- How would this run in real time? (API latency, streaming design)
