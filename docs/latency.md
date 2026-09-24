# Local API latency

Measured against the local API at `http://127.0.0.1:8000` using one reused HTTP session. Five warm-up calls per group were excluded; 100 measured calls were made for each group. Transaction requests used the first row of `notebooks/tabular_timesplit_report/demo_transactions.csv`; text requests used representative SMS and URL inputs; graph requests used node 0. Measurements and summaries are in `latency_requests.csv` and `latency_summary.csv`.

| Request group | Requests | p50 (ms) | p95 (ms) | Mean (ms) | Min (ms) | Max (ms) | Non-2xx |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 400 | 2.3627 | 547.0106 | 135.7458 | 1.2994 | 636.3443 | 0 |
| transaction | 100 | 528.7472 | 605.3371 | 536.1008 | 480.4600 | 636.3443 | 0 |
| sms | 100 | 2.4976 | 4.3593 | 2.6408 | 1.5956 | 4.6554 | 0 |
| url | 100 | 1.8455 | 4.3541 | 2.4449 | 1.5477 | 5.6503 | 0 |
| graph | 100 | 1.6246 | 2.7898 | 1.7968 | 1.2994 | 4.2020 | 0 |
