# Local API latency

Measured loopback `/score/text` inference latency from the running local API.
The client reused one HTTP session; warm-up calls were excluded. The request-level measurements are in `latency_requests.csv`, and the summaries below are read from `latency_summary.csv`.

| Request group | Requests | p50 (ms) | p95 (ms) | Mean (ms) | Min (ms) | Max (ms) | Non-2xx |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 200 | 1.75 | 2.42 | 1.85 | 1.55 | 3.74 | 0 |
| sms | 100 | 1.76 | 2.45 | 1.87 | 1.58 | 3.74 | 0 |
| url | 100 | 1.75 | 2.35 | 1.84 | 1.55 | 3.33 | 0 |
