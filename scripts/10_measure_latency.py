"""Measure local API latency for representative transaction, text, and graph calls."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
REQUEST_COLUMNS = ["request_id", "endpoint", "kind", "status_code", "elapsed_ms"]
SUMMARY_COLUMNS = [
    "group", "request_count", "p50_ms", "p95_ms", "mean_ms", "min_ms",
    "max_ms", "non_2xx_count",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100,
                        help="Measured calls per request group (default: 100).")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Unrecorded warm-up calls per request group (default: 5).")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--demo-csv",
        type=Path,
        default=ROOT / "notebooks" / "tabular_timesplit_report" / "demo_transactions.csv",
    )
    return parser.parse_args()


def request_groups(args):
    if not args.demo_csv.exists():
        raise FileNotFoundError(f"Demo transaction CSV not found: {args.demo_csv}")
    demo = pd.read_csv(args.demo_csv, nrows=1)
    if demo.empty:
        raise ValueError(f"Demo transaction CSV has no rows: {args.demo_csv}")
    # Exclude the target and transaction identifier from model inputs.
    features = demo.drop(columns=[c for c in ("isFraud", "TransactionID")
                                 if c in demo.columns]).iloc[0]
    features = {
        key: (None if pd.isna(value) else
              value.item() if isinstance(value, np.generic) else value)
        for key, value in features.items()
    }
    return [
        ("/score/transaction", "transaction", "POST",
         {"features": features}),
        ("/score/text", "sms", "POST",
         {"text": "Your payment is pending. Verify your account now.", "kind": "sms"}),
        ("/score/text", "url", "POST",
         {"text": "https://secure-account.example.invalid/verify", "kind": "url"}),
        ("/graph/node/0", "graph", "GET", None),
    ]


def send(session, base_url, group, timeout):
    endpoint, kind, method, payload = group
    start = time.perf_counter()
    response = session.request(
        method,
        f"{base_url.rstrip('/')}{endpoint}",
        params={"max_neighbours": 25} if kind == "graph" else None,
        json=payload,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response.status_code, elapsed_ms


def main():
    args = parse_args()
    if args.requests < 1 or args.warmup < 0:
        raise ValueError("--requests must be positive and --warmup cannot be negative")
    groups = request_groups(args)
    rows = []
    request_id = 0
    with requests.Session() as session:
        # Fail early with an actionable error if the local service is not running.
        try:
            health = session.get(f"{args.base_url.rstrip('/')}/health", timeout=args.timeout)
            health.raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(f"Local API is unavailable at {args.base_url}: {exc}") from exc

        for group in groups:
            for _ in range(args.warmup):
                send(session, args.base_url, group, args.timeout)
            for _ in range(args.requests):
                status, elapsed_ms = send(session, args.base_url, group, args.timeout)
                request_id += 1
                rows.append({
                    "request_id": request_id,
                    "endpoint": group[0],
                    "kind": group[1],
                    "status_code": status,
                    "elapsed_ms": elapsed_ms,
                })

    requests_frame = pd.DataFrame(rows, columns=REQUEST_COLUMNS)
    requests_path = ROOT / "docs" / "latency_requests.csv"
    summary_path = ROOT / "docs" / "latency_summary.csv"
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    requests_frame.to_csv(requests_path, index=False)

    summary_rows = []
    groups_for_summary = [("all", requests_frame)]
    groups_for_summary += [
        (label, requests_frame.loc[requests_frame["kind"] == kind])
        for label, kind in [("transaction", "transaction"), ("sms", "sms"),
                            ("url", "url"), ("graph", "graph")]
    ]
    for label, frame in groups_for_summary:
        values = frame["elapsed_ms"].to_numpy(dtype=np.float64)
        summary_rows.append({
            "group": label,
            "request_count": len(frame),
            "p50_ms": np.percentile(values, 50),
            "p95_ms": np.percentile(values, 95),
            "mean_ms": values.mean(),
            "min_ms": values.min(),
            "max_ms": values.max(),
            "non_2xx_count": int((~frame["status_code"].between(200, 299)).sum()),
        })
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    print(f"Measured {len(requests_frame)} calls ({args.requests} per request group).")
    print(f"Requests: {requests_path.relative_to(ROOT)}")
    print(f"Summary:  {summary_path.relative_to(ROOT)}")
    print(pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_string(index=False))


if __name__ == "__main__":
    main()
