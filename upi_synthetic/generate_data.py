"""Generate a deterministic, explicitly synthetic UPI-like transaction dataset.

The rows and labels are artificial scenario demonstrations, not observed UPI activity.
Output defaults to an ignored notebooks report directory, never the protected data/ tree.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "notebooks" / "upi_synthetic_report" / "upi_synthetic_transactions.csv"
PSP_HANDLES = ("okaxis", "okhdfcbank", "okicici", "oksbi", "ybl", "ibl", "paytm")
MERCHANT_MCCS = ("5411", "5812", "5541", "4900", "5999", "5311")
TRANSACTION_TYPES = ("P2P", "P2M")
INITIATION_MODES = ("Collect Request", "Scan QR", "Contact/Phone Number")


def _vpa(prefix: str, account_id: int, handle: str) -> str:
    """Make a fake VPA-shaped identifier with no real person's details."""
    return f"{prefix}{account_id:06d}@{handle}"


def _assign(frame: pd.DataFrame, index: int, **values) -> None:
    for column, value in values.items():
        frame.at[index, column] = value


def build_dataset(rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Create baseline rows, then inject three observable fraud scenarios."""
    if rows < 200:
        raise ValueError("rows must be at least 200 so each scenario can be represented")

    rng = np.random.default_rng(seed)
    sender_count = max(100, rows // 5)
    merchant_count = max(30, rows // 40)
    sender_handles = rng.choice(PSP_HANDLES, size=sender_count)
    merchant_handles = rng.choice(PSP_HANDLES, size=merchant_count)
    sender_vpas = np.array([
        _vpa("user", i, sender_handles[i]) for i in range(sender_count)
    ], dtype=object)
    merchant_vpas = np.array([
        _vpa("shop", i, merchant_handles[i]) for i in range(merchant_count)
    ], dtype=object)

    sender_ids = rng.integers(0, sender_count, size=rows)
    receiver_ids = rng.integers(0, sender_count, size=rows)
    receiver_ids = np.where(receiver_ids == sender_ids, (receiver_ids + 1) % sender_count,
                            receiver_ids)
    p2m = rng.random(rows) < 0.28
    receiver_vpa = sender_vpas[receiver_ids].copy()
    merchant_ids = rng.integers(0, merchant_count, size=rows)
    receiver_vpa[p2m] = merchant_vpas[merchant_ids[p2m]]

    amounts = np.clip(rng.lognormal(mean=4.0, sigma=1.0, size=rows), 1.0, 20_000.0)
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    offsets = rng.integers(0, 90 * 24 * 60 * 60, size=rows)
    timestamps = start + pd.to_timedelta(offsets, unit="s")
    transaction_type = np.where(p2m, "P2M", "P2P")
    initiation_mode = np.empty(rows, dtype=object)
    p2p_rows, p2m_rows = np.flatnonzero(~p2m), np.flatnonzero(p2m)
    initiation_mode[p2p_rows] = rng.choice(
        INITIATION_MODES, size=len(p2p_rows), p=(0.18, 0.18, 0.64)
    )
    initiation_mode[p2m_rows] = rng.choice(
        INITIATION_MODES, size=len(p2m_rows), p=(0.04, 0.72, 0.24)
    )

    receiver_is_new = (rng.random(rows) < 0.08).astype("int8")
    receiver_age_hours = np.where(
        receiver_is_new == 1,
        rng.uniform(0.1, 72.0, size=rows),
        rng.uniform(72.0, 24.0 * 365 * 5, size=rows),
    )
    merchant_age = np.full(rows, np.nan, dtype="float64")
    merchant_age[p2m] = rng.uniform(24.0, 24.0 * 365 * 8, size=p2m.sum())
    merchant_txn_count = np.zeros(rows, dtype="int16")
    merchant_txn_count[p2m] = rng.poisson(3.0, size=p2m.sum()) + 1
    sender_average = rng.lognormal(mean=4.0, sigma=0.75, size=sender_count)

    frame = pd.DataFrame({
        "timestamp": timestamps,
        "amount": amounts.round(2),
        "sender_vpa": sender_vpas[sender_ids],
        "receiver_vpa": receiver_vpa,
        "transaction_type": transaction_type,
        "initiation_mode": initiation_mode,
        "merchant_mcc": np.where(
            p2m, rng.choice(MERCHANT_MCCS, size=rows), ""
        ),
        "failed_pin_attempts": np.minimum(rng.poisson(0.12, size=rows), 3).astype("int8"),
        "transaction_status": np.full(rows, "SUCCESS", dtype=object),
        "receiver_is_new": receiver_is_new,
        "receiver_age_hours": receiver_age_hours.round(2),
        "merchant_vpa_age_hours": merchant_age.round(2),
        "merchant_transactions_1h": merchant_txn_count,
        "sender_transactions_10m": (rng.poisson(0.45, size=rows) + 1).astype("int16"),
        "small_failed_transactions_1h": rng.binomial(2, 0.025, size=rows).astype("int8"),
        "amount_to_sender_average": (amounts / sender_average[sender_ids]).round(3),
        "fraud_typology": np.full(rows, "none", dtype=object),
        "is_fraud": np.zeros(rows, dtype="int8"),
    })

    scenario_size = max(2, int(round(rows * 0.004)))
    velocity_episodes = max(1, int(round(rows * 0.002)))
    minimum_velocity_rows = velocity_episodes * 4
    required = scenario_size * 2 + minimum_velocity_rows
    if required >= rows:
        raise ValueError("rows is too small to allocate scenario rows safely")

    available = np.arange(rows)
    rng.shuffle(available)
    cursor = 0

    # Collect-request scams: a high amount is requested from a newly seen VPA.
    collect_indices = available[cursor:cursor + scenario_size]
    cursor += scenario_size
    for index in collect_indices:
        receiver_id = sender_count + int(rng.integers(0, sender_count))
        handle = str(rng.choice(PSP_HANDLES))
        _assign(
            frame, int(index), timestamp=start + pd.Timedelta(seconds=int(rng.integers(0, 90 * 86400))),
            amount=round(float(rng.uniform(5_000, 35_000)), 2),
            receiver_vpa=_vpa("newuser", receiver_id, handle),
            transaction_type="P2P", initiation_mode="Collect Request", merchant_mcc="",
            receiver_is_new=1, receiver_age_hours=round(float(rng.uniform(0.1, 24)), 2),
            merchant_vpa_age_hours=np.nan, merchant_transactions_1h=0,
            failed_pin_attempts=int(rng.integers(0, 3)),
            amount_to_sender_average=round(float(rng.uniform(4, 16)), 3),
            fraud_typology="collect_request_scam", is_fraud=1,
        )

    # QR spoofing: bursts target a newly created merchant VPA.
    qr_indices = available[cursor:cursor + scenario_size]
    cursor += scenario_size
    qr_offset = 0
    while qr_offset < len(qr_indices):
        remaining = len(qr_indices) - qr_offset
        group_size = min(int(rng.integers(3, 9)), remaining)
        if remaining - group_size == 1:
            group_size += 1
        group_end = qr_offset + group_size
        group = qr_indices[qr_offset:group_end]
        merchant_id = merchant_count + qr_offset
        vpa = _vpa("newshop", merchant_id, str(rng.choice(PSP_HANDLES)))
        anchor = start + pd.Timedelta(seconds=int(rng.integers(0, 90 * 86400)))
        burst_size = len(group)
        for position, index in enumerate(group):
            _assign(
                frame, int(index),
                timestamp=anchor + pd.Timedelta(seconds=position * int(rng.integers(10, 90))),
                amount=round(float(rng.uniform(100, 8_000)), 2),
                receiver_vpa=vpa, transaction_type="P2M", initiation_mode="Scan QR",
                merchant_mcc=str(rng.choice(MERCHANT_MCCS)), receiver_is_new=1,
                receiver_age_hours=round(float(rng.uniform(0.1, 12)), 2),
                merchant_vpa_age_hours=round(float(rng.uniform(0.1, 24)), 2),
                merchant_transactions_1h=burst_size,
                transaction_status="SUCCESS", fraud_typology="qr_code_spoofing", is_fraud=1,
            )
        qr_offset = group_end

    # Velocity attacks have explicit failed micro-transactions before a large success.
    for episode in range(velocity_episodes):
        failed_count = int(rng.integers(2, 5))
        slots = available[cursor:cursor + failed_count + 1]
        cursor += failed_count + 1
        sender_id = int(rng.integers(0, sender_count))
        receiver_id = int(rng.integers(0, sender_count))
        while receiver_id == sender_id:
            receiver_id = int(rng.integers(0, sender_count))
        anchor = start + pd.Timedelta(seconds=int(rng.integers(0, 90 * 86400)))
        for attempt, index in enumerate(slots[:-1]):
            _assign(
                frame, int(index), timestamp=anchor + pd.Timedelta(minutes=attempt - failed_count),
                amount=round(float(rng.uniform(5, 90)), 2),
                sender_vpa=sender_vpas[sender_id], receiver_vpa=sender_vpas[receiver_id],
                transaction_type="P2P", initiation_mode="Contact/Phone Number",
                merchant_mcc="", failed_pin_attempts=int(rng.integers(1, 4)),
                transaction_status="FAILED", receiver_is_new=0,
                receiver_age_hours=round(float(rng.uniform(100, 24_000)), 2),
                merchant_vpa_age_hours=np.nan, merchant_transactions_1h=0,
                sender_transactions_10m=attempt + 1,
                small_failed_transactions_1h=attempt + 1,
                amount_to_sender_average=round(float(rng.uniform(0.05, 0.8)), 3),
                fraud_typology="velocity_attack_probe", is_fraud=0,
            )
        _assign(
            frame, int(slots[-1]), timestamp=anchor,
            amount=round(float(rng.uniform(10_000, 60_000)), 2),
            sender_vpa=sender_vpas[sender_id], receiver_vpa=sender_vpas[receiver_id],
            transaction_type="P2P", initiation_mode="Contact/Phone Number",
            merchant_mcc="", failed_pin_attempts=failed_count,
            transaction_status="SUCCESS", receiver_is_new=0,
            receiver_age_hours=round(float(rng.uniform(100, 24_000)), 2),
            merchant_vpa_age_hours=np.nan, merchant_transactions_1h=0,
            sender_transactions_10m=failed_count + 1,
            small_failed_transactions_1h=failed_count,
            amount_to_sender_average=round(float(rng.uniform(10, 40)), 3),
            fraud_typology="velocity_attack", is_fraud=1,
        )

    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame.insert(0, "transaction_id", [f"UPI-SYN-{i:012d}" for i in range(1, rows + 1)])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = build_dataset(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} synthetic UPI-like rows to {args.output}")
    print("This dataset is rule-generated demonstration data, not observed UPI activity.")
    print("Label counts:", frame["is_fraud"].value_counts().sort_index().to_dict())
    print("Fraud typologies:", frame.loc[frame["is_fraud"] == 1, "fraud_typology"].value_counts().to_dict())


if __name__ == "__main__":
    main()
