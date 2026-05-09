"""Generate synthetic weld-sensor time-series with injected anomalies."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["voltage", "current", "temperature", "gas_flow", "wire_speed"]


def generate(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=n, freq="1s"),
            "voltage": rng.normal(28.0, 1.2, n),
            "current": rng.normal(150.0, 8.0, n),
            "temperature": rng.normal(420.0, 25.0, n),
            "gas_flow": rng.normal(20.0, 1.5, n),
            "wire_speed": rng.normal(8.0, 0.4, n),
        }
    )

    n_anom = int(0.06 * n)
    anom_idx = rng.choice(n, n_anom, replace=False)
    for i in anom_idx:
        kind = rng.integers(0, 4)
        if kind == 0:                                  # rule violation: voltage spike
            df.loc[i, "voltage"] = rng.uniform(33, 38)
        elif kind == 1:                                # rule violation: temperature drop
            df.loc[i, "temperature"] = rng.uniform(280, 320)
        elif kind == 2:                                # subtle multivariate
            df.loc[i, "voltage"] = 30.5
            df.loc[i, "current"] = 175.0
            df.loc[i, "gas_flow"] = 23.0
        else:                                          # gas-flow drop
            df.loc[i, "gas_flow"] = rng.uniform(8, 13)

    df["is_truth_anomaly"] = 0
    df.loc[anom_idx, "is_truth_anomaly"] = 1
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--out", default="data/signals.parquet")
    args = p.parse_args()

    df = generate(args.rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    truth_rate = df["is_truth_anomaly"].mean()
    print(f"Wrote {len(df):,} rows ({truth_rate:.1%} injected anomalies) → {args.out}")


if __name__ == "__main__":
    main()
