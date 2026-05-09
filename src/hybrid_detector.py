"""Hybrid (rules + isolation forest) anomaly detector."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .isolation_forest import IFAnomalyModel
from .rule_engine import apply_rules

FEATURES = ["voltage", "current", "temperature", "gas_flow", "wire_speed"]


class HybridDetector:
    def __init__(self):
        self.if_model = IFAnomalyModel()

    def fit(self, df: pd.DataFrame) -> "HybridDetector":
        # Fit IF only on rows that pass rules — closer to "known good"
        rules_out = apply_rules(df)
        clean = df[~rules_out["rule_triggered"]]
        self.if_model.fit(clean, FEATURES)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        rules_out = apply_rules(df)
        if_out = self.if_model.score(df)

        combined = pd.concat([df.reset_index(drop=True), rules_out.reset_index(drop=True), if_out.reset_index(drop=True)], axis=1)
        combined["score"] = np.maximum(combined["rule_severity"], combined["if_score"])
        combined["is_anomaly"] = combined["rule_triggered"] | combined["if_is_anomaly"]
        combined["reasons"] = combined.apply(
            lambda r: list(r["rule_reasons"]) + ([f"if_score={r['if_score']:.2f}"] if r["if_is_anomaly"] else []),
            axis=1,
        )
        return combined


def _cmd_fit(args):
    df = pd.read_parquet(args.inp)
    det = HybridDetector().fit(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(det, args.out)
    print(f"Fit on {len(df):,} rows → {args.out}")


def _cmd_score(args):
    det: HybridDetector = joblib.load(args.model)
    df = pd.read_parquet(args.inp)
    scored = det.score(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.out, index=False)
    n_anom = int(scored["is_anomaly"].sum())
    print(f"Scored {len(scored):,} rows; flagged {n_anom:,} anomalies "
          f"({n_anom/len(scored):.1%}) → {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--in", dest="inp", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(func=_cmd_fit)

    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_score)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
