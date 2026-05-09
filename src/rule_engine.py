"""Deterministic engineering rules — explainable first line of defense."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class Rule:
    rule_id: str
    description: str
    predicate: Callable[[pd.DataFrame], pd.Series]
    severity: float  # 0.0 - 1.0


RULES: list[Rule] = [
    Rule("R001", "voltage out of safe band [25, 32] V",
         lambda df: (df["voltage"] < 25) | (df["voltage"] > 32),
         severity=0.9),
    Rule("R002", "current out of safe band [120, 180] A",
         lambda df: (df["current"] < 120) | (df["current"] > 180),
         severity=0.9),
    Rule("R003", "temperature below 360 C",
         lambda df: df["temperature"] < 360,
         severity=0.85),
    Rule("R004", "gas flow below 16 L/min",
         lambda df: df["gas_flow"] < 16,
         severity=0.95),
    Rule("R005", "wire speed out of band [6.5, 9.5]",
         lambda df: (df["wire_speed"] < 6.5) | (df["wire_speed"] > 9.5),
         severity=0.7),
]


def apply_rules(df: pd.DataFrame, rules: list[Rule] | None = None) -> pd.DataFrame:
    rules = rules or RULES
    out = pd.DataFrame(index=df.index)
    out["rule_triggered"] = False
    out["rule_severity"] = 0.0
    out["rule_reasons"] = [[] for _ in range(len(df))]

    for rule in rules:
        mask = rule.predicate(df).fillna(False)
        out.loc[mask, "rule_triggered"] = True
        out.loc[mask, "rule_severity"] = out.loc[mask, "rule_severity"].clip(lower=rule.severity)
        for i in df.index[mask]:
            out.at[i, "rule_reasons"].append(f"{rule.rule_id}: {rule.description}")
    return out
