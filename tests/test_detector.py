"""Smoke tests for rule engine + hybrid detector."""
from __future__ import annotations

import pandas as pd

from src.generate_signals import generate
from src.hybrid_detector import HybridDetector
from src.rule_engine import apply_rules


def test_rule_engine_flags_voltage_spike():
    df = pd.DataFrame({
        "voltage": [28, 35, 27],
        "current": [150, 150, 150],
        "temperature": [420, 420, 420],
        "gas_flow": [20, 20, 20],
        "wire_speed": [8, 8, 8],
    })
    out = apply_rules(df)
    assert out.loc[1, "rule_triggered"] is True or out.loc[1, "rule_triggered"]
    assert out.loc[0, "rule_triggered"] in (False, 0)


def test_hybrid_detector_runs():
    df = generate(500)
    det = HybridDetector().fit(df)
    scored = det.score(df)
    assert "score" in scored
    assert "is_anomaly" in scored
    # we injected 6%, so reasonable bound
    rate = scored["is_anomaly"].mean()
    assert 0.02 <= rate <= 0.25
