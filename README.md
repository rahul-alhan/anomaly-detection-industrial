# Industrial Anomaly Detection — Hybrid (Isolation Forest + Rules)

Hybrid anomaly detection on industrial weld-sensor time-series. Combines a **statistical model (Isolation Forest)** with a **deterministic rule engine** to surface anomalies that pure threshold rules miss — and to provide explainable reasons engineers will actually trust.

> Mirrors a 5-week PoC at Algoworks (client: CRC Evans) that surfaced ~6% failure rate not captured by existing threshold rules.

---

## Why Hybrid?

| Approach | Strength | Weakness |
|---|---|---|
| Threshold rules only | Explainable, easy to tune | Misses subtle multi-variate anomalies |
| Isolation Forest only | Catches multivariate patterns | Black-box; engineers don't trust it alone |
| **Hybrid (this repo)** | Catches both, scored together, with reasons | More config to maintain — worth it |

The hybrid detector emits an **anomaly score** *and* a list of **triggered reasons** (e.g. `"voltage > 32V (rule)"`, `"if_score=0.78 (model)"`) so a human can act on it.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic weld sensor data with injected anomalies
python -m src.generate_signals --rows 5000 --out data/signals.parquet

# 2. Fit the hybrid detector
python -m src.hybrid_detector fit \
  --in data/signals.parquet \
  --out artifacts/detector.pkl

# 3. Score new data
python -m src.hybrid_detector score \
  --model artifacts/detector.pkl \
  --in data/signals.parquet \
  --out reports/scored.parquet

# 4. Launch the dashboard
streamlit run dashboard/streamlit_dashboard.py
```

---

## Detector Architecture

```
sensor stream ──┬──▶  rule_engine.py     ──┐
                │                          │
                └──▶  isolation_forest.py ─┤
                                           ▼
                                  hybrid_detector.py
                                  (combine + explain)
                                           │
                                           ▼
                                  {score, is_anomaly, reasons}
```

### Rule Engine (`rule_engine.py`)
Hardcoded engineering thresholds — voltage, current, temperature, gas-flow ranges.
Each rule has an `id`, `description`, `predicate`, and `severity`.

### Isolation Forest (`isolation_forest.py`)
Standard scikit-learn `IsolationForest` with contamination=`auto`, fit on a "known good" reference window.

### Hybrid Combiner (`hybrid_detector.py`)
- `is_anomaly = rule_triggered OR model_anomaly`
- `score = max(normalized_rule_severity, normalized_model_score)`
- `reasons = [...rule_reasons, ...model_reason]`

---

## Repository Layout

```
anomaly-detection-industrial/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── generate_signals.py
│   ├── isolation_forest.py
│   ├── rule_engine.py
│   └── hybrid_detector.py
├── notebooks/
│   └── sensor_analysis.ipynb
├── dashboard/
│   └── streamlit_dashboard.py
└── tests/
    └── test_detector.py
```

---

## What the PoC Showed

- **6%** of welds flagged by the hybrid detector were missed by the existing rule-only system
- Stakeholder dashboard delivered in **Week 1** of a 5-week engagement → bought trust early
- Reasons displayed alongside scores → engineers actually adopted the dashboard, not bypassed it

---

## Notes

- Synthetic data here is generated to mimic the real signal characteristics — code is portable to actual sensor streams (just swap the loader).
- The dashboard is intentionally simple — the goal is to put anomalies in front of an engineer in under 10 seconds, not to win a design award.

---

## License

MIT
