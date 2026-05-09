"""Streamlit dashboard for browsing scored anomalies."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")
st.title("Weld Sensor Anomaly Detection")

scored_path = Path("reports/scored.parquet")
if not scored_path.exists():
    st.error("Run the scoring step first: see README quickstart.")
    st.stop()

df = pd.read_parquet(scored_path)
st.sidebar.metric("Rows", f"{len(df):,}")
st.sidebar.metric("Anomalies", f"{int(df['is_anomaly'].sum()):,}")
st.sidebar.metric("Anomaly rate", f"{df['is_anomaly'].mean():.1%}")

feature = st.sidebar.selectbox(
    "Feature to plot",
    ["voltage", "current", "temperature", "gas_flow", "wire_speed"],
)

st.subheader(f"{feature} over time")
fig = px.scatter(
    df, x="ts", y=feature, color="is_anomaly",
    color_discrete_map={True: "#e74c3c", False: "#95a5a6"},
    opacity=0.6, height=420,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top 20 anomalies (by score)")
top = df[df["is_anomaly"]].nlargest(20, "score")[
    ["ts", "voltage", "current", "temperature", "gas_flow", "wire_speed", "score", "reasons"]
]
top["reasons"] = top["reasons"].apply(lambda r: "; ".join(r))
st.dataframe(top, use_container_width=True)

if "is_truth_anomaly" in df:
    st.subheader("Confusion vs injected truth")
    cross = pd.crosstab(df["is_truth_anomaly"], df["is_anomaly"])
    st.dataframe(cross)
