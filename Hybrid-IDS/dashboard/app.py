"""Streamlit SOC-style dashboard for Hybrid IDS events."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_LOG = Path("data") / "processed" / "events.jsonl"


LABEL_COLORS = {
	"BENIGN": "#1f7a1f",
	"BRUTE_FORCE": "#ff9f1c",
	"DOS": "#ff6b6b",
	"DDOS": "#ff006e",
	"PORTSCAN": "#ffb703",
	"WEB_ATTACK": "#ff7a00",
	"BOTNET": "#ff4d6d",
	"INFILTRATION": "#ff5f5f",
	"UNKNOWN_ATTACK": "#8b5cf6",
}


def load_events(log_path: Path, tail: int) -> pd.DataFrame:
	if not log_path.exists():
		return pd.DataFrame(columns=["timestamp", "label", "risk", "anomaly", "top_probs"])

	rows = []
	with log_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			try:
				rows.append(json.loads(line))
			except json.JSONDecodeError:
				continue

	if not rows:
		return pd.DataFrame(columns=["timestamp", "label", "risk", "anomaly", "top_probs"])

	if tail is not None and tail > 0:
		rows = rows[-tail:]

	df = pd.DataFrame(rows)
	df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
	return df


def label_style(label: str) -> str:
	color = LABEL_COLORS.get(label, "#9ca3af")
	return f"color: {color}; font-weight: 700;"


def main() -> None:
	st.set_page_config(page_title="Hybrid IDS SOC", layout="wide")

	st.markdown(
		"""
		<style>
		body, .stApp { background-color: #0b0f19; color: #e2e8f0; }
		.block-container { padding-top: 1.2rem; }
		.stDataFrame { background-color: #0f172a; }
		.card { background: #0f172a; border: 1px solid #1f2937; padding: 0.75rem; border-radius: 0.5rem; }
		.alert { background: #2a0b1f; border: 1px solid #7f1d1d; padding: 0.5rem; border-radius: 0.5rem; }
		</style>
		""",
		unsafe_allow_html=True,
	)

	st.title("Hybrid IDS SOC Dashboard")

	with st.sidebar:
		st.subheader("Settings")
		log_path = st.text_input("JSONL log path", value=str(DEFAULT_LOG))
		tail = st.slider("Tail events", min_value=100, max_value=5000, value=1000, step=100)
		refresh = st.slider("Refresh (sec)", min_value=1, max_value=2, value=1)
		st.caption("Auto-refresh is enabled")
		st_autorefresh = getattr(st, "autorefresh", None)
		if st_autorefresh:
			st_autorefresh(interval=refresh * 1000, key="refresh")

	log_path = Path(log_path)
	df = load_events(log_path, tail)

	if df.empty:
		st.warning("No events yet. Run stream.py with --log to generate events.")
		st.stop()

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Events", len(df))
	col2.metric("Unique Labels", df["label"].nunique())
	col3.metric("Avg Risk", f"{df['risk'].mean():.4f}")
	col4.metric("Max Risk", f"{df['risk'].max():.4f}")

	st.markdown("---")

	left, right = st.columns([2, 1])
	with left:
		st.subheader("Live Events")
		display = df.copy()
		display["time"] = df["timestamp"].dt.strftime("%H:%M:%S")
		display = display[["time", "label", "risk", "anomaly", "top_probs"]]
		styled = display.style.applymap(lambda _: label_style(_), subset=["label"])
		st.dataframe(styled, use_container_width=True, height=420)

	with right:
		st.subheader("Alerts")
		alerts = df[df["label"].isin(["UNKNOWN_ATTACK", "BRUTE_FORCE", "DOS", "DDOS", "PORTSCAN", "WEB_ATTACK", "BOTNET", "INFILTRATION"])]
		alerts = alerts.sort_values("timestamp", ascending=False).head(10)
		if alerts.empty:
			st.write("No alerts yet.")
		else:
			for _, row in alerts.iterrows():
				label = row["label"]
				time_str = row["timestamp"].strftime("%H:%M:%S") if pd.notnull(row["timestamp"]) else "-"
				st.markdown(
					f"<div class='alert'>[{time_str}] <b>{label}</b> | risk={row['risk']:.4f} | anomaly={row['anomaly']:.4f}</div>",
					unsafe_allow_html=True,
				)

	st.markdown("---")

	chart_col1, chart_col2, chart_col3 = st.columns(3)
	with chart_col1:
		st.subheader("Risk Over Time")
		chart_df = df.copy()
		chart_df = chart_df.set_index("timestamp").sort_index()
		st.line_chart(chart_df["risk"], height=250)

	with chart_col2:
		st.subheader("Anomaly Distribution")
		st.area_chart(df["anomaly"], height=250)

	with chart_col3:
		st.subheader("Class Distribution")
		counts = df["label"].value_counts().rename_axis("label").reset_index(name="count")
		st.bar_chart(counts.set_index("label"), height=250)

	st.caption(f"Last update: {datetime.utcnow().strftime('%H:%M:%S')} UTC")
	if not st_autorefresh:
		time.sleep(refresh)
		rerun = getattr(st, "rerun", None)
		if rerun:
			rerun()


if __name__ == "__main__":
	main()
