"""Streaming replay for Hybrid IDS.

Replays feature batches from a file and emits structured event lines.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from config import enrich_event
from infer import infer_scores, load_input


def iter_batches(X: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
	for start in range(0, X.shape[0], batch_size):
		yield X[start : start + batch_size]


def format_event(
	*,
	label: str,
	risk: float,
	anomaly: float,
	reconstruction_error: float,
	threshold_p95: float,
	top_probs: list[tuple[int, float]],
) -> str:
	event = {
		"timestamp": time.time(),
		"label": label,
		"risk": float(risk),
		"anomaly": float(anomaly),
		"reconstruction_error": float(reconstruction_error),
		"anomaly_threshold_p95": float(threshold_p95),
		"top_probs": [(int(c), float(p)) for c, p in top_probs],
	}
	enrich_event(event, anomaly_threshold_p95=threshold_p95)
	return json.dumps(event)


def main() -> None:
	parser = argparse.ArgumentParser(description="Hybrid-IDS streaming replay")
	parser.add_argument(
		"--project-root",
		type=str,
		default=None,
		help="Path to Hybrid-IDS folder (defaults to auto-detect).",
	)
	parser.add_argument(
		"--input",
		type=str,
		required=True,
		help="Path to .npy or .csv feature file (scaled or raw features).",
	)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--interval", type=float, default=0.5)
	parser.add_argument("--max-batches", type=int, default=None)
	parser.add_argument("--top-k", type=int, default=3)
	parser.add_argument("--w1", type=float, default=0.7)
	parser.add_argument("--w2", type=float, default=0.3)
	parser.add_argument("--threshold-high", type=float, default=None)
	parser.add_argument(
		"--input-scaled",
		action="store_true",
		help="Set when --input is already scaled with the saved scaler.",
	)
	parser.add_argument(
		"--log",
		type=str,
		default=None,
		help="Optional path to JSONL log file for events.",
	)
	args = parser.parse_args()

	project_root = Path(args.project_root).resolve() if args.project_root else None
	X = load_input(Path(args.input))

	batch_count = 0
	log_handle = None
	try:
		if args.log:
			log_path = Path(args.log)
			log_path.parent.mkdir(parents=True, exist_ok=True)
			log_handle = log_path.open("a", encoding="utf-8", buffering=1)

		for batch in iter_batches(X, args.batch_size):
			result = infer_scores(
				batch,
				project_root=project_root,
				w1=args.w1,
				w2=args.w2,
				threshold_high=args.threshold_high,
				batch_size=args.batch_size,
				input_scaled=args.input_scaled,
			)

			labels = result["labels"]
			risk = result["risk"]
			anomaly = result["anomaly_score"]
			recon = result["reconstruction_error"]
			threshold_p95 = float(result.get("threshold_p95", 0.0))
			proba = result["probabilities"]

			for i in range(len(labels)):
				top_idx = np.argsort(proba[i])[::-1][: args.top_k]
				top_probs = [(int(idx), float(proba[i][idx])) for idx in top_idx]
				line = format_event(
					label=str(labels[i]),
					risk=float(risk[i]),
					anomaly=float(anomaly[i]),
					reconstruction_error=float(recon[i]),
					threshold_p95=threshold_p95,
					top_probs=top_probs,
				)
				print(line)
				if log_handle is not None:
					log_handle.write(line + "\n")
					log_handle.flush()

			batch_count += 1
			if args.max_batches is not None and batch_count >= args.max_batches:
				break
			if args.interval > 0:
				time.sleep(args.interval)
	except KeyboardInterrupt:
		print("Interrupted by user.")
	finally:
		if log_handle is not None:
			log_handle.close()


if __name__ == "__main__":
	main()
