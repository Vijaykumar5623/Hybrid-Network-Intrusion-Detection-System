"""Inference module for Hybrid IDS.

Runs the fusion engine on a single flow or batch of flows.
Supports inputs:
- .npy file containing a feature matrix
- .csv file with numeric feature columns
- manual comma-separated feature list
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fusion import fuse_scores, get_paths, load_artifacts


def load_input(path: Path) -> np.ndarray:
	if path.suffix.lower() == ".npy":
		return np.load(path)
	if path.suffix.lower() == ".csv":
		df = pd.read_csv(path)
		return df.select_dtypes(include=["number"]).to_numpy(dtype=np.float32, copy=False)
	raise ValueError("Unsupported input file type. Use .npy or .csv")


def parse_features(raw: str) -> np.ndarray:
	parts = [p.strip() for p in raw.split(",") if p.strip()]
	if not parts:
		raise ValueError("No features provided.")
	values = np.array([float(p) for p in parts], dtype=np.float32)
	return values.reshape(1, -1)


def build_fusion_artifacts(project_root: Optional[Path] = None):
	paths = get_paths(project_root)
	return load_artifacts(paths.models_dir)


def infer_scores(
	X: np.ndarray,
	*,
	project_root: Optional[Path] = None,
	artifacts=None,
	w1: float = 0.7,
	w2: float = 0.3,
	threshold_high: Optional[float] = None,
	batch_size: int = 256,
	input_scaled: bool = False,
) -> dict:
	artifacts = artifacts or build_fusion_artifacts(project_root)
	return fuse_scores(
		X,
		artifacts=artifacts,
		w1=w1,
		w2=w2,
		threshold_high=threshold_high,
		batch_size=batch_size,
		input_scaled=input_scaled,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Hybrid-IDS inference CLI")
	parser.add_argument(
		"--project-root",
		type=str,
		default=None,
		help="Path to Hybrid-IDS folder (defaults to auto-detect).",
	)
	parser.add_argument("--input", type=str, default=None, help="Path to .npy or .csv feature file")
	parser.add_argument(
		"--features",
		type=str,
		default=None,
		help="Comma-separated feature values for a single flow",
	)
	parser.add_argument("--w1", type=float, default=0.7)
	parser.add_argument("--w2", type=float, default=0.3)
	parser.add_argument("--threshold-high", type=float, default=None)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument(
		"--input-scaled",
		action="store_true",
		help="Set when --input is already scaled with the saved scaler.",
	)
	parser.add_argument("--top-k", type=int, default=3, help="Show top-k class probabilities")
	args = parser.parse_args()

	project_root = Path(args.project_root).resolve() if args.project_root else None
	artifacts = build_fusion_artifacts(project_root)

	X: Optional[np.ndarray] = None
	if args.input:
		X = load_input(Path(args.input))
	elif args.features:
		X = parse_features(args.features)
	else:
		raise SystemExit("Provide --input or --features")

	result = infer_scores(
		X,
		artifacts=artifacts,
		w1=args.w1,
		w2=args.w2,
		threshold_high=args.threshold_high,
		batch_size=args.batch_size,
		input_scaled=args.input_scaled,
	)

	labels = result["labels"]
	risk = result["risk"]
	anomaly = result["anomaly_score"]
	proba = result["probabilities"]

	for i in range(min(len(labels), 10)):
		top_idx = np.argsort(proba[i])[::-1][: args.top_k]
		top_probs = [(int(idx), float(proba[i][idx])) for idx in top_idx]
		print(
			f"[{i}] label={labels[i]} risk={risk[i]:.4f} anomaly={anomaly[i]:.4f} top_probs={top_probs}"
		)


if __name__ == "__main__":
	main()
