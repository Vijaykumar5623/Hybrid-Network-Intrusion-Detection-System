"""Fusion engine for Hybrid IDS.

Combines classifier probabilities and autoencoder reconstruction error to produce
risk scores and final labels (including ZERO_DAY for emerging anomalies).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import load


LABEL_TO_ID = {
	"BENIGN": 0,
	"BRUTE_FORCE": 1,
	"DOS": 2,
	"DDOS": 3,
	"PORTSCAN": 4,
	"WEB_ATTACK": 5,
	"BOTNET": 6,
	"INFILTRATION": 7,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
BENIGN_CLASS = LABEL_TO_ID["BENIGN"]


@dataclass(frozen=True)
class Paths:
	project_root: Path
	models_dir: Path


def _find_project_root(start: Optional[Path] = None) -> Path:
	here = (start or Path(__file__)).resolve()
	for parent in [here.parent, *here.parents]:
		if (parent / "data").exists() and (parent / "src").exists():
			return parent
	return here.parent


def get_paths(project_root: Optional[Path] = None) -> Paths:
	root = (project_root or _find_project_root()).resolve()
	return Paths(project_root=root, models_dir=root / "models")


@dataclass
class FusionArtifacts:
	classifier: object
	autoencoder: object
	scaler: object
	ae_stats: dict


def load_artifacts(models_dir: Path) -> FusionArtifacts:
	try:
		from tensorflow import keras
	except Exception as e:
		raise ImportError(
			"TensorFlow is not installed. Install tensorflow (or tensorflow-directml on Windows)."
		) from e

	classifier = load(models_dir / "classifier.pkl")
	autoencoder = keras.models.load_model(models_dir / "autoencoder.h5")
	scaler = load(models_dir / "scaler.pkl")
	ae_stats = load(models_dir / "ae_stats.pkl")

	return FusionArtifacts(
		classifier=classifier,
		autoencoder=autoencoder,
		scaler=scaler,
		ae_stats=ae_stats,
	)


def _ensure_2d(X: np.ndarray) -> np.ndarray:
	if X.ndim == 1:
		return X.reshape(1, -1)
	if X.ndim != 2:
		raise ValueError("X must be a 1D or 2D array of features.")
	return X


def _predict_proba(classifier: object, X: np.ndarray) -> np.ndarray:
	if hasattr(classifier, "predict_proba"):
		return classifier.predict_proba(X)
	# xgboost.Booster fallback
	try:
		import xgboost as xgb
		return classifier.predict(xgb.DMatrix(X))
	except Exception as e:
		raise TypeError("Classifier does not support predict_proba or Booster predict.") from e


def reconstruction_error(autoencoder: object, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
	X_hat = autoencoder.predict(X, batch_size=batch_size, verbose=0)
	return np.mean(np.square(X - X_hat), axis=1)


def fuse_scores(
	X: np.ndarray,
	*,
	artifacts: FusionArtifacts,
	w1: float = 0.7,
	w2: float = 0.3,
	threshold_high: Optional[float] = None,
	batch_size: int = 256,
	input_scaled: bool = False,
) -> dict:
	X = _ensure_2d(X)

	# Scale if raw features are provided
	X_scaled = X if input_scaled else artifacts.scaler.transform(X)

	proba = _predict_proba(artifacts.classifier, X_scaled)
	attack_prob = 1.0 - proba[:, BENIGN_CLASS]

	recon = reconstruction_error(artifacts.autoencoder, X_scaled, batch_size=batch_size)
	mean = float(artifacts.ae_stats.get("train", {}).get("mean", 0.0))
	std = float(artifacts.ae_stats.get("train", {}).get("std", 1.0))
	if std <= 0:
		std = 1.0

	anomaly_norm = (recon - mean) / std
	anomaly_norm = np.maximum(anomaly_norm, 0.0)

	risk = (w1 * attack_prob) + (w2 * anomaly_norm)

	if threshold_high is None:
		threshold_high = float(artifacts.ae_stats.get("threshold_p99", 0.0))
	threshold_p95 = float(artifacts.ae_stats.get("threshold_p95", 0.0))
	zero_day_threshold = mean + (2.0 * std)

	pred_class = np.argmax(proba, axis=1)
	top1_prob = np.max(proba, axis=1)
	labels: list[str] = []
	for i, cls in enumerate(pred_class):
		if top1_prob[i] < 0.6 and recon[i] > zero_day_threshold:
			labels.append("ZERO_DAY")
		elif cls != BENIGN_CLASS:
			labels.append(ID_TO_LABEL.get(int(cls), str(int(cls))))
		else:
			labels.append("BENIGN")

	return {
		"labels": np.array(labels, dtype=object),
		"risk": risk,
		"anomaly_score": anomaly_norm,
		"reconstruction_error": recon,
		"probabilities": proba,
		"attack_prob": attack_prob,
		"top1_prob": top1_prob,
		"threshold_p95": threshold_p95,
		"threshold_p99": float(threshold_high),
		"zero_day_threshold": zero_day_threshold,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Hybrid-IDS fusion engine")
	parser.add_argument(
		"--project-root",
		type=str,
		default=None,
		help="Path to Hybrid-IDS folder (defaults to auto-detect).",
	)
	parser.add_argument("--w1", type=float, default=0.7)
	parser.add_argument("--w2", type=float, default=0.3)
	parser.add_argument(
		"--threshold-high",
		type=float,
		default=None,
		help="Override anomaly threshold (defaults to ae_stats p99).",
	)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument(
		"--input",
		type=str,
		default=None,
		help="Optional path to .npy with feature matrix (raw, unscaled).",
	)
	args = parser.parse_args()

	project_root = Path(args.project_root).resolve() if args.project_root else None
	paths = get_paths(project_root)
	artifacts = load_artifacts(paths.models_dir)

	if args.input:
		X = np.load(args.input)
		result = fuse_scores(
			X,
			artifacts=artifacts,
			w1=args.w1,
			w2=args.w2,
			threshold_high=args.threshold_high,
			batch_size=args.batch_size,
		)
		print("Labels:", result["labels"][:10])
		print("Risk:", result["risk"][:10])
		print("Anomaly score:", result["anomaly_score"][:10])
	else:
		print("Fusion engine ready. Provide --input <features.npy> to score.")


if __name__ == "__main__":
	main()
