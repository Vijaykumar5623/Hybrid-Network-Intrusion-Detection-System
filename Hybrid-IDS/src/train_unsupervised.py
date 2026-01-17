"""Train the unsupervised autoencoder for the Hybrid IDS project.

This script:
- Loads benign-only arrays from data/processed/ (X_train_ae.npy, X_test_ae.npy)
- Builds a Dense autoencoder: input -> 64 -> 32 -> 16 -> 32 -> 64 -> input
- Trains with MSE loss and Adam optimizer
- Computes reconstruction error statistics on train/test
- Saves model to models/autoencoder.h5
- Saves reconstruction stats to models/ae_stats.pkl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import dump


@dataclass(frozen=True)
class Paths:
	project_root: Path
	processed_dir: Path
	models_dir: Path


def _find_project_root(start: Optional[Path] = None) -> Path:
	here = (start or Path(__file__)).resolve()
	for parent in [here.parent, *here.parents]:
		if (parent / "data").exists() and (parent / "src").exists():
			return parent
	return here.parent


def get_paths(project_root: Optional[Path] = None) -> Paths:
	root = (project_root or _find_project_root()).resolve()
	return Paths(
		project_root=root,
		processed_dir=root / "data" / "processed",
		models_dir=root / "models",
	)


def load_benign_artifacts(processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
	X_train = np.load(processed_dir / "X_train_ae.npy")
	X_test = np.load(processed_dir / "X_test_ae.npy")

	if X_train.ndim != 2 or X_test.ndim != 2:
		raise ValueError("Expected X_train_ae/X_test_ae to be 2D arrays.")
	if X_train.shape[1] != X_test.shape[1]:
		raise ValueError("Train/test feature dimension mismatch.")
	if X_train.shape[0] < 2 or X_test.shape[0] < 1:
		raise ValueError("Not enough samples to train/evaluate autoencoder.")

	# Ensure float32 for TensorFlow efficiency
	X_train = X_train.astype(np.float32, copy=False)
	X_test = X_test.astype(np.float32, copy=False)
	return X_train, X_test


def maybe_subsample(
	X: np.ndarray,
	*,
	max_samples: Optional[int],
	random_state: int,
) -> np.ndarray:
	if not max_samples:
		return X
	if max_samples <= 0:
		raise ValueError("max_samples must be a positive integer")
	if X.shape[0] <= max_samples:
		return X

	rng = np.random.default_rng(random_state)
	idx = rng.choice(X.shape[0], size=max_samples, replace=False)
	return X[idx]


def build_autoencoder(input_dim: int):
	try:
		from tensorflow import keras
	except Exception as e:
		raise ImportError(
			"TensorFlow is not installed in this environment. Install tensorflow-directml (Windows) or tensorflow."
		) from e

	inputs = keras.Input(shape=(input_dim,), name="features")
	x = keras.layers.Dense(64, activation="relu")(inputs)
	x = keras.layers.Dense(32, activation="relu")(x)
	encoded = keras.layers.Dense(16, activation="relu")(x)
	x = keras.layers.Dense(32, activation="relu")(encoded)
	x = keras.layers.Dense(64, activation="relu")(x)
	outputs = keras.layers.Dense(input_dim, activation=None, name="reconstruction")(x)

	model = keras.Model(inputs=inputs, outputs=outputs, name="dense_autoencoder")
	model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
	return model


def reconstruction_errors(model, X: np.ndarray, batch_size: int) -> np.ndarray:
	X_hat = model.predict(X, batch_size=batch_size, verbose=0)
	# Per-sample MSE
	err = np.mean(np.square(X - X_hat), axis=1)
	return err.astype(np.float64, copy=False)


def summarize_errors(err: np.ndarray) -> dict:
	return {
		"mean": float(np.mean(err)),
		"std": float(np.std(err)),
		"p95": float(np.percentile(err, 95)),
		"p99": float(np.percentile(err, 99)),
	}


def save_artifacts(models_dir: Path, model, stats: dict) -> tuple[Path, Path]:
	models_dir.mkdir(parents=True, exist_ok=True)
	model_path = models_dir / "autoencoder.h5"
	stats_path = models_dir / "ae_stats.pkl"

	model.save(model_path)
	dump(stats, stats_path)
	return model_path, stats_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Train dense autoencoder for Hybrid-IDS")
	parser.add_argument(
		"--project-root",
		type=str,
		default=None,
		help="Path to Hybrid-IDS folder (defaults to auto-detect).",
	)
	parser.add_argument("--epochs", type=int, default=40)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument(
		"--max-train-samples",
		type=int,
		default=None,
		help="Optional cap on benign train rows (useful for quick tests).",
	)
	parser.add_argument(
		"--max-test-samples",
		type=int,
		default=None,
		help="Optional cap on benign test rows (useful for quick tests).",
	)
	args = parser.parse_args()

	project_root = Path(args.project_root).resolve() if args.project_root else None
	paths = get_paths(project_root)

	X_train, X_test = load_benign_artifacts(paths.processed_dir)
	X_train = maybe_subsample(X_train, max_samples=args.max_train_samples, random_state=args.random_state)
	X_test = maybe_subsample(X_test, max_samples=args.max_test_samples, random_state=args.random_state)

	print(f"Loaded X_train_ae={X_train.shape}, X_test_ae={X_test.shape}")

	model = build_autoencoder(input_dim=int(X_train.shape[1]))
	model.summary()

	model.fit(
		X_train,
		X_train,
		epochs=int(args.epochs),
		batch_size=int(args.batch_size),
		shuffle=True,
		verbose=1,
	)

	err_train = reconstruction_errors(model, X_train, batch_size=int(args.batch_size))
	err_test = reconstruction_errors(model, X_test, batch_size=int(args.batch_size))

	train_stats = summarize_errors(err_train)
	test_stats = summarize_errors(err_test)

	print("Reconstruction error stats (train):")
	print(
		f"  mean={train_stats['mean']:.6f}  std={train_stats['std']:.6f}  p95={train_stats['p95']:.6f}  p99={train_stats['p99']:.6f}"
	)
	print("Reconstruction error stats (test):")
	print(
		f"  mean={test_stats['mean']:.6f}  std={test_stats['std']:.6f}  p95={test_stats['p95']:.6f}  p99={test_stats['p99']:.6f}"
	)

	stats = {
		"input_dim": int(X_train.shape[1]),
		"epochs": int(args.epochs),
		"batch_size": int(args.batch_size),
		"random_state": int(args.random_state),
		"train": train_stats,
		"test": test_stats,
		# Commonly-used thresholds (pick one in fusion phase)
		"threshold_p95": float(train_stats["p95"]),
		"threshold_p99": float(train_stats["p99"]),
	}

	model_path, stats_path = save_artifacts(paths.models_dir, model, stats)
	print(f"Saved autoencoder to: {model_path}")
	print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
	main()
