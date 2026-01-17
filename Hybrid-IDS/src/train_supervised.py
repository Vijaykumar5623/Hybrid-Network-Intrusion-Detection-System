"""Train the supervised classifier (XGBoost) for the Hybrid IDS project.

This script:
- Loads scaled arrays from data/processed/
- Trains an XGBoost multi-class classifier (8 classes)
- Handles class imbalance via per-sample class weights
- Evaluates on the held-out test set and prints metrics
- Saves the trained model to models/classifier.pkl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import dump
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight


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
NUM_CLASSES = 8


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


def load_artifacts(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	X_train = np.load(processed_dir / "X_train.npy")
	X_test = np.load(processed_dir / "X_test.npy")
	y_train = np.load(processed_dir / "y_train.npy")
	y_test = np.load(processed_dir / "y_test.npy")

	if X_train.ndim != 2 or X_test.ndim != 2:
		raise ValueError("Expected X_train/X_test to be 2D arrays.")
	if y_train.ndim != 1 or y_test.ndim != 1:
		raise ValueError("Expected y_train/y_test to be 1D arrays.")
	if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
		raise ValueError("Mismatched X/y lengths.")
	if X_train.shape[1] != X_test.shape[1]:
		raise ValueError("Train/test feature dimension mismatch.")

	for name, y in (("y_train", y_train), ("y_test", y_test)):
		min_y = int(np.min(y))
		max_y = int(np.max(y))
		if min_y < 0 or max_y >= NUM_CLASSES:
			raise ValueError(f"{name} labels out of range: min={min_y}, max={max_y}")

	return X_train, X_test, y_train.astype(np.int64, copy=False), y_test.astype(np.int64, copy=False)


def maybe_subsample(
	X: np.ndarray,
	y: np.ndarray,
	*,
	max_samples: Optional[int],
	random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
	if not max_samples:
		return X, y
	if max_samples <= 0:
		raise ValueError("max_samples must be a positive integer")
	if X.shape[0] <= max_samples:
		return X, y

	sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=random_state)
	idx, _ = next(sss.split(X, y))
	return X[idx], y[idx]


def maybe_print_roc_pr(y_true: np.ndarray, y_proba: np.ndarray) -> None:
	"""Optional ROC/PR summary per class.

	Computes one-vs-rest AUCs. If sklearn metrics aren't available or computation
	fails for a class, this will gracefully skip.
	"""
	try:
		from sklearn.metrics import average_precision_score, roc_auc_score
	except Exception:
		return

	y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
	# y_proba is shape [n_samples, num_class]
	if y_proba.ndim != 2 or y_proba.shape[1] != NUM_CLASSES:
		return

	print("Per-class ROC-AUC / PR-AUC (one-vs-rest):")
	for class_id in range(NUM_CLASSES):
		try:
			roc = roc_auc_score(y_bin[:, class_id], y_proba[:, class_id])
			pr = average_precision_score(y_bin[:, class_id], y_proba[:, class_id])
			name = ID_TO_LABEL.get(class_id, str(class_id))
			print(f"  {name:13s} ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")
		except Exception:
			continue


def train_xgboost(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	*,
	random_state: int,
	n_estimators: int,
	max_depth: int,
	learning_rate: float,
	subsample: float,
	colsample_bytree: float,
	reg_lambda: float,
	min_child_weight: float,
	verbosity: int,
) -> tuple[object, np.ndarray]:
	try:
		import xgboost as xgb
	except ImportError as e:
		raise ImportError(
			"xgboost is not installed in this environment. Install it with: pip install xgboost"
		) from e

	# Multiclass imbalance handling: use per-sample weights (balanced)
	sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

	# NOTE:
	# XGBClassifier enforces contiguous class labels. Our taxonomy is fixed to 0..7,
	# but some datasets can be missing an entire class (e.g., no BOTNET rows),
	# producing non-contiguous labels during training. The low-level training API
	# supports this as long as labels are < num_class.
	params = {
		"objective": "multi:softprob",
		"num_class": NUM_CLASSES,
		"eval_metric": "mlogloss",
		"max_depth": int(max_depth),
		"eta": float(learning_rate),
		"subsample": float(subsample),
		"colsample_bytree": float(colsample_bytree),
		"lambda": float(reg_lambda),
		"min_child_weight": float(min_child_weight),
		"seed": int(random_state),
		"verbosity": int(verbosity),
		"tree_method": "hist",
		"nthread": -1,
	}

	dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
	dtest = xgb.DMatrix(X_test, label=y_test)

	booster = xgb.train(
		params=params,
		dtrain=dtrain,
		num_boost_round=int(n_estimators),
		evals=[(dtest, "test")],
		verbose_eval=False,
	)

	y_proba = booster.predict(dtest)
	return booster, y_proba


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> None:
	acc = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average="macro")
	cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

	print(f"Accuracy: {acc:.6f}")
	print(f"Macro F1:  {f1:.6f}")
	print("Confusion matrix (rows=true, cols=pred):")
	print(cm)
	print("Classification report:")
	print(
		classification_report(
			y_true,
			y_pred,
			labels=list(range(NUM_CLASSES)),
			target_names=[ID_TO_LABEL[i] for i in range(NUM_CLASSES)],
			digits=4,
			zero_division=0,
		)
	)

	if y_proba is not None:
		maybe_print_roc_pr(y_true, y_proba)


def save_model(models_dir: Path, model: object) -> Path:
	models_dir.mkdir(parents=True, exist_ok=True)
	out_path = models_dir / "classifier.pkl"
	dump(model, out_path)
	return out_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Train XGBoost supervised classifier for Hybrid-IDS")
	parser.add_argument(
		"--project-root",
		type=str,
		default=None,
		help="Path to Hybrid-IDS folder (defaults to auto-detect).",
	)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--n-estimators", type=int, default=300)
	parser.add_argument("--max-depth", type=int, default=6)
	parser.add_argument("--learning-rate", type=float, default=0.1)
	parser.add_argument("--subsample", type=float, default=0.8)
	parser.add_argument("--colsample-bytree", type=float, default=0.8)
	parser.add_argument("--reg-lambda", type=float, default=1.0)
	parser.add_argument("--min-child-weight", type=float, default=1.0)
	parser.add_argument("--verbosity", type=int, default=1)
	parser.add_argument(
		"--max-train-samples",
		type=int,
		default=None,
		help="Optional stratified cap on training rows (useful for quick tests).",
	)
	parser.add_argument(
		"--max-test-samples",
		type=int,
		default=None,
		help="Optional stratified cap on test rows (useful for quick tests).",
	)
	parser.add_argument(
		"--no-curves",
		action="store_true",
		help="Skip optional ROC/PR AUC per-class summary.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Only verify artifacts load; do not train.",
	)
	args = parser.parse_args()

	project_root = Path(args.project_root).resolve() if args.project_root else None
	paths = get_paths(project_root)

	X_train, X_test, y_train, y_test = load_artifacts(paths.processed_dir)
	X_train, y_train = maybe_subsample(
		X_train,
		y_train,
		max_samples=args.max_train_samples,
		random_state=args.random_state,
	)
	X_test, y_test = maybe_subsample(
		X_test,
		y_test,
		max_samples=args.max_test_samples,
		random_state=args.random_state,
	)
	print(f"Loaded X_train={X_train.shape}, X_test={X_test.shape}")
	print(f"Loaded y_train={y_train.shape}, y_test={y_test.shape}")

	if args.dry_run:
		print("Dry-run complete.")
		return

	model, y_proba = train_xgboost(
		X_train,
		y_train,
		X_test,
		y_test,
		random_state=args.random_state,
		n_estimators=args.n_estimators,
		max_depth=args.max_depth,
		learning_rate=args.learning_rate,
		subsample=args.subsample,
		colsample_bytree=args.colsample_bytree,
		reg_lambda=args.reg_lambda,
		min_child_weight=args.min_child_weight,
		verbosity=args.verbosity,
	)

	y_pred = np.argmax(y_proba, axis=1)
	evaluate(y_test, y_pred, None if args.no_curves else y_proba)

	out_path = save_model(paths.models_dir, model)
	print(f"Saved model to: {out_path}")


if __name__ == "__main__":
	main()
