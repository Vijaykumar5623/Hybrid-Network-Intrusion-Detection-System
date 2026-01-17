"""Preprocessing pipeline for the Hybrid IDS project.

This script:
- Loads all CSVs from data/training/*.csv
- Concatenates into a single DataFrame
- Maps raw labels into a unified multi-class taxonomy
- Cleans numeric feature columns
- Splits supervised (80/20) and autoencoder BENIGN-only (70/30)
- Fits StandardScaler on supervised train set and exports scaled arrays
- Saves artifacts to data/processed/ and scaler to models/scaler.pkl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


LABEL_COL_CANDIDATES = ("Label", "label", "LABEL")


@dataclass(frozen=True)
class Paths:
	project_root: Path
	data_training_dir: Path
	data_processed_dir: Path
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
		data_training_dir=root / "data" / "training",
		data_processed_dir=root / "data" / "processed",
		models_dir=root / "models",
	)


def iter_training_csvs(training_dir: Path) -> list[Path]:
	return sorted(training_dir.glob("*.csv"))


def load_training_data(csv_paths: Iterable[Path]) -> pd.DataFrame:
	frames: list[pd.DataFrame] = []
	for path in csv_paths:
		try:
			frames.append(pd.read_csv(path, low_memory=False))
		except pd.errors.ParserError:
			# Fallback for rare malformed rows/quoting issues.
			frames.append(pd.read_csv(path, low_memory=False, engine="python"))
	if not frames:
		raise FileNotFoundError("No CSV files found under data/training/*.csv")
	return pd.concat(frames, ignore_index=True)


def find_label_column(df: pd.DataFrame) -> str:
	for candidate in LABEL_COL_CANDIDATES:
		if candidate in df.columns:
			return candidate
	raise KeyError(
		f"Could not find label column. Expected one of: {', '.join(LABEL_COL_CANDIDATES)}"
	)


def map_raw_label_to_category(raw: object) -> str:
	s = str(raw).strip()
	if not s or s.lower() == "nan":
		return "UNKNOWN"

	# Allow already-normalized taxonomy labels
	if s in LABEL_TO_ID:
		return s

	# CICIDS-style label mapping rules
	if s in ("FTP-Patator", "SSH-Patator"):
		return "BRUTE_FORCE"
	if s in ("DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"):
		return "DOS"
	if s == "DDoS":
		return "DDOS"
	if s == "PortScan":
		return "PORTSCAN"
	if s in ("Web Attack XSS", "Web Attack Brute Force", "Web Attack SQL Injection"):
		return "WEB_ATTACK"
	if s == "Bot":
		return "BOTNET"
	if s == "Infiltration":
		return "INFILTRATION"
	if s == "BENIGN":
		return "BENIGN"

	# Fallbacks for common variants
	lowered = s.lower()
	if lowered == "benign":
		return "BENIGN"
	if "ddos" == lowered:
		return "DDOS"
	if lowered.startswith("dos"):
		return "DOS"
	if "portscan" in lowered:
		return "PORTSCAN"
	if lowered.startswith("web attack"):
		return "WEB_ATTACK"
	if lowered == "bot":
		return "BOTNET"
	if "infiltration" in lowered:
		return "INFILTRATION"
	return "UNKNOWN"


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


def encode_labels(categories: pd.Series) -> np.ndarray:
	unknown = sorted(set(categories.unique()) - set(LABEL_TO_ID.keys()))
	if unknown:
		raise ValueError(
			"Found labels that do not map to the 8-class taxonomy: " + ", ".join(unknown)
		)
	return categories.map(LABEL_TO_ID).astype(np.int64).to_numpy()


def split_metadata_and_features(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, list[str]]:
	# Metadata: common flow identifier columns + all non-numeric columns except label
	common_meta = {
		"Flow ID",
		"Source IP",
		"Src IP",
		"Destination IP",
		"Dst IP",
		"Source Port",
		"Src Port",
		"Destination Port",
		"Dst Port",
		"Protocol",
		"Timestamp",
	}

	# Keep feature selection simple and robust: treat only well-known identifiers as metadata.
	# Many CICIDS feature columns can be read as object dtype (e.g., stray spaces); we still want them.
	meta_cols = [c for c in df.columns if c in common_meta]
	feature_cols = [c for c in df.columns if c != label_col and c not in set(meta_cols)]
	return df[meta_cols].copy() if meta_cols else df.iloc[:, 0:0].copy(), feature_cols


def clean_numeric_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
	x = df[feature_cols].copy()

	for col in feature_cols:
		x[col] = pd.to_numeric(x[col], errors="coerce")

	x = x.replace([np.inf, -np.inf], np.nan)

	# Fill NaNs using per-column median; fall back to 0 if a column is entirely NaN.
	medians = x.median(numeric_only=True)
	x = x.fillna(medians)
	x = x.fillna(0.0)

	return x


def maybe_stratify(y: np.ndarray) -> Optional[np.ndarray]:
	unique, counts = np.unique(y, return_counts=True)
	if len(unique) < 2:
		return None
	if np.min(counts) < 2:
		return None
	return y


def save_artifacts(
	processed_dir: Path,
	models_dir: Path,
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	X_train_ae: np.ndarray,
	X_test_ae: np.ndarray,
	scaler: StandardScaler,
) -> None:
	processed_dir.mkdir(parents=True, exist_ok=True)
	models_dir.mkdir(parents=True, exist_ok=True)

	np.save(processed_dir / "X_train.npy", X_train)
	np.save(processed_dir / "X_test.npy", X_test)
	np.save(processed_dir / "y_train.npy", y_train)
	np.save(processed_dir / "y_test.npy", y_test)
	np.save(processed_dir / "X_train_ae.npy", X_train_ae)
	np.save(processed_dir / "X_test_ae.npy", X_test_ae)

	dump(scaler, models_dir / "scaler.pkl")


def run_pipeline(project_root: Optional[Path] = None, random_state: int = 42) -> None:
	paths = get_paths(project_root)
	csvs = iter_training_csvs(paths.data_training_dir)
	df = load_training_data(csvs)

	label_col = find_label_column(df)
	categories = df[label_col].map(map_raw_label_to_category)

	# Keep only rows that map into the requested taxonomy
	df = df.loc[categories.isin(LABEL_TO_ID.keys())].copy()
	categories = categories.loc[df.index]

	_, feature_cols = split_metadata_and_features(df, label_col=label_col)
	if not feature_cols:
		raise ValueError("No ML feature columns detected after separating metadata.")

	x_df = clean_numeric_features(df, feature_cols)
	y = encode_labels(categories)

	X_supervised = x_df.to_numpy(dtype=np.float32, copy=False)
	y_supervised = y

	stratify = maybe_stratify(y_supervised)
	X_train_raw, X_test_raw, y_train, y_test = train_test_split(
		X_supervised,
		y_supervised,
		test_size=0.2,
		random_state=random_state,
		stratify=stratify,
	)

	# Autoencoder dataset: BENIGN only, split 70/30
	benign_mask = y_supervised == LABEL_TO_ID["BENIGN"]
	X_benign = X_supervised[benign_mask]
	if X_benign.shape[0] < 2:
		raise ValueError("Not enough BENIGN rows to create autoencoder dataset.")

	X_train_ae_raw, X_test_ae_raw = train_test_split(
		X_benign,
		test_size=0.3,
		random_state=random_state,
		shuffle=True,
	)

	scaler = StandardScaler()
	scaler.fit(X_train_raw)

	X_train = scaler.transform(X_train_raw).astype(np.float32, copy=False)
	X_test = scaler.transform(X_test_raw).astype(np.float32, copy=False)
	X_train_ae = scaler.transform(X_train_ae_raw).astype(np.float32, copy=False)
	X_test_ae = scaler.transform(X_test_ae_raw).astype(np.float32, copy=False)

	save_artifacts(
		processed_dir=paths.data_processed_dir,
		models_dir=paths.models_dir,
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
		X_train_ae=X_train_ae,
		X_test_ae=X_test_ae,
		scaler=scaler,
	)

	print(f"Loaded {len(csvs)} CSV(s) -> {len(df):,} rows")
	print(f"Features: {len(feature_cols)}")
	print("Label distribution (kept):")
	counts = categories.value_counts().sort_index()
	for name, count in counts.items():
		print(f"  {name:13s} -> {int(count):,}")
	print(f"Saved artifacts to: {paths.data_processed_dir}")
	print(f"Saved scaler to: {paths.models_dir / 'scaler.pkl'}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Hybrid-IDS preprocessing pipeline")
	parser.add_argument(
		"--project-root",
		type=str,
		default=None,
		help="Path to Hybrid-IDS folder (defaults to auto-detect).",
	)
	parser.add_argument("--random-state", type=int, default=42)
	args = parser.parse_args()

	project_root = Path(args.project_root).resolve() if args.project_root else None
	run_pipeline(project_root=project_root, random_state=args.random_state)


if __name__ == "__main__":
	main()
