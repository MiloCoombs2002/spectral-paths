"""Benchmark SpectralPathClassifier against binary-classification baselines."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, r2_score
from sklearn.model_selection import train_test_split

from spectral_paths.model import SpectralPathClassifier, SpectralPathRegressor
from spectral_paths.utils.helpers import _clip_probabilities

BlasThreadPolicy = Literal["auto", "none", "single", "manual"]


@dataclass(frozen=True)
class ClassifierConfig:
    """Model configuration used by the benchmark harness."""

    max_paths: int
    block_size_factor: int
    lambda_grid: tuple[float, ...]
    k_values: tuple[int, ...]
    early_stopping_patience: int
    early_stopping_tol: float
    greedy_subsample: int | None


class DatasetSpec(TypedDict):
    """Benchmark specification for a binary classification dataset."""

    label: str
    loader: str
    name: str
    version: int | None


DATASETS: dict[str, DatasetSpec] = {
    "sklearn:breast-cancer": {
        "label": "Breast Cancer Wisconsin",
        "loader": "sklearn",
        "name": "breast_cancer",
        "version": None,
    },
    "openml:phoneme": {
        "label": "OpenML Phoneme",
        "loader": "openml",
        "name": "phoneme",
        "version": 1,
    },
    "openml:wdbc": {
        "label": "OpenML WDBC",
        "loader": "openml",
        "name": "wdbc",
        "version": 1,
    },
    "openml:diabetes": {
        "label": "OpenML Diabetes",
        "loader": "openml",
        "name": "diabetes",
        "version": 1,
    },
    "openml:spambase": {
        "label": "OpenML Spambase",
        "loader": "openml",
        "name": "spambase",
        "version": 1,
    },
    "openml:banknote-authentication": {
        "label": "OpenML Banknote Authentication",
        "loader": "openml",
        "name": "banknote-authentication",
        "version": 1,
    },
    "openml:ilpd": {
        "label": "OpenML ILPD",
        "loader": "openml",
        "name": "ilpd",
        "version": 1,
    },
    "openml:qsar-biodeg": {
        "label": "OpenML QSAR Biodeg",
        "loader": "openml",
        "name": "qsar-biodeg",
        "version": 1,
    },
}

CURATED_DATASET_KEYS: tuple[str, ...] = (
    "sklearn:breast-cancer",
    "openml:banknote-authentication",
    "openml:diabetes",
    "openml:phoneme",
    "openml:qsar-biodeg",
    "openml:wdbc",
)

DEFAULT_CONFIG = ClassifierConfig(
    max_paths=256,
    block_size_factor=1,
    lambda_grid=tuple(np.logspace(-4, 0, 12)),
    k_values=(1, 2, 3, 4),
    early_stopping_patience=4,
    early_stopping_tol=1e-4,
    greedy_subsample=4000,
)

FAST_CONFIG = ClassifierConfig(
    max_paths=96,
    block_size_factor=1,
    lambda_grid=tuple(np.logspace(-4, -1, 6)),
    k_values=(1, 2, 3),
    early_stopping_patience=3,
    early_stopping_tol=1e-4,
    greedy_subsample=1500,
)


def available_dataset_choices() -> tuple[str, ...]:
    """Return CLI dataset choices."""
    return ("all", "curated", *DATASETS.keys())


def selected_dataset_keys(choice: str) -> list[str]:
    """Expand a CLI selection to concrete dataset keys."""
    if choice == "all":
        return list(DATASETS.keys())
    if choice == "curated":
        return list(CURATED_DATASET_KEYS)
    return [choice]


def _encode_binary_labels(y: np.ndarray) -> np.ndarray:
    """Map an arbitrary binary target vector to {0, 1}."""
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"Expected exactly 2 classes, got {classes.size}.")
    return (y == classes[1]).astype(int)


def _coerce_features_to_float_matrix(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert mixed-type feature matrices into float arrays for benchmarking."""
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        frame = pd.DataFrame(X)

    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        codes, _ = pd.factorize(series.astype(str), sort=True)
        frame[column] = codes.astype(float)

    return np.asarray(frame, dtype=float)


def _load_sklearn_dataset(_: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Load the local sklearn breast-cancer dataset."""
    dataset = load_breast_cancer()
    return "Breast Cancer Wisconsin", dataset.data.astype(float), dataset.target.astype(int)


def _load_openml_dataset(dataset_key: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Load a binary OpenML dataset with numeric features."""
    spec = DATASETS[dataset_key]
    X, y = fetch_openml(
        name=spec["name"],
        version=spec["version"],
        as_frame=True,
        parser="auto",
        return_X_y=True,
    )
    y_bin = _encode_binary_labels(np.asarray(y))
    return spec["label"], _coerce_features_to_float_matrix(X), y_bin


def benchmark_dataset(choice: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Return benchmark features and binary targets by dataset key."""
    spec = DATASETS[choice]
    loaders: dict[str, Callable[[str], tuple[str, np.ndarray, np.ndarray]]] = {
        "sklearn": _load_sklearn_dataset,
        "openml": _load_openml_dataset,
    }
    return loaders[spec["loader"]](choice)


def build_spectral_path_classifier(
    *, feature_count: int, config: ClassifierConfig, verbose: bool = False
) -> SpectralPathClassifier:
    """Construct the binary spectral-path classifier benchmark model."""
    return SpectralPathClassifier(
        max_paths=config.max_paths,
        block_size=config.block_size_factor * feature_count,
        lambda_grid=config.lambda_grid,
        k_values=config.k_values,
        scaler_type="robust_tanh",
        bound_percentiles=(5, 95),
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_tol=config.early_stopping_tol,
        greedy_subsample=config.greedy_subsample,
        verbose=verbose,
    )


def build_spectral_path_regressor(
    *, feature_count: int, config: ClassifierConfig, verbose: bool = False
) -> SpectralPathRegressor:
    """Construct the regression baseline used on binary labels."""
    return SpectralPathRegressor(
        max_paths=config.max_paths,
        block_size=config.block_size_factor * feature_count,
        lambda_grid=config.lambda_grid,
        k_values=config.k_values,
        scaler_type="robust_tanh",
        bound_percentiles=(5, 95),
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_tol=config.early_stopping_tol,
        greedy_subsample=config.greedy_subsample,
        verbose=verbose,
    )


def evaluate_regressor_baseline(
    model: SpectralPathRegressor,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Train the regressor on 0/1 labels and derive classification metrics."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    raw_preds = model.predict(X_te)
    prob_pos = _clip_probabilities(raw_preds, eps=eps)
    y_pred = (prob_pos >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(y_te, prob_pos, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "r2": float(r2_score(y_te, raw_preds)),
        "fit_time_sec": fit_time,
    }


def evaluate_classifier_model(
    name: str,
    model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> dict[str, float]:
    """Fit a probabilistic classifier and compute common evaluation metrics."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    prob_pos = np.asarray(model.predict_proba(X_te))[:, 1]
    y_pred = (prob_pos >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(y_te, prob_pos, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "r2": float(r2_score(y_te, prob_pos)),
        "fit_time_sec": fit_time,
    }


def benchmark_one_dataset(
    choice: str,
    *,
    mode: str = "default",
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Run the full classifier benchmark on one dataset."""
    label, X, y = benchmark_dataset(choice)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    config = FAST_CONFIG if mode == "fast" else DEFAULT_CONFIG
    feature_count = X.shape[1]

    results: dict[str, dict[str, float]] = {}
    results["SpectralPathClassifier"] = evaluate_classifier_model(
        "SpectralPathClassifier",
        build_spectral_path_classifier(
            feature_count=feature_count,
            config=config,
            verbose=verbose,
        ),
        X_tr,
        y_tr,
        X_te,
        y_te,
    )
    results["SpectralPathRegressor"] = evaluate_regressor_baseline(
        build_spectral_path_regressor(
            feature_count=feature_count,
            config=config,
            verbose=verbose,
        ),
        X_tr,
        y_tr.astype(float),
        X_te,
        y_te.astype(float),
    )
    results["LogisticRegression"] = evaluate_classifier_model(
        "LogisticRegression",
        LogisticRegression(max_iter=2000, solver="liblinear"),
        X_tr,
        y_tr,
        X_te,
        y_te,
    )
    results["RandomForestClassifier"] = evaluate_classifier_model(
        "RandomForestClassifier",
        RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=1),
        X_tr,
        y_tr,
        X_te,
        y_te,
    )
    results["HistGradientBoostingClassifier"] = evaluate_classifier_model(
        "HistGradientBoostingClassifier",
        HistGradientBoostingClassifier(random_state=random_state),
        X_tr,
        y_tr,
        X_te,
        y_te,
    )
    return label, results


def main() -> None:
    """Run the classification benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=available_dataset_choices(), default="all")
    parser.add_argument("--mode", choices=("default", "fast"), default="default")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    for dataset_choice in selected_dataset_keys(args.dataset):
        label, results = benchmark_one_dataset(
            dataset_choice, mode=args.mode, verbose=args.verbose
        )
        print(f"\nDataset: {label}")
        print(f"{'Model':32} {'Log loss':>10} {'Accuracy':>10} {'R²':>10} {'Fit (s)':>10}")
        for model_name, metrics in results.items():
            print(
                f"{model_name:32} "
                f"{metrics['log_loss']:>10.4f} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['r2']:>10.4f} "
                f"{metrics['fit_time_sec']:>10.3f}"
            )


if __name__ == "__main__":
    main()
