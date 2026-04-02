"""Plot test R^2 box plots for the UCI ML datasets used in the examples."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict

from spectral_paths.model import SpectralPathRegressor


class DatasetSpec(TypedDict):
    """Specification for one UCI ML dataset."""

    name: str
    uci_id: int
    target_index: int


DATASETS: list[DatasetSpec] = [
    {"name": "Concrete Compressive Strength", "uci_id": 165, "target_index": 0},
]

TEST_SIZE = 0.2
NUM_SPLITS = 10
SEED_START = 42
OUTPUT_PATH = Path("plots/uci_ml_box_plots.png")


@dataclass
class Data:
    """UCI dataset data payload."""

    features: DataFrame
    targets: DataFrame


def load_uci_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Fetch one UCI ML dataset and coerce it into numpy arrays."""
    uci_dataset: dotdict = fetch_ucirepo(id=spec["uci_id"])
    data = cast(Data, uci_dataset.data)

    X_array = np.asarray(data.features, dtype=float)
    y_raw = data.targets
    if hasattr(y_raw, "to_numpy"):
        y_array = y_raw.to_numpy(dtype=float)  # type: ignore[call-arg]
    else:
        y_array = np.asarray(y_raw, dtype=float)

    if y_array.ndim == 2:
        if spec["target_index"] >= y_array.shape[1]:
            raise ValueError(
                f"target_index={spec['target_index']} is out of bounds for "
                f"{spec['name']} with {y_array.shape[1]} targets."
            )
        y_vector = y_array[:, spec["target_index"]]
    else:
        if spec["target_index"] != 0:
            raise ValueError(
                f"{spec['name']} has a single target, so target_index must be 0."
            )
        y_vector = y_array.ravel()

    return X_array, np.asarray(y_vector, dtype=float).ravel()


def fit_and_score(X: np.ndarray, y: np.ndarray, random_state: int) -> float:
    """Train the spectral path model on one split and return test R^2."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state
    )

    n_features = X.shape[1]
    model = SpectralPathRegressor(
        max_paths=512,
        block_size=n_features,
        lambda_grid=list(np.logspace(-5, -1, 25)),
        l_max=None,
        scaler_type="robust_tanh",
        bound_percentiles=(5, 95),
        verbose=False,
        k_values=(1, 2, 3, 4),
        early_stopping_patience=5,
        early_stopping_tol=1e-3,
        greedy_subsample=5000,
    )
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    return float(r2_score(y_te, y_hat))


def main() -> None:
    """Generate one horizontal box plot per UCI ML dataset."""
    all_scores: list[list[float]] = []
    labels: list[str] = []

    for spec in DATASETS:
        print(f"Fetching dataset: {spec['name']}")
        fetch_start = time.perf_counter()
        X, y = load_uci_dataset(spec)
        print(
            f"Loaded {spec['name']} in {time.perf_counter() - fetch_start:.2f}s "
            f"with X.shape={X.shape}, y.shape={y.shape}"
        )

        dataset_scores: list[float] = []
        for split_idx in range(NUM_SPLITS):
            seed = SEED_START + split_idx
            print(
                f"[{spec['name']}] split {split_idx + 1}/{NUM_SPLITS} "
                f"(random_state={seed})"
            )
            score = fit_and_score(X, y, random_state=seed)
            dataset_scores.append(score)
            print(f"[{spec['name']}] test R^2 = {score:.4f}")

        all_scores.append(dataset_scores)
        labels.append(spec["name"])

    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 5.5),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "savefig.dpi": 300,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 2))
    ax.boxplot(all_scores, vert=False, patch_artist=False)
    ax.set_xlabel(r"Test $R^2$", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
