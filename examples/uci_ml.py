"""Example module showing model usage with UCI ML Repository datasets."""

import time
from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict

from spectral_paths.model import SpectralPathRegressor

verbose = False
extra_verbose = False
print_metadata = False


class DatasetSpec(TypedDict):
    """DatasetSpec schema."""

    name: str
    uci_id: int

@dataclass
class Data:
    """."""
    features: DataFrame
    targets: DataFrame


datasets: list[DatasetSpec] = [
    {"name": "Energy Efficiency", "uci_id": 242},
    {"name": "Concrete Compressive Strength", "uci_id": 165},
    {"name": "Wine Quality", "uci_id": 186},
    {"name": "Phishing Websites", "uci_id": 327},
    {"name": "Superconductivity", "uci_id": 464},
]


def _to_numpy_features_and_target(dataset: dotdict) -> tuple[np.ndarray, np.ndarray]:
    """Convert a fetched UCI dataset object to numpy feature/target arrays."""
    data = cast(Data, dataset.data)
    X = np.asarray(data.features, dtype=float)
    y_raw = data.targets

    if hasattr(y_raw, "to_numpy"):
        y_array = y_raw.to_numpy(dtype=float)  # type: ignore[call-arg]
    else:
        y_array = np.asarray(y_raw, dtype=float)

    if y_array.ndim == 2:
        if y_array.shape[1] == 1:
            y = y_array[:, 0]
        else:
            # Some UCI datasets expose multiple targets; use the first target
            # column here to keep the example aligned with this regressor API.
            y = y_array[:, 0]
    else:
        y = y_array.ravel()

    return X, np.asarray(y, dtype=float).ravel()


if __name__ == "__main__":
    for dataset in datasets:
        print(f"Dataset: {dataset['name']}", end="")
        uci_dataset: dotdict = fetch_ucirepo(id=dataset["uci_id"])

        if print_metadata:
            print(uci_dataset.metadata)
            print(uci_dataset.variables)

        X, y = _to_numpy_features_and_target(uci_dataset)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        print(f". Training rows {X_tr.shape[0]}")

        D = X.shape[1]

        mdl = SpectralPathRegressor(
            max_paths=512,
            block_size=1 * D,
            lambda_grid=list(np.logspace(-5, -1, 25)),
            l_max=None,
            scaler_type="robust_tanh",
            bound_percentiles=(5, 95),
            verbose=verbose,
            k_values=(1,2,3,4),
            early_stopping_patience=5,
            early_stopping_tol= 1e-5,
            greedy_subsample=5000
        )

        if extra_verbose:
            print("Training model...")

        t0 = time.perf_counter()
        mdl.fit(X_tr, y_tr)
        t1 = time.perf_counter()

        print(f"lambda: {mdl.lambda_}")
        yhat = mdl.predict(X_te)
        r2 = float(r2_score(y_te, yhat))

        print(f"{'R²':>10} {'Train time (s)':>16}")
        print(f"{r2:>10.4f} {t1 - t0:>16.3f}\n")

        if mdl.fit_report_ is not None and extra_verbose:
            print("\n=== Timing ===")
            print(f"Greedy selection: {mdl.fit_report_.greedy_time_sec:.3f}s")
            print(f"Final solve:      {mdl.fit_report_.final_solve_time_sec:.3f}s")
            print(f"Selected cols:    {mdl.fit_report_.selected_count} (+ intercept)")
            print(f"λ*:               {mdl.fit_report_.lambda_star}")
            print(f"Stopped early:    {mdl.fit_report_.stopped_early}")
