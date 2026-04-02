"""
CPU benchmark for SpectralPathRegressor using the example dataset sources.

This benchmark follows the same data sources used by `examples/openml.py`,
`examples/pmlb.py`, and `examples/uci_ml.py`. It reports cold vs warm timings for a
small set of model configurations.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, TypedDict

import numpy as np
from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from spectral_paths.model import SpectralPathRegressor

BlasThreadPolicy = Literal["auto", "none", "single", "manual"]


@dataclass(frozen=True)
class BenchmarkCase:
    """A single benchmark configuration."""

    name: str
    use_float32: bool
    lambda_parallel_workers: int
    blas_thread_policy: BlasThreadPolicy


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration used by the benchmark harness."""

    max_paths: int
    block_size_factor: int
    lambda_grid: tuple[float, ...]
    k_values: tuple[int, ...]
    early_stopping_patience: int
    early_stopping_tol: float
    greedy_subsample: int | None


class OpenMLSpec(TypedDict):
    """Benchmark specification for an OpenML dataset."""

    label: str
    openml_name: str
    version: int


class PMLBSpec(TypedDict):
    """Benchmark specification for a PMLB dataset."""

    label: str
    pmlb_name: str


class UCISpec(TypedDict):
    """Benchmark specification for a UCI dataset."""

    label: str
    uci_id: int


@dataclass
class UCIData:
    """Typed representation of the UCI data payload."""

    features: DataFrame
    targets: DataFrame


class UCIDatasetLike(Protocol):
    """Minimal protocol for fetched UCI dataset objects."""

    data: UCIData


OPENML_DATASETS: dict[str, OpenMLSpec] = {
    "concrete-slump": {
        "label": "OpenML Concrete Slump",
        "openml_name": "slump",
        "version": 2,
    },
    "yacht-hydrodynamics": {
        "label": "OpenML Yacht Hydrodynamics",
        "openml_name": "yacht_hydrodynamics",
        "version": 1,
    },
}

PMLB_DATASETS: dict[str, PMLBSpec] = {
    "echo-months": {
        "label": "PMLB Echo Cardiogram",
        "pmlb_name": "1199_BNG_echoMonths",
    },
    "wind-speed": {
        "label": "PMLB Wind Speed",
        "pmlb_name": "503_wind",
    },
}

UCI_DATASETS: dict[str, UCISpec] = {
    "energy-efficiency": {
        "label": "UCI Energy Efficiency",
        "uci_id": 242,
    },
    "concrete-strength": {
        "label": "UCI Concrete Compressive Strength",
        "uci_id": 165,
    },
}


def _load_openml_dataset(dataset_key: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Load an OpenML dataset matching `examples/openml.py`."""
    from sklearn.datasets import fetch_openml

    spec = OPENML_DATASETS[dataset_key]
    X, y = fetch_openml(
        name=spec["openml_name"],
        version=spec["version"],
        as_frame=False,
        parser="auto",
        return_X_y=True,
    )

    if spec["openml_name"] == "slump":
        y_arr = np.asarray(y, dtype=float)[:, 0].ravel()
    else:
        y_arr = np.asarray(y, dtype=float).ravel()

    return spec["label"], np.asarray(X, dtype=float), y_arr


def _load_pmlb_dataset(dataset_key: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Load a PMLB dataset matching `examples/pmlb.py`."""
    examples_dir = Path(__file__).resolve().parent
    if str(examples_dir) in sys.path:
        sys.path.remove(str(examples_dir))

    from pmlb import fetch_data  # type: ignore[attr-defined]

    spec = PMLB_DATASETS[dataset_key]
    df = fetch_data(spec["pmlb_name"], return_X_y=False)
    X = np.asarray(df.iloc[:, :-1].values, dtype=float)
    y = np.asarray(df.iloc[:, -1].values, dtype=float).ravel()
    return spec["label"], X, y


def _to_numpy_features_and_target(dataset: UCIDatasetLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert a fetched UCI dataset object to numpy arrays."""
    data = dataset.data
    X = np.asarray(data.features, dtype=float)
    y_raw = data.targets

    if hasattr(y_raw, "to_numpy"):
        y_array = y_raw.to_numpy(dtype=float)  # type: ignore[call-arg]
    else:
        y_array = np.asarray(y_raw, dtype=float)

    if y_array.ndim == 2:
        y = y_array[:, 0]
    else:
        y = y_array.ravel()

    return X, np.asarray(y, dtype=float).ravel()


def _load_uci_dataset(dataset_key: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Load a UCI dataset matching `examples/uci_ml.py`."""
    from ucimlrepo import fetch_ucirepo

    spec = UCI_DATASETS[dataset_key]
    dataset = fetch_ucirepo(id=spec["uci_id"])
    X, y = _to_numpy_features_and_target(dataset)
    return spec["label"], X, y


DATASET_LOADERS: dict[str, tuple[str, Callable[[str], tuple[str, np.ndarray, np.ndarray]]]] = {
    "openml": ("OpenML", _load_openml_dataset),
    "pmlb": ("PMLB", _load_pmlb_dataset),
    "uci": ("UCI", _load_uci_dataset),
}

EXAMPLE_MODE_CONFIGS: dict[str, ModelConfig] = {
    "openml": ModelConfig(
        max_paths=512,
        block_size_factor=1,
        lambda_grid=tuple(np.logspace(-5, -1, 25)),
        k_values=(1, 2, 3, 4),
        early_stopping_patience=5,
        early_stopping_tol=1e-3,
        greedy_subsample=None,
    ),
    "pmlb": ModelConfig(
        max_paths=512,
        block_size_factor=1,
        lambda_grid=tuple(np.logspace(-5, -1, 25)),
        k_values=(1, 2, 3, 4),
        early_stopping_patience=5,
        early_stopping_tol=1e-5,
        greedy_subsample=5000,
    ),
    "uci": ModelConfig(
        max_paths=512,
        block_size_factor=1,
        lambda_grid=tuple(np.logspace(-5, -1, 25)),
        k_values=(1, 2, 3, 4),
        early_stopping_patience=5,
        early_stopping_tol=1e-5,
        greedy_subsample=5000,
    ),
}

FAST_MODE_CONFIG = ModelConfig(
    max_paths=128,
    block_size_factor=1,
    lambda_grid=tuple(np.logspace(-4, -1, 8)),
    k_values=(1, 2, 3),
    early_stopping_patience=3,
    early_stopping_tol=1e-5,
    greedy_subsample=1500,
)


def available_dataset_choices() -> tuple[str, ...]:
    """Return CLI dataset choices."""
    return (
        "all",
        *(f"openml:{key}" for key in OPENML_DATASETS),
        *(f"pmlb:{key}" for key in PMLB_DATASETS),
        *(f"uci:{key}" for key in UCI_DATASETS),
    )


def selected_dataset_keys(choice: str) -> list[str]:
    """Expand a CLI selection to concrete dataset keys."""
    if choice == "all":
        return [
            *(f"openml:{key}" for key in OPENML_DATASETS),
            *(f"pmlb:{key}" for key in PMLB_DATASETS),
            *(f"uci:{key}" for key in UCI_DATASETS),
        ]
    return [choice]


def benchmark_dataset(choice: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Return benchmark features and target by provider-qualified dataset key."""
    provider, dataset_key = choice.split(":", maxsplit=1)
    _, loader = DATASET_LOADERS[provider]
    return loader(dataset_key)


def model_config_for(provider: str, mode: str) -> ModelConfig:
    """Return the model configuration for a provider/mode pair."""
    if mode == "example":
        return EXAMPLE_MODE_CONFIGS[provider]
    if mode == "fast":
        return FAST_MODE_CONFIG
    raise ValueError(f"Unknown benchmark mode {mode!r}")


def make_model(
    provider: str,
    X: np.ndarray,
    *,
    mode: str,
    use_float32: bool,
    lambda_parallel_workers: int,
    blas_thread_policy: BlasThreadPolicy,
) -> SpectralPathRegressor:
    """Create the model used across benchmark runs."""
    config = model_config_for(provider, mode)
    D = X.shape[1]
    greedy_subsample = config.greedy_subsample
    if greedy_subsample is not None:
        greedy_subsample = min(greedy_subsample, int(0.8 * X.shape[0]))

    return SpectralPathRegressor(
        max_paths=min(config.max_paths, 16 * D) if mode == "fast" else config.max_paths,
        block_size=max(1, config.block_size_factor * D),
        lambda_grid=config.lambda_grid,
        scaler_type="robust_tanh",
        bound_percentiles=(5.0, 95.0),
        verbose=False,
        k_values=config.k_values,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_tol=config.early_stopping_tol,
        greedy_subsample=greedy_subsample,
        use_float32=use_float32,
        lambda_parallel_workers=lambda_parallel_workers,
        blas_thread_policy=blas_thread_policy,
    )


def run_once(
    provider: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    use_float32: bool,
    lambda_parallel_workers: int,
    blas_thread_policy: BlasThreadPolicy,
) -> dict[str, float]:
    """Fit and score one model instance."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_model(
        provider,
        X,
        mode=mode,
        use_float32=use_float32,
        lambda_parallel_workers=lambda_parallel_workers,
        blas_thread_policy=blas_thread_policy,
    )

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    wall_sec = time.perf_counter() - t0
    y_hat = model.predict(X_te)
    r2 = float(r2_score(y_te, y_hat))
    val_r2 = (
        model.fit_report_.history[-1][3]
        if model.fit_report_ and model.fit_report_.history
        else float("nan")
    )
    selected_count = float(model.fit_report_.selected_count if model.fit_report_ else 0)

    return {
        "wall_sec": wall_sec,
        "selected_count": selected_count,
        "lambda_star": float(model.lambda_ or 0.0),
        "val_r2": float(val_r2),
        "test_r2": r2,
        "resolved_blas_threads": float(
            model.fit_report_.blas_threads.resolved_threads
            if model.fit_report_ and model.fit_report_.blas_threads.resolved_threads is not None
            else -1.0
        ),
    }


def run_case(dataset_choice: str, mode: str, case: BenchmarkCase) -> None:
    """Run and print one benchmark case."""
    provider = dataset_choice.split(":", maxsplit=1)[0]
    label, X, y = benchmark_dataset(dataset_choice)
    cold = run_once(
        provider,
        X,
        y,
        mode=mode,
        use_float32=case.use_float32,
        lambda_parallel_workers=case.lambda_parallel_workers,
        blas_thread_policy=case.blas_thread_policy,
    )
    warm = run_once(
        provider,
        X,
        y,
        mode=mode,
        use_float32=case.use_float32,
        lambda_parallel_workers=case.lambda_parallel_workers,
        blas_thread_policy=case.blas_thread_policy,
    )

    print(
        f"{label:>32} | {case.name:>20} | "
        f"cold={cold['wall_sec']:.3f}s warm={warm['wall_sec']:.3f}s | "
        f"paths={warm['selected_count']:.0f} λ={warm['lambda_star']:.5f} "
        "blas="
        f"{int(warm['resolved_blas_threads']) if warm['resolved_blas_threads'] >= 0 else 'sys'} | "
        f"val_r2={warm['val_r2']:.4f} test_r2={warm['test_r2']:.4f}"
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=available_dataset_choices(),
        default="all",
        help="Benchmark dataset selection.",
    )
    parser.add_argument(
        "--mode",
        choices=("example", "fast"),
        default="example",
        help="Use the original example-script hyperparameters or a reduced fast benchmark config.",
    )
    args = parser.parse_args()

    cases = (
        BenchmarkCase(
            "float64-auto",
            use_float32=False,
            lambda_parallel_workers=1,
            blas_thread_policy="auto",
        ),
        BenchmarkCase(
            "float64-single",
            use_float32=False,
            lambda_parallel_workers=1,
            blas_thread_policy="single",
        ),
        BenchmarkCase(
            "float64-none",
            use_float32=False,
            lambda_parallel_workers=1,
            blas_thread_policy="none",
        ),
        BenchmarkCase(
            "float64-lambda-x2",
            use_float32=False,
            lambda_parallel_workers=2,
            blas_thread_policy="auto",
        ),
        BenchmarkCase(
            "float32-auto",
            use_float32=True,
            lambda_parallel_workers=1,
            blas_thread_policy="auto",
        ),
    )

    print(f"mode={args.mode}")
    print("dataset                          | case                 | timings | metrics")
    for dataset_choice in selected_dataset_keys(args.dataset):
        for case in cases:
            run_case(dataset_choice, args.mode, case)


if __name__ == "__main__":
    main()
