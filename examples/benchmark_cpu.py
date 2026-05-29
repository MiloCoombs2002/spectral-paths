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
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, TypedDict

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from spectral_paths.model import SpectralPathRegressor
from spectral_paths.utils.helpers import _primitive_and_order

if TYPE_CHECKING:
    from pandas import DataFrame
else:
    DataFrame = Any

BlasThreadPolicy = Literal["auto", "none", "single", "manual"]


@dataclass(frozen=True)
class BenchmarkCase:
    """A single benchmark configuration."""

    name: str
    regularization_mode: Literal["uniform", "complexity_weighted"]
    path_complexity: Literal["total_order", "sparsity", "harmonic_order"] | None
    complexity_penalty_schedule: Literal["linear", "exponential"] | None
    complexity_penalty_strength: float | None
    use_float32: bool = False
    lambda_parallel_workers: int = 1
    blas_thread_policy: BlasThreadPolicy = "auto"


class BenchmarkResult(TypedDict):
    """One benchmark result row."""

    dataset: str
    case: str
    regularization: str
    cold_sec: float
    warm_sec: float
    selected_count: float
    lambda_star: float
    resolved_blas_threads: float
    val_r2: float
    test_r2: float
    delta_vs_baseline: float
    mean_total_order: float
    max_total_order: float
    mean_sparsity: float
    max_sparsity: float
    mean_harmonic_order: float
    max_harmonic_order: float


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
    adaptive_block_size: bool = True
    min_block_size: int = 1


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
    "wine-quality": {
        "label": "UCI Wine Quality",
        "uci_id": 186,
    },
    "phishing-websites": {
        "label": "UCI Phishing Websites",
        "uci_id": 327,
    },
    "superconductivity": {
        "label": "UCI Superconductivity",
        "uci_id": 464,
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

OVERGROW_MODE_CONFIGS: dict[str, ModelConfig] = {
    "openml": ModelConfig(
        max_paths=768,
        block_size_factor=1,
        lambda_grid=tuple(np.logspace(-5, -1, 25)),
        k_values=(1, 2, 3, 4),
        early_stopping_patience=12,
        early_stopping_tol=1e-5,
        greedy_subsample=None,
        adaptive_block_size=True,
        min_block_size=1,
    ),
    "pmlb": ModelConfig(
        max_paths=768,
        block_size_factor=1,
        lambda_grid=tuple(np.logspace(-5, -1, 25)),
        k_values=(1, 2, 3, 4),
        early_stopping_patience=12,
        early_stopping_tol=1e-5,
        greedy_subsample=5000,
        adaptive_block_size=True,
        min_block_size=1,
    ),
    "uci": ModelConfig(
        max_paths=768,
        block_size_factor=1,
        lambda_grid=tuple(np.logspace(-5, -1, 25)),
        k_values=(1, 2, 3, 4),
        early_stopping_patience=12,
        early_stopping_tol=1e-5,
        greedy_subsample=5000,
        adaptive_block_size=True,
        min_block_size=1,
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
        "all-uci",
        "all-broad",
        *(f"openml:{key}" for key in OPENML_DATASETS),
        *(f"pmlb:{key}" for key in PMLB_DATASETS),
        *(f"uci:{key}" for key in UCI_DATASETS),
    )


def selected_dataset_keys(choice: str) -> list[str]:
    """Expand a CLI selection to concrete dataset keys."""
    if choice == "all":
        return ["openml:concrete-slump", "openml:yacht-hydrodynamics"]
    if choice == "all-uci":
        return [f"uci:{key}" for key in UCI_DATASETS]
    if choice == "all-broad":
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
    if mode == "overgrow":
        return OVERGROW_MODE_CONFIGS[provider]
    if mode == "fast":
        return FAST_MODE_CONFIG
    raise ValueError(f"Unknown benchmark mode {mode!r}")


def make_model(
    provider: str,
    X: np.ndarray,
    *,
    mode: str,
    case: BenchmarkCase,
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
        adaptive_block_size=config.adaptive_block_size,
        min_block_size=config.min_block_size,
        use_float32=case.use_float32,
        lambda_parallel_workers=case.lambda_parallel_workers,
        blas_thread_policy=case.blas_thread_policy,
        regularization_mode=case.regularization_mode,
        path_complexity=case.path_complexity or "total_order",
        complexity_penalty_schedule=case.complexity_penalty_schedule or "exponential",
        complexity_penalty_strength=case.complexity_penalty_strength or 1.0,
    )


def run_once(
    provider: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    case: BenchmarkCase,
) -> dict[str, float]:
    """Fit and score one model instance."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_model(
        provider,
        X,
        mode=mode,
        case=case,
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
    selected_paths = model.selected_paths_ or []

    total_orders = np.asarray(
        [sum(abs(path_value) for path_value in path) for path in selected_paths],
        dtype=float,
    )
    sparsities = np.asarray(
        [sum(1 for path_value in path if path_value != 0) for path in selected_paths],
        dtype=float,
    )
    harmonic_orders = np.asarray(
        [float(_primitive_and_order(path)[1]) for path in selected_paths],
        dtype=float,
    )

    def mean_or_zero(values: np.ndarray) -> float:
        return float(values.mean()) if values.size > 0 else 0.0

    def max_or_zero(values: np.ndarray) -> float:
        return float(values.max()) if values.size > 0 else 0.0

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
        "mean_total_order": mean_or_zero(total_orders),
        "max_total_order": max_or_zero(total_orders),
        "mean_sparsity": mean_or_zero(sparsities),
        "max_sparsity": max_or_zero(sparsities),
        "mean_harmonic_order": mean_or_zero(harmonic_orders),
        "max_harmonic_order": max_or_zero(harmonic_orders),
    }


def _regularization_label(case: BenchmarkCase) -> str:
    """Return a compact regularization label for one benchmark case."""
    if case.regularization_mode == "complexity_weighted":
        return (
            f"{case.path_complexity}:{case.complexity_penalty_schedule}:"
            f"{case.complexity_penalty_strength:.2f}"
        )
    return case.regularization_mode


def run_case(
    dataset_choice: str,
    mode: str,
    case: BenchmarkCase,
    *,
    baseline_test_r2: float | None = None,
) -> BenchmarkResult:
    """Run one benchmark case and return a structured result row."""
    provider = dataset_choice.split(":", maxsplit=1)[0]
    label, X, y = benchmark_dataset(dataset_choice)
    cold = run_once(
        provider,
        X,
        y,
        mode=mode,
        case=case,
    )
    warm = run_once(
        provider,
        X,
        y,
        mode=mode,
        case=case,
    )

    delta_vs_baseline = 0.0
    if baseline_test_r2 is not None:
        delta_vs_baseline = warm["test_r2"] - baseline_test_r2

    return {
        "dataset": label,
        "case": case.name,
        "regularization": _regularization_label(case),
        "cold_sec": cold["wall_sec"],
        "warm_sec": warm["wall_sec"],
        "selected_count": warm["selected_count"],
        "lambda_star": warm["lambda_star"],
        "resolved_blas_threads": warm["resolved_blas_threads"],
        "val_r2": warm["val_r2"],
        "test_r2": warm["test_r2"],
        "delta_vs_baseline": delta_vs_baseline,
        "mean_total_order": warm["mean_total_order"],
        "max_total_order": warm["max_total_order"],
        "mean_sparsity": warm["mean_sparsity"],
        "max_sparsity": warm["max_sparsity"],
        "mean_harmonic_order": warm["mean_harmonic_order"],
        "max_harmonic_order": warm["max_harmonic_order"],
    }


def print_result_row(result: BenchmarkResult) -> None:
    """Print one benchmark result row."""
    print(
        f"{result['dataset']:>32} | {result['case']:>28} | "
        f"cold={result['cold_sec']:.3f}s warm={result['warm_sec']:.3f}s | "
        f"paths={result['selected_count']:.0f} λ={result['lambda_star']:.5f} "
        "blas="
        f"{int(result['resolved_blas_threads']) if result['resolved_blas_threads'] >= 0 else 'sys'} | "
        f"reg={result['regularization']} | val_r2={result['val_r2']:.4f} "
        f"test_r2={result['test_r2']:.4f} Δ={result['delta_vs_baseline']:+.4f} "
        f"| meanL={result['mean_total_order']:.2f} maxL={result['max_total_order']:.0f}"
    )


def results_to_markdown(
    results: list[BenchmarkResult],
    *,
    mode: str,
    dataset_choice: str,
) -> str:
    """Render benchmark results as a Markdown table."""
    lines = [
        "# Complexity-Aware Ridge Benchmark Results",
        "",
        f"- mode: `{mode}`",
        f"- dataset selection: `{dataset_choice}`",
        "",
        "| Dataset | Case | Regularization | Cold (s) | Warm (s) | Paths | Lambda | Test R2 | Delta vs Baseline | Mean Total Order | Max Total Order | Mean Sparsity | Max Sparsity | Mean Harmonic Order | Max Harmonic Order |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result['dataset']} | "
            f"{result['case']} | "
            f"`{result['regularization']}` | "
            f"{result['cold_sec']:.3f} | "
            f"{result['warm_sec']:.3f} | "
            f"{result['selected_count']:.0f} | "
            f"{result['lambda_star']:.5f} | "
            f"{result['test_r2']:.4f} | "
            f"{result['delta_vs_baseline']:+.4f} | "
            f"{result['mean_total_order']:.2f} | "
            f"{result['max_total_order']:.0f} | "
            f"{result['mean_sparsity']:.2f} | "
            f"{result['max_sparsity']:.0f} | "
            f"{result['mean_harmonic_order']:.2f} | "
            f"{result['max_harmonic_order']:.0f} |"
        )

    return "\n".join(lines) + "\n"


def benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Return the first-pass complexity-aware regularization experiment matrix."""
    cases: list[BenchmarkCase] = [
        BenchmarkCase(
            name="baseline-uniform",
            regularization_mode="uniform",
            path_complexity=None,
            complexity_penalty_schedule=None,
            complexity_penalty_strength=None,
        )
    ]

    for path_complexity in ("total_order", "sparsity", "harmonic_order"):
        for schedule in ("linear", "exponential"):
            cases.append(
                BenchmarkCase(
                    name=f"{path_complexity}-{schedule}",
                    regularization_mode="complexity_weighted",
                    path_complexity=path_complexity,
                    complexity_penalty_schedule=schedule,
                    complexity_penalty_strength=1.0,
                )
            )
    return tuple(cases)


def overgrow_strength_sweep_cases() -> tuple[BenchmarkCase, ...]:
    """Return the targeted overgrow experiment matrix."""
    cases: list[BenchmarkCase] = [
        BenchmarkCase(
            name="baseline-uniform",
            regularization_mode="uniform",
            path_complexity=None,
            complexity_penalty_schedule=None,
            complexity_penalty_strength=None,
            blas_thread_policy="auto",
        )
    ]

    for path_complexity in ("sparsity", "total_order"):
        for strength in (1.0, 2.0, 4.0, 8.0):
            cases.append(
                BenchmarkCase(
                    name=f"{path_complexity}-exp-{strength:g}",
                    regularization_mode="complexity_weighted",
                    path_complexity=path_complexity,
                    complexity_penalty_schedule="exponential",
                    complexity_penalty_strength=strength,
                    blas_thread_policy="auto",
                )
            )
    return tuple(cases)


def focused_overgrow_cases() -> tuple[BenchmarkCase, ...]:
    """Return a focused overgrow sweep over the strongest exponential families."""
    cases: list[BenchmarkCase] = [
        BenchmarkCase(
            name="baseline-uniform",
            regularization_mode="uniform",
            path_complexity=None,
            complexity_penalty_schedule=None,
            complexity_penalty_strength=None,
            blas_thread_policy="auto",
        )
    ]

    for strength in (3.0, 4.0, 6.0, 8.0, 10.0):
        cases.append(
            BenchmarkCase(
                name=f"total_order-exp-{strength:g}",
                regularization_mode="complexity_weighted",
                path_complexity="total_order",
                complexity_penalty_schedule="exponential",
                complexity_penalty_strength=strength,
                blas_thread_policy="auto",
            )
        )

    for strength in (4.0, 6.0, 8.0, 10.0):
        cases.append(
            BenchmarkCase(
                name=f"sparsity-exp-{strength:g}",
                regularization_mode="complexity_weighted",
                path_complexity="sparsity",
                complexity_penalty_schedule="exponential",
                complexity_penalty_strength=strength,
                blas_thread_policy="auto",
            )
        )

    return tuple(cases)


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
        choices=("example", "fast", "overgrow"),
        default="example",
        help="Use the original example-script hyperparameters or a reduced fast benchmark config.",
    )
    parser.add_argument(
        "--case-set",
        choices=("baseline", "full", "overgrow-strength", "focused-overgrow"),
        default="full",
        help="Choose which regularization cases to run.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to save the benchmark results as a Markdown table.",
    )
    args = parser.parse_args()

    if args.case_set == "baseline":
        cases = benchmark_cases()[:1]
    elif args.case_set == "focused-overgrow":
        cases = focused_overgrow_cases()
    elif args.case_set == "overgrow-strength":
        cases = overgrow_strength_sweep_cases()
    else:
        cases = benchmark_cases()
    results: list[BenchmarkResult] = []

    print(f"mode={args.mode}")
    print("dataset                          | case                         | timings | metrics")
    for dataset_choice in selected_dataset_keys(args.dataset):
        baseline_result: BenchmarkResult | None = None
        for idx, case in enumerate(cases):
            result = run_case(
                dataset_choice,
                args.mode,
                case,
                baseline_test_r2=(
                    baseline_result["test_r2"] if baseline_result is not None else None
                ),
            )
            if idx == 0:
                baseline_result = result
                result["delta_vs_baseline"] = 0.0
            print_result_row(result)
            results.append(result)

    if args.output_markdown is not None:
        output_path = args.output_markdown
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            results_to_markdown(results, mode=args.mode, dataset_choice=args.dataset),
            encoding="utf-8",
        )
        print(f"\nSaved Markdown report to {output_path}")


if __name__ == "__main__":
    main()
