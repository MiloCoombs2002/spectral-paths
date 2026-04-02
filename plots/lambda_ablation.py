"""Plot test R^2 across lambda values for the concrete compressive dataset."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from spectral_paths.model import SpectralPathRegressor

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25
MAX_PATHS = 512
LAMBDA_GRID = list(np.logspace(-5, -2, 25))
GREEDY_SUBSAMPLE = 5000
OUTPUT_PATH = Path("plots/concrete_lambda_ablation.png")


def main() -> None:
    """Fit one model per lambda and plot test R^2 versus lambda."""
    print("Fetching UCI dataset 165...")
    fetch_start = time.perf_counter()
    ds = fetch_ucirepo(id=165)
    print(f"Dataset fetch finished in {time.perf_counter() - fetch_start:.2f}s")

    X = np.asarray(ds.data.features, dtype=float)
    y = np.asarray(ds.data.targets, dtype=float).ravel()
    print(f"Loaded arrays: X.shape={X.shape}, y.shape={y.shape}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    print(
        "Split sizes: "
        f"train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}"
    )

    block_size = min(X.shape[1], MAX_PATHS)
    curve_rows: list[dict[str, float]] = []

    for idx, lambda_value in enumerate(LAMBDA_GRID, start=1):
        print(
            f"[Sweep {idx:02d}/{len(LAMBDA_GRID):02d}] "
            f"fitting lambda={lambda_value:.6g}, max_paths={MAX_PATHS}, "
            f"block_size={block_size}, greedy_subsample={GREEDY_SUBSAMPLE}"
        )
        model = SpectralPathRegressor(
            max_paths=MAX_PATHS,
            block_size=X.shape[1],
            lambda_grid=[lambda_value],
            l_max=None,
            scaler_type="robust_tanh",
            bound_percentiles=(5, 95),
            verbose=False,
            k_values=(1,2,3,4),
            early_stopping_patience=5,
            early_stopping_tol=1e-5,
            greedy_subsample=GREEDY_SUBSAMPLE,
        )
        fit_start = time.perf_counter()
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        fit_time = time.perf_counter() - fit_start

        y_test_pred = model.predict(X_test)
        test_r2 = float(r2_score(y_test, y_test_pred))
        selected_paths = 1 + (
            model.fit_report_.selected_count if model.fit_report_ else MAX_PATHS
        )
        print(
            f"[Sweep {idx:02d}/{len(LAMBDA_GRID):02d}] "
            f"fit finished in {fit_time:.2f}s | "
            f"selected_paths={selected_paths} | test_r2={test_r2:.4f}"
        )
        curve_rows.append(
            {
                "lambda": float(lambda_value),
                "test_r2": test_r2,
                "selected_paths": float(selected_paths),
            }
        )

    df_curve = pd.DataFrame(curve_rows).sort_values("lambda").reset_index(drop=True)
    best_row = df_curve.loc[df_curve["test_r2"].idxmax()]

    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 4),
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

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(
        df_curve["lambda"],
        df_curve["test_r2"],
        marker="o",
        linewidth=1,
        color="tab:blue",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$", color="black", fontsize=14)
    ax.set_ylabel(r"$R^2$", color="black", fontsize=14)
    ax.tick_params(axis="x", labelcolor="black")
    ax.tick_params(axis="y", labelcolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)

    print(f"Saved plot to {OUTPUT_PATH}")
    print("\n=== Test R^2 by lambda ===")
    print(df_curve.to_string(index=False))
    print(
        "\nBest lambda on test sweep: "
        f"lambda={best_row['lambda']:.6g}, test_r2={best_row['test_r2']:.4f}, "
        f"selected_paths={int(best_row['selected_paths'])}"
    )


if __name__ == "__main__":
    main()
