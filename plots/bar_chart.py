"""Plot feature-importance bars for the current spectral path model."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from spectral_paths.model import SpectralPathRegressor

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25
MAX_PATHS = 25
LAMBDA_GRID = list(np.logspace(-5, -1, 25))
GREEDY_SUBSAMPLE = 5000
IMPORTANCE_SPLIT = "test"
IMPORTANCE_AGGREGATE = "mean_abs"
OUTPUT_PATH = Path("plots/bars.png")


def main() -> None:
    """Fit the current model and plot learned per-feature importances."""
    print("Fetching UCI dataset 165...")
    fetch_start = time.perf_counter()
    ds = fetch_ucirepo(id=165)
    print(f"Dataset fetch finished in {time.perf_counter() - fetch_start:.2f}s")

    X = np.asarray(ds.data.features, dtype=float)
    y = np.asarray(ds.data.targets, dtype=float).ravel()
    feature_names = [str(name) for name in ds.data.features.columns]
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
    print(
        f"Fitting model: max_paths={MAX_PATHS}, block_size={block_size}, "
        f"greedy_subsample={GREEDY_SUBSAMPLE}"
    )
    model = SpectralPathRegressor(
        max_paths=MAX_PATHS,
        block_size=block_size,
        lambda_grid=LAMBDA_GRID,
        l_max=None,
        scaler_type="robust_tanh",
        bound_percentiles=(5, 95),
        verbose=False,
        k_values=(1, 2),
        early_stopping_patience=5,
        early_stopping_tol=1e-5,
        greedy_subsample=GREEDY_SUBSAMPLE,
    )
    fit_start = time.perf_counter()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    print(f"Fit finished in {time.perf_counter() - fit_start:.2f}s")

    print("Evaluating on test split...")
    y_test_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    mae = float(mean_absolute_error(y_test, y_test_pred))
    r2 = float(r2_score(y_test, y_test_pred))
    sigma_test = float(y_test.std(ddof=0)) or 1.0
    nrmse_sigma = rmse / sigma_test

    print(f"\n=== {MAX_PATHS}-path spectral path model ===")
    print(f"λ* = {model.lambda_}")
    print(f"Test: R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f} | NRMSEσ={nrmse_sigma:.3f}")

    if IMPORTANCE_SPLIT == "train":
        X_importance = X_train
    elif IMPORTANCE_SPLIT == "val":
        X_importance = X_val
    elif IMPORTANCE_SPLIT == "test":
        X_importance = X_test
    else:
        raise ValueError("IMPORTANCE_SPLIT must be one of: 'train', 'val', 'test'.")

    print(
        "\nImportance is computed as the normalized mean absolute partial derivative "
        "of the fitted predictor with respect to each raw input feature."
    )
    print(
        f"Using split='{IMPORTANCE_SPLIT}' with aggregate='{IMPORTANCE_AGGREGATE}' "
        f"on {X_importance.shape[0]} rows."
    )

    importance_pct = model.feature_importance_gradient(
        X_importance,
        normalize=True,
        as_percentage=True,
        aggregate=IMPORTANCE_AGGREGATE,
    )
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance_pct": importance_pct}
    ).sort_values("importance_pct", ascending=True, ignore_index=True)

    print("\nFeature importance (%):")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(importance_df)

    model.print_feature_transforms(feature_names=feature_names)
    model.print_top_equation(n_terms=12, feature_names=feature_names, include_intercept=True)

    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 5.2),
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

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.barh(importance_df["feature"], importance_df["importance_pct"])
    ax.set_xlabel("Feature importance (\%)", fontsize=14)  # noqa
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
