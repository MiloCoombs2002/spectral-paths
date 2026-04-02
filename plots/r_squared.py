"""Plot train/validation performance curves for the current spectral path model."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from spectral_paths.model import SpectralPathRegressor

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25
M_MAX = 40
LAMBDA_GRID = list(np.logspace(-5, -1, 25))
GREEDY_SUBSAMPLE = 5000
OUTPUT_PATH = Path("plots/concrete_train_val_curves.png")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return RMSE and R^2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2


def main() -> None:
    """Run a capacity sweep using the current SpectralPathRegressor API."""
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

    sigma_train = float(y_train.std(ddof=0)) or 1.0
    sigma_val = float(y_val.std(ddof=0)) or 1.0
    sigma_test = float(y_test.std(ddof=0)) or 1.0

    n_features = X.shape[1]
    curve_rows: list[dict[str, float | int]] = []
    fitted_models: dict[int, SpectralPathRegressor] = {}

    for max_paths in range(1, M_MAX + 1):
        block_size = min(n_features, max_paths)
        print(
            f"[Sweep {max_paths:02d}/{M_MAX:02d}] "
            f"starting fit: max_paths={max_paths}, block_size={block_size}, "
            f"greedy_subsample={GREEDY_SUBSAMPLE}"
        )
        model = SpectralPathRegressor(
            max_paths=max_paths,
            block_size=block_size,
            lambda_grid=LAMBDA_GRID,
            l_max=None,
            scaler_type="robust_tanh",
            bound_percentiles=(5, 95),
            verbose=False,
            k_values=(1,),
            early_stopping_patience=5,
            early_stopping_tol=1e-5,
            greedy_subsample=GREEDY_SUBSAMPLE,
        )
        fit_start = time.perf_counter()
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        fit_time = time.perf_counter() - fit_start
        print(f"[Sweep {max_paths:02d}/{M_MAX:02d}] fit finished in {fit_time:.2f}s")

        predict_start = time.perf_counter()
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        predict_time = time.perf_counter() - predict_start

        rmse_train, r2_train = metrics(y_train, y_train_pred)
        rmse_val, r2_val = metrics(y_val, y_val_pred)

        total_paths = 1 + (model.fit_report_.selected_count if model.fit_report_ else max_paths)
        print(
            f"[Sweep {max_paths:02d}/{M_MAX:02d}] "
            f"selected_paths={total_paths}, lambda={model.lambda_}, "
            f"train_r2={r2_train:.4f}, val_r2={r2_val:.4f}, "
            f"predict_time={predict_time:.2f}s"
        )
        curve_rows.append(
            {
                "M": int(total_paths),
                "max_paths": max_paths,
                "lambda": float(model.lambda_ or model.lambda_grid[0]),
                "r2_train": r2_train,
                "nrmse_sigma_train": rmse_train / sigma_train,
                "r2_val": r2_val,
                "nrmse_sigma_val": rmse_val / sigma_val,
            }
        )
        fitted_models[max_paths] = model

    print("Capacity sweep complete. Selecting best validation model...")
    df_curve = pd.DataFrame(curve_rows).sort_values("M").reset_index(drop=True)
    best_row = df_curve.loc[df_curve["r2_val"].idxmax()]
    best_max_paths = int(best_row["max_paths"])
    best_model = fitted_models[best_max_paths]
    best_selected_total_paths = (
        1 + (best_model.fit_report_.selected_count if best_model.fit_report_ else best_max_paths)
    )
    print(
        f"Best model uses max_paths={best_max_paths}, selected_paths={best_selected_total_paths}, "
        f"lambda={best_model.lambda_}, val_r2={best_row['r2_val']:.4f}"
    )

    print("Evaluating best model on train, validation, and test splits...")
    y_train_hat = best_model.predict(X_train)
    y_val_hat = best_model.predict(X_val)
    y_test_hat = best_model.predict(X_test)

    rmse_train_star, r2_train_star = metrics(y_train, y_train_hat)
    rmse_val_star, r2_val_star = metrics(y_val, y_val_hat)
    rmse_test_star, r2_test_star = metrics(y_test, y_test_hat)

    test_row = pd.DataFrame(
        [
            {
                "M*": best_selected_total_paths,
                "max_paths": best_max_paths,
                "lambda*": float(best_model.lambda_ or best_model.lambda_grid[0]),
                "R2_train": r2_train_star,
                "R2_val": r2_val_star,
                "R2_test": r2_test_star,
                "NRMSEσ_train": rmse_train_star / sigma_train,
                "NRMSEσ_val": rmse_val_star / sigma_val,
                "NRMSEσ_test": rmse_test_star / sigma_test,
            }
        ]
    )

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

    fig, ax1 = plt.subplots(figsize=(8, 4))

    line_train_r2, = ax1.plot(
        df_curve["M"], df_curve["r2_train"], marker="o", linewidth=1,
        color="tab:blue", label="Train $R^2$"
    )
    line_val_r2, = ax1.plot(
        df_curve["M"], df_curve["r2_val"], marker="o", linewidth=1,
        color="#6baed6", linestyle="--", label="Val $R^2$"
    )
    ax1.set_xlabel("Number of paths", color="black", fontsize=14)
    ax1.set_ylabel("$R^2$", color="black", fontsize=14)
    ax1.tick_params(axis="x", labelcolor="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.spines["top"].set_visible(False)
    ax1.grid(False)

    ax2 = ax1.twinx()
    line_train_nrmse, = ax2.plot(
        df_curve["M"], df_curve["nrmse_sigma_train"], marker="o", linewidth=1,
        color="tab:orange", label="Train NRMSE$_\\sigma$"
    )
    line_val_nrmse, = ax2.plot(
        df_curve["M"], df_curve["nrmse_sigma_val"], marker="o", linewidth=1,
        color="#fdae6b", linestyle="--", label="Val NRMSE$_\\sigma$"
    )
    ax2.set_ylabel("NRMSE$_\\sigma$", color="black", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.spines["top"].set_visible(False)

    lines = [line_train_r2, line_val_r2, line_train_nrmse, line_val_nrmse]
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels, loc="center right", frameon=False, fontsize=14)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)

    print(f"Saved plot to {OUTPUT_PATH}")
    print("\n=== Selected capacity (validation-optimal) and TEST results ===")
    print(test_row.to_string(index=False))


if __name__ == "__main__":
    main()
