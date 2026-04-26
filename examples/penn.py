"""Example module showing model usage with UCI datasets."""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from spectral_paths.model import SpectralPathRegressor

# Avoid shadowing the third-party `pmlb` package with this file's name.
_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) in sys.path:
    sys.path.remove(str(_examples_dir))
from pmlb import fetch_data  # type: ignore[attr-defined, E402]  # noqa: E402

verbose = False
extra_verbose = False


datasets: list[dict[str, str]] = [
    {"name": "Echo Cardiogram", "pmlb_name": "1199_BNG_echoMonths"},
    {"name": "Wind Speed", "pmlb_name": "503_wind"},
    {"name": "CPU Utilisation", "pmlb_name": "197_cpu_act"}
]


if __name__ == "__main__":
    for dataset in datasets:

        print(f"Dataset: {dataset["name"]}", end="")
        # Fetch data as a pandas DataFrame
        df = fetch_data(dataset["pmlb_name"], return_X_y=False)

        # Convention: last column is the target
        X = df.iloc[:, :-1].values  # type: ignore[assignment]
        y = df.iloc[:, -1].values.astype(float).ravel()  # type: ignore[assignment]

        # Train / test split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        print(f". Training rows {X_tr.shape[0]}. D: {X_tr.shape[1]}")

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
