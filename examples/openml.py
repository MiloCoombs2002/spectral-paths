"""Example module showing model usage with OpenML datasets."""
import time
from typing import TypedDict

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from spectral_paths.model import SpectralPathRegressor

verbose = False
extra_verbose = False

class DatasetSpec(TypedDict):
    """DatasetSpec schema."""
    name: str
    openml_name: str
    version: int

datasets: list[DatasetSpec] = [
    {"name": "Concrete Slump", "openml_name": "slump", "version": 2},
    {"name": "Yacht Hydrodynamics", "openml_name": "yacht_hydrodynamics", "version": 1},
    {
        "name": "Cancer Drug Response",
        "openml_name": "Cancer_Drug_Response",
        "version": 1
    },
    {"name": "Aquatic Toxicity", "openml_name": "qsar_aquatic_toxicity", "version": 1},
#   {"name": "Titanic Survival", "openml_name": "Concrete Slump", "version": 1},
    {"name": "Izmir Weather", "openml_name": "weather_izmir", "version": 1},
    {"name": "Ankara Weather", "openml_name": "weather_ankara", "version": 1},
]

if __name__ == "__main__":
    for dataset in datasets:

        print(f"Dataset: {dataset["name"]}")
        X, y = fetch_openml(
            name=dataset["openml_name"],
            version=dataset["version"],
            as_frame=False,
            parser="auto",
            return_X_y=True,
        )

        # Ensure target is float and 1D
        if dataset["openml_name"] == "slump":
            y = np.asarray(y, dtype=float)[:, 0].ravel()
        else:
            y = np.asarray(y, dtype=float).ravel()

        # Train / test split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        D = X.shape[1]

        mdl = SpectralPathRegressor(
            total_cols=30*D if dataset["openml_name"] !="Cancer_Drug_Response" else 3*D,
            block_size=1 * D,
            lambda_grid=list(np.logspace(-5, -1, 25)),
            l_max=None,
            scaler_type="robust_tanh",
            bound_percentiles=(5, 95),
            batch_rows=2048,
            verbose=verbose,
            random_state=42,
            val_size=0.25,
            final_lambda_refit=True,
            normalize_columns=True,
            normalize_intercept=False,
            k_values=(1,2,3,4),
            # New improved parameters
            early_stopping_patience=5,
            early_stopping_tol= 1e-5,
            adaptive_block_size=True,
            min_block_size=1,
            use_importance_ordering=True,
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
