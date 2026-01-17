"""Example module showing model usage with an OpenML dataset."""
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from spectral_paths.model import SpectralPathRegressor

X, y = fetch_openml(
    name="weather_ankara",
    version=1,
    as_frame=False,
    parser="auto",
    return_X_y=True,
)

# Ensure target is float and 1D
y = np.asarray(y, dtype=float).ravel()

# Train / test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42
)

D = X.shape[1]

if __name__ == "__main__":
    mdl = SpectralPathRegressor(
        total_cols=40*D, # 4 * D,
        block_size=1 * D,
        lambda_grid=list(np.logspace(-5, -1, 25)),
        l_max=None,
        scaler_type="standard_percentile_minmax",
        bound_percentiles=(2.0, 98.0),
        batch_rows=2048,
        verbose=True,
        random_state=42,
        val_size=0.25,
        final_lambda_refit=True,
        normalize_columns=True,
        normalize_intercept=False,
        k_values=(1,2,3),
        # New improved parameters
        early_stopping_patience=5,
        early_stopping_tol= 1e-5,
        adaptive_block_size=True,
        min_block_size=1,
        use_importance_ordering=True,
    )
    print("Training model...")
    t0 = time.perf_counter()
    mdl.fit(X_tr, y_tr)
    t1 = time.perf_counter()

    print(f"\nTotal training time: {t1-t0:.3f}s")

    yhat = mdl.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, yhat)))
    r2 = float(r2_score(y_te, yhat))

    print("\n=== Test ===")
    print(f"R²:   {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    if mdl.fit_report_ is not None:
        print("\n=== Timing ===")
        print(f"Greedy selection: {mdl.fit_report_.greedy_time_sec:.3f}s")
        print(f"Final solve:      {mdl.fit_report_.final_solve_time_sec:.3f}s")
        print(f"Selected cols:    {mdl.fit_report_.selected_count} (+ intercept)")
        print(f"λ*:               {mdl.fit_report_.lambda_star}")
        print(f"Stopped early:    {mdl.fit_report_.stopped_early}")
