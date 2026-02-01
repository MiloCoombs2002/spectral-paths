# spectral-paths

Spectral-path regression in Python. This project provides a `SpectralPathRegressor` model that builds spectral path features, selects a sparse dictionary with a greedy procedure, and fits ridge-style coefficients with optional early stopping and final refit. It uses directioanl harmonics to approximate teh target as a smooth function and is inspired by Chebyshev polynomials.

## Features
- Spectral-path feature construction with configurable sparsity (`k_values`).
- Greedy selection with validation-based early stopping.
- Robust input scaling options via `AngularTransformer`.
- Simple, scikit-learn–style API (`fit`, `predict`).

## Installation
This repo uses Poetry.

```bash
poetry install
```

Alternatively, install dependencies directly (Python 3.12+):

```bash
pip install -r <(poetry export -f requirements.txt --without-hashes)
```

## Quick start
```python
import numpy as np
from spectral_paths.model import SpectralPathRegressor

# Fake data
rng = np.random.default_rng(0)
X = rng.normal(size=(500, 10))
y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(scale=0.1, size=500)

D = X.shape[1]
model = SpectralPathRegressor(
    total_cols=30 * D,
    block_size=1 * D,
    lambda_grid=np.logspace(-5, -1, 25),
    scaler_type="robust_tanh",
    bound_percentiles=(5, 95),
    val_size=0.25,
    k_values=(1, 2, 3, 4),
    early_stopping_patience=5,
    early_stopping_tol=1e-5,
    adaptive_block_size=True,
    min_block_size=1,
    use_importance_ordering=True,
)

model.fit(X, y)
preds = model.predict(X)
```

## Examples
- OpenML datasets: `examples/openml.py`
- PMLB datasets: `examples/pmlb.py`

Run an example with:

```bash
python examples/openml.py
```

Note: OpenML and PMLB examples download datasets over the network.

## Project layout
- `src/spectral_paths/model.py`: `SpectralPathRegressor` implementation.
- `src/spectral_paths/utils/`: feature construction, preprocessing, and helper utilities.
- `examples/`: runnable scripts for benchmarks and demos.

## Development
- Lint: `ruff check .`
- Type check: `mypy src`
- Tests: `pytest`

## License
See `LICENSE`.
