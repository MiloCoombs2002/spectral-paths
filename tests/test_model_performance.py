"""Tests for performance-related model behavior."""

from __future__ import annotations

import numpy as np

from spectral_paths.model import SpectralPathRegressor


def _make_regression_data(
    *, seed: int = 0, n_samples: int = 120, n_features: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = (
        1.4 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.3 * X[:, 0] * X[:, 2]
        + rng.normal(scale=0.05, size=n_samples)
    )
    return X, y


def _make_model(**overrides: object) -> SpectralPathRegressor:
    kwargs: dict[str, object] = {
        "max_paths": 8,
        "block_size": 2,
        "lambda_grid": (0.001, 0.01, 0.1),
        "k_values": (1, 2),
        "val_size": 0.2,
        "random_state": 7,
        "verbose": False,
        "final_lambda_refit": True,
        "early_stopping_patience": 2,
        "early_stopping_tol": 1e-5,
        "use_importance_ordering": True,
        "adaptive_block_size": True,
    }
    kwargs.update(overrides)
    return SpectralPathRegressor(**kwargs)


def test_fit_report_exposes_phase_timings() -> None:
    X, y = _make_regression_data()
    model = _make_model(greedy_subsample=60)

    model.fit(X, y)

    assert model.fit_report_ is not None
    timings = model.fit_report_.phase_timings
    assert timings.preprocessing_sec >= 0.0
    assert timings.greedy_accumulation_sec >= 0.0
    assert timings.greedy_scoring_sec >= 0.0
    assert timings.lambda_sweep_sec >= 0.0
    assert timings.final_normal_eqn_sec >= 0.0
    assert timings.final_solve_sec >= 0.0
    assert timings.total_fit_sec >= timings.greedy_accumulation_sec
    assert model.fit_report_.blas_threads.policy == "auto"


def test_lambda_parallel_matches_sequential(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

    X, y = _make_regression_data(seed=1)
    sequential = _make_model(lambda_parallel_workers=1)
    parallel = _make_model(lambda_parallel_workers=2)

    sequential.fit(X, y)
    parallel.fit(X, y)

    assert sequential.selected_paths_ == parallel.selected_paths_
    assert sequential.lambda_ == parallel.lambda_
    np.testing.assert_allclose(
        sequential.predict(X),
        parallel.predict(X),
        rtol=1e-8,
        atol=1e-8,
    )


def test_float32_predictions_remain_close() -> None:
    X, y = _make_regression_data(seed=2, n_samples=180, n_features=6)
    model64 = _make_model(max_paths=10, block_size=3, lambda_parallel_workers=1)
    model32 = _make_model(
        max_paths=10,
        block_size=3,
        lambda_parallel_workers=1,
        use_float32=True,
    )

    model64.fit(X, y)
    model32.fit(X, y)

    preds64 = model64.predict(X)
    preds32 = model32.predict(X)
    np.testing.assert_allclose(preds32, preds64, rtol=5e-4, atol=5e-4)
    assert abs(model64.score(X, y) - model32.score(X, y)) < 0.01


def test_single_lambda_single_k_and_subsample_fit() -> None:
    X, y = _make_regression_data(seed=3, n_samples=90, n_features=4)
    model = _make_model(
        lambda_grid=(0.01,),
        k_values=(1,),
        greedy_subsample=24,
        max_paths=6,
        block_size=1,
    )

    model.fit(X, y)

    assert model.lambda_ == 0.01
    assert model.selected_paths_ is not None
    assert len(model.selected_paths_) <= 6
    assert model.predict(X[:5]).shape == (5,)


def test_blas_policy_auto_resolves_none_for_small_width() -> None:
    model = _make_model(max_paths=8, lambda_grid=(0.01, 0.1), k_values=(1, 2))
    model.n_features_in_ = 5

    assert model._resolve_blas_threads(n_train_rows=120) is None


def test_blas_policy_auto_resolves_single_for_high_width() -> None:
    model = _make_model(max_paths=512, lambda_grid=tuple(np.logspace(-5, -1, 25)))
    model.n_features_in_ = 30

    assert model._resolve_blas_threads(n_train_rows=500) == 1


def test_invalid_manual_blas_threads_raises() -> None:
    try:
        _make_model(blas_thread_policy="manual", blas_threads=0)
    except ValueError as exc:
        assert "blas_threads" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid manual blas_threads")


def test_none_and_single_blas_policies_match_predictions() -> None:
    X, y = _make_regression_data(seed=5, n_samples=160, n_features=6)
    model_none = _make_model(blas_thread_policy="none")
    model_single = _make_model(blas_thread_policy="single")

    model_none.fit(X, y)
    model_single.fit(X, y)

    assert model_none.fit_report_ is not None
    assert model_single.fit_report_ is not None
    assert model_none.fit_report_.blas_threads.resolved_threads is None
    assert model_single.fit_report_.blas_threads.resolved_threads == 1
    np.testing.assert_allclose(
        model_none.predict(X),
        model_single.predict(X),
        rtol=1e-8,
        atol=1e-8,
    )
