"""Tests for the spectral-path binary classifier."""

from __future__ import annotations

import numpy as np

from examples.benchmark_classification import (
    benchmark_one_dataset,
    build_spectral_path_regressor,
    evaluate_regressor_baseline,
)
from spectral_paths.model import SpectralPathClassifier


def _make_binary_data(
    *, seed: int = 0, n_samples: int = 180, n_features: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    logits = 1.6 * X[:, 0] - 1.1 * X[:, 1]
    if n_features >= 4:
        logits = logits + 0.6 * X[:, 2] * X[:, 3]
    if n_features >= 5:
        logits = logits - 0.5 * X[:, 4]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs, size=n_samples).astype(int)
    return X, y


def _make_classifier(
    *,
    max_paths: int = 8,
    block_size: int = 2,
    lambda_grid: tuple[float, ...] = (0.001, 0.01, 0.1),
    k_values: tuple[int, ...] = (1, 2),
    verbose: bool = False,
    use_float32: bool = False,
    lambda_parallel_workers: int = 1,
    final_lambda_refit: bool = True,
    greedy_subsample: float | int | None = None,
) -> SpectralPathClassifier:
    return SpectralPathClassifier(
        max_paths=max_paths,
        block_size=block_size,
        lambda_grid=lambda_grid,
        k_values=k_values,
        verbose=verbose,
        use_float32=use_float32,
        lambda_parallel_workers=lambda_parallel_workers,
        final_lambda_refit=final_lambda_refit,
        greedy_subsample=greedy_subsample,
        early_stopping_patience=2,
        early_stopping_tol=1e-5,
        irls_max_iter=60,
        irls_tol=1e-7,
    )


def test_classifier_fit_predict_and_predict_proba() -> None:
    """Classifier should fit and emit valid probabilities and labels."""
    X, y = _make_binary_data()
    model = _make_classifier(greedy_subsample=80)

    model.fit(X, y)

    probas = model.predict_proba(X[:8])
    preds = model.predict(X[:8])
    assert probas.shape == (8, 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-8)
    assert set(np.unique(preds)).issubset({0, 1})
    assert model.score(X, y) > 0.65


def test_classifier_rejects_non_binary_labels() -> None:
    """Classifier must require labels encoded exactly as 0/1."""
    X, y = _make_binary_data()
    y_bad = y.astype(float)
    y_bad[0] = 2.0
    model = _make_classifier()

    try:
        model.fit(X, y_bad)
    except ValueError as exc:
        assert "0/1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-binary labels")


def test_classifier_float32_predictions_remain_close() -> None:
    """Float32 mode should stay close to the float64 probability baseline."""
    X, y = _make_binary_data(seed=2, n_samples=220, n_features=7)
    model64 = _make_classifier(max_paths=10, block_size=3)
    model32 = _make_classifier(max_paths=10, block_size=3, use_float32=True)

    model64.fit(X, y)
    model32.fit(X, y)

    prob64 = model64.predict_proba(X)[:, 1]
    prob32 = model32.predict_proba(X)[:, 1]
    np.testing.assert_allclose(prob32, prob64, rtol=5e-4, atol=5e-4)


def test_classifier_parallel_lambda_matches_sequential() -> None:
    """Parallel lambda scoring should preserve classifier probabilities."""
    X, y = _make_binary_data(seed=5, n_samples=150, n_features=5)
    sequential = _make_classifier(lambda_parallel_workers=1)
    parallel = _make_classifier(lambda_parallel_workers=2)

    sequential.fit(X, y)
    parallel.fit(X, y)

    assert sequential.selected_paths_ == parallel.selected_paths_
    assert sequential.lambda_ == parallel.lambda_
    np.testing.assert_allclose(
        parallel.predict_proba(X)[:, 1],
        sequential.predict_proba(X)[:, 1],
        rtol=1e-8,
        atol=1e-8,
    )


def test_classifier_regularization_shrinks_coefficients() -> None:
    """Stronger L2 regularization should shrink the learned coefficients."""
    X, y = _make_binary_data(seed=8, n_samples=160, n_features=5)
    weak = _make_classifier(lambda_grid=(1e-4,), final_lambda_refit=False)
    strong = _make_classifier(lambda_grid=(1.0,), final_lambda_refit=False)

    weak.fit(X, y)
    strong.fit(X, y)

    assert weak.coef_ is not None
    assert strong.coef_ is not None
    assert np.linalg.norm(strong.coef_[1:]) < np.linalg.norm(weak.coef_[1:])


def test_classifier_history_records_log_loss() -> None:
    """Greedy history should record finite validation log-loss values."""
    X, y = _make_binary_data(seed=13, n_samples=140, n_features=5)
    model = _make_classifier(lambda_grid=(0.01,), k_values=(1,), greedy_subsample=50)

    model.fit(X, y)

    assert model.fit_report_ is not None
    assert model.fit_report_.history
    last_history_value = model.fit_report_.history[-1][-1]
    assert np.isfinite(last_history_value)
    assert last_history_value >= 0.0


def test_regressor_binary_baseline_metrics_are_finite() -> None:
    """The binary regressor baseline should produce finite comparison metrics."""
    X, y = _make_binary_data(seed=21, n_samples=120, n_features=4)
    split = 90
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    regressor = build_spectral_path_regressor(
        feature_count=X.shape[1],
        config=type(
            "Cfg",
            (),
            {
                "max_paths": 16,
                "block_size_factor": 1,
                "lambda_grid": (0.01, 0.1),
                "k_values": (1, 2),
                "early_stopping_patience": 2,
                "early_stopping_tol": 1e-5,
                "greedy_subsample": 60,
            },
        )(),
        verbose=False,
    )

    metrics = evaluate_regressor_baseline(
        regressor, X_tr, y_tr.astype(float), X_te, y_te.astype(float)
    )
    assert np.isfinite(metrics["log_loss"])
    assert np.isfinite(metrics["accuracy"])
    assert np.isfinite(metrics["r2"])


def test_classification_benchmark_smoke() -> None:
    """The local binary benchmark path should run end-to-end."""
    label, results = benchmark_one_dataset("sklearn:breast-cancer", mode="fast")
    assert label
    expected_models = {
        "SpectralPathClassifier",
        "SpectralPathRegressor",
        "LogisticRegression",
        "RandomForestClassifier",
        "HistGradientBoostingClassifier",
    }
    assert expected_models.issubset(results.keys())
    for metrics in results.values():
        assert np.isfinite(metrics["log_loss"])
        assert np.isfinite(metrics["accuracy"])
        assert np.isfinite(metrics["r2"])
