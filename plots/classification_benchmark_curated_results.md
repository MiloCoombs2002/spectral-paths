# Curated Binary Classification Benchmark Results

Run date: 2026-05-29

Setup:
- 6 dataset “smooth numeric core” benchmark
- `examples/benchmark_classification.py`
- dataset selection: `curated`
- `mode="default"`
- 80/20 stratified train/test split with `random_state=42`
- `SpectralPathRegressor` baseline trained on `0/1` labels; its log-loss uses clipped predictions in `[eps, 1-eps]`

Curated suite:
- Breast Cancer Wisconsin
- OpenML Banknote Authentication
- OpenML Diabetes
- OpenML Phoneme
- OpenML QSAR Biodeg
- OpenML WDBC

Note:
- `Breast Cancer Wisconsin` and `WDBC` are related benchmark families, so this curated suite is cleaner than the broad table but still not perfectly non-redundant.
- `Spambase` and `ILPD` were intentionally dropped from this view because they are less aligned with the “smooth numeric geometry” story for spectral paths.

Lower log loss is better. Higher accuracy and `R^2` are better.

## Log Loss

| Dataset | SpectralPathClassifier | SpectralPathRegressor | LogisticRegression | RandomForestClassifier | HistGradientBoostingClassifier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Breast Cancer Wisconsin | 0.1480 | 0.1147 | 0.0915 | 0.1094 | 0.1035 |
| OpenML Banknote Authentication | 0.0020 | 0.0158 | 0.0324 | 0.0232 | 0.0056 |
| OpenML Diabetes | 0.5017 | 0.6579 | 0.4913 | 0.4990 | 0.6252 |
| OpenML Phoneme | 0.3727 | 0.5081 | 0.4969 | 0.3406 | 0.2877 |
| OpenML QSAR Biodeg | 0.3310 | 0.4413 | 0.3714 | 0.3040 | 0.3240 |
| OpenML WDBC | 0.1605 | 0.1150 | 0.1191 | 0.1135 | 0.1142 |

## Accuracy

| Dataset | SpectralPathClassifier | SpectralPathRegressor | LogisticRegression | RandomForestClassifier | HistGradientBoostingClassifier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Breast Cancer Wisconsin | 0.9737 | 0.9649 | 0.9649 | 0.9474 | 0.9737 |
| OpenML Banknote Authentication | 1.0000 | 1.0000 | 0.9855 | 0.9964 | 0.9964 |
| OpenML Diabetes | 0.7532 | 0.7532 | 0.7273 | 0.7597 | 0.7532 |
| OpenML Phoneme | 0.8437 | 0.8501 | 0.7364 | 0.8982 | 0.8696 |
| OpenML QSAR Biodeg | 0.8720 | 0.8483 | 0.8578 | 0.8626 | 0.8768 |
| OpenML WDBC | 0.9561 | 0.9649 | 0.9386 | 0.9737 | 0.9649 |

## R²

| Dataset | SpectralPathClassifier | SpectralPathRegressor | LogisticRegression | RandomForestClassifier | HistGradientBoostingClassifier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Breast Cancer Wisconsin | 0.8728 | 0.8453 | 0.8834 | 0.8621 | 0.8818 |
| OpenML Banknote Authentication | 0.9986 | 0.9934 | 0.9606 | 0.9851 | 0.9922 |
| OpenML Diabetes | 0.2663 | 0.2527 | 0.2778 | 0.2718 | 0.2049 |
| OpenML Phoneme | 0.4683 | 0.4793 | 0.1965 | 0.6329 | 0.5792 |
| OpenML QSAR Biodeg | 0.5462 | 0.5204 | 0.5184 | 0.5930 | 0.5909 |
| OpenML WDBC | 0.8741 | 0.8728 | 0.8306 | 0.8646 | 0.8897 |

## Quick Takeaways

- `SpectralPathClassifier` clearly improves over the spectral regressor baseline on probability quality for `Banknote Authentication`, `Diabetes`, `Phoneme`, and `QSAR Biodeg`.
- The strongest classifier result remains `Banknote Authentication`, where the spectral classifier is best on all three reported metrics.
- `Phoneme` is still hard for the spectral family relative to trees, but the classifier is much better calibrated than the spectral regressor there.
- `Breast Cancer Wisconsin` and `WDBC` remain strong but not dominant results; the spectral classifier is competitive, while logistic regression and tree baselines still edge it out on log loss.
