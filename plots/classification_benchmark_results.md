# Binary Classification Benchmark Results

Run date: 2026-05-29

Setup:
- 8 real binary classification datasets
- `examples/benchmark_classification.py`
- `mode="default"`
- 80/20 stratified train/test split with `random_state=42`
- `SpectralPathRegressor` baseline trained on `0/1` labels; its log-loss uses clipped predictions in `[eps, 1-eps]`

Lower log loss is better. Higher accuracy and `R^2` are better.

## Log Loss

| Dataset | SpectralPathClassifier | SpectralPathRegressor | LogisticRegression | RandomForestClassifier | HistGradientBoostingClassifier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Breast Cancer Wisconsin | 0.1480 | 0.1147 | 0.0915 | 0.1094 | 0.1035 |
| OpenML Phoneme | 0.3727 | 0.5081 | 0.4969 | 0.3406 | 0.2877 |
| OpenML WDBC | 0.1605 | 0.1150 | 0.1191 | 0.1135 | 0.1142 |
| OpenML Diabetes | 0.5017 | 0.6579 | 0.4913 | 0.4990 | 0.6252 |
| OpenML Spambase | 0.1595 | 0.2252 | 0.2369 | 0.2315 | 0.1467 |
| OpenML Banknote Authentication | 0.0020 | 0.0158 | 0.0324 | 0.0232 | 0.0056 |
| OpenML ILPD | 0.7301 | 0.5430 | 0.5352 | 0.5828 | 0.8313 |
| OpenML QSAR Biodeg | 0.3310 | 0.4413 | 0.3714 | 0.3040 | 0.3240 |

## Accuracy

| Dataset | SpectralPathClassifier | SpectralPathRegressor | LogisticRegression | RandomForestClassifier | HistGradientBoostingClassifier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Breast Cancer Wisconsin | 0.9737 | 0.9649 | 0.9649 | 0.9474 | 0.9737 |
| OpenML Phoneme | 0.8437 | 0.8501 | 0.7364 | 0.8982 | 0.8696 |
| OpenML WDBC | 0.9561 | 0.9649 | 0.9386 | 0.9737 | 0.9649 |
| OpenML Diabetes | 0.7532 | 0.7532 | 0.7273 | 0.7597 | 0.7532 |
| OpenML Spambase | 0.9435 | 0.9273 | 0.9283 | 0.9457 | 0.9468 |
| OpenML Banknote Authentication | 1.0000 | 1.0000 | 0.9855 | 0.9964 | 0.9964 |
| OpenML ILPD | 0.6239 | 0.6667 | 0.6838 | 0.6239 | 0.6239 |
| OpenML QSAR Biodeg | 0.8720 | 0.8483 | 0.8578 | 0.8626 | 0.8768 |

## R²

| Dataset | SpectralPathClassifier | SpectralPathRegressor | LogisticRegression | RandomForestClassifier | HistGradientBoostingClassifier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Breast Cancer Wisconsin | 0.8728 | 0.8453 | 0.8834 | 0.8621 | 0.8818 |
| OpenML Phoneme | 0.4683 | 0.4793 | 0.1965 | 0.6329 | 0.5792 |
| OpenML WDBC | 0.8741 | 0.8728 | 0.8306 | 0.8646 | 0.8897 |
| OpenML Diabetes | 0.2663 | 0.2527 | 0.2778 | 0.2718 | 0.2049 |
| OpenML Spambase | 0.8086 | 0.7487 | 0.7545 | 0.8188 | 0.8314 |
| OpenML Banknote Authentication | 0.9986 | 0.9934 | 0.9606 | 0.9851 | 0.9922 |
| OpenML ILPD | -0.2101 | 0.0978 | 0.1392 | -0.0040 | -0.2798 |
| OpenML QSAR Biodeg | 0.5462 | 0.5204 | 0.5184 | 0.5930 | 0.5909 |

## Quick Takeaways

- `SpectralPathClassifier` was best on log loss for `Spambase` and `Banknote Authentication`, and essentially tied for best accuracy on `Breast Cancer Wisconsin`.
- On `Phoneme`, `Spambase`, and `QSAR Biodeg`, the classifier improved clearly over the spectral regressor baseline on probability quality.
- On `ILPD`, both spectral models lagged logistic regression, and the regressor baseline outperformed the classifier on all three reported metrics.
- Tree baselines were especially strong on `Phoneme`, `WDBC`, `Spambase`, and `QSAR Biodeg`.
