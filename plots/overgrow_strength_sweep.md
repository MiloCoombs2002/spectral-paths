# Complexity-Aware Ridge Benchmark Results

- mode: `overgrow`
- dataset selection: `all`

| Dataset | Case | Regularization | Cold (s) | Warm (s) | Paths | Lambda | Test R2 | Delta vs Baseline | Mean Total Order | Max Total Order | Mean Sparsity | Max Sparsity | Mean Harmonic Order | Max Harmonic Order |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenML Concrete Slump | baseline-uniform | `uniform` | 0.664 | 0.574 | 290 | 0.04642 | -0.7066 | +0.0000 | 6.62 | 42 | 3.08 | 4 | 3.30 | 42 |
| OpenML Concrete Slump | sparsity-exp-1 | `sparsity:exponential:1.00` | 0.220 | 0.217 | 264 | 0.06813 | 0.0352 | +0.7418 | 6.97 | 38 | 2.96 | 4 | 3.82 | 38 |
| OpenML Concrete Slump | sparsity-exp-2 | `sparsity:exponential:2.00` | 0.410 | 0.409 | 351 | 0.04642 | -0.1613 | +0.5453 | 7.20 | 48 | 3.02 | 4 | 3.69 | 48 |
| OpenML Concrete Slump | sparsity-exp-4 | `sparsity:exponential:4.00` | 0.061 | 0.060 | 114 | 0.06813 | 0.0297 | +0.7363 | 3.68 | 17 | 2.75 | 3 | 1.20 | 17 |
| OpenML Concrete Slump | sparsity-exp-8 | `sparsity:exponential:8.00` | 3.641 | 3.431 | 768 | 0.06813 | 0.2309 | +0.9376 | 5.93 | 10 | 3.58 | 4 | 1.07 | 5 |
| OpenML Concrete Slump | total_order-exp-1 | `total_order:exponential:1.00` | 0.233 | 0.232 | 295 | 0.03162 | -0.3753 | +0.3313 | 7.94 | 42 | 2.89 | 4 | 4.80 | 42 |
| OpenML Concrete Slump | total_order-exp-2 | `total_order:exponential:2.00` | 0.079 | 0.079 | 157 | 0.10000 | -0.3167 | +0.3899 | 6.53 | 23 | 2.82 | 4 | 3.97 | 23 |
| OpenML Concrete Slump | total_order-exp-4 | `total_order:exponential:4.00` | 0.042 | 0.043 | 74 | 0.10000 | 0.3218 | +1.0285 | 2.96 | 4 | 2.26 | 3 | 1.08 | 2 |
| OpenML Concrete Slump | total_order-exp-8 | `total_order:exponential:8.00` | 0.076 | 0.075 | 163 | 0.06813 | 0.3398 | +1.0465 | 3.82 | 5 | 2.64 | 4 | 1.06 | 2 |
| OpenML Yacht Hydrodynamics | baseline-uniform | `uniform` | 0.131 | 0.131 | 144 | 0.01000 | 0.9894 | +0.0000 | 6.40 | 24 | 2.38 | 4 | 4.05 | 24 |
| OpenML Yacht Hydrodynamics | sparsity-exp-1 | `sparsity:exponential:1.00` | 0.890 | 0.903 | 421 | 0.00001 | 0.9976 | +0.0082 | 15.02 | 70 | 2.36 | 4 | 11.83 | 70 |
| OpenML Yacht Hydrodynamics | sparsity-exp-2 | `sparsity:exponential:2.00` | 0.404 | 0.399 | 299 | 0.00001 | 0.9973 | +0.0079 | 11.21 | 50 | 2.27 | 4 | 8.30 | 50 |
| OpenML Yacht Hydrodynamics | sparsity-exp-4 | `sparsity:exponential:4.00` | 2.549 | 2.589 | 640 | 0.01468 | 0.9962 | +0.0068 | 17.65 | 107 | 2.39 | 4 | 13.63 | 107 |
| OpenML Yacht Hydrodynamics | sparsity-exp-8 | `sparsity:exponential:8.00` | 0.834 | 0.848 | 465 | 0.00002 | 0.9985 | +0.0091 | 13.29 | 78 | 2.45 | 4 | 9.61 | 78 |
| OpenML Yacht Hydrodynamics | total_order-exp-1 | `total_order:exponential:1.00` | 0.096 | 0.096 | 165 | 0.00681 | 0.9879 | -0.0015 | 8.06 | 24 | 2.10 | 4 | 6.03 | 24 |
| OpenML Yacht Hydrodynamics | total_order-exp-2 | `total_order:exponential:2.00` | 1.194 | 1.181 | 603 | 0.01468 | 0.9979 | +0.0085 | 26.59 | 100 | 1.99 | 4 | 23.84 | 100 |
| OpenML Yacht Hydrodynamics | total_order-exp-4 | `total_order:exponential:4.00` | 0.569 | 0.569 | 487 | 0.01468 | 0.9967 | +0.0072 | 27.61 | 81 | 1.78 | 4 | 25.72 | 81 |
| OpenML Yacht Hydrodynamics | total_order-exp-8 | `total_order:exponential:8.00` | 0.926 | 0.885 | 520 | 0.01468 | 0.9959 | +0.0065 | 30.01 | 84 | 1.60 | 4 | 28.39 | 84 |
