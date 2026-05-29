# Complexity-Aware Ridge Benchmark Results

- mode: `overgrow`
- dataset selection: `openml:yacht-hydrodynamics`

| Dataset | Case | Regularization | Cold (s) | Warm (s) | Paths | Lambda | Test R2 | Delta vs Baseline | Mean Total Order | Max Total Order | Mean Sparsity | Max Sparsity | Mean Harmonic Order | Max Harmonic Order |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenML Yacht Hydrodynamics | baseline-uniform | `uniform` | 0.194 | 0.131 | 144 | 0.01000 | 0.9894 | +0.0000 | 6.40 | 24 | 2.38 | 4 | 4.05 | 24 |
| OpenML Yacht Hydrodynamics | total_order-exp-3 | `total_order:exponential:3.00` | 0.816 | 0.789 | 494 | 0.02154 | 0.9971 | +0.0077 | 24.09 | 78 | 1.92 | 4 | 21.72 | 78 |
| OpenML Yacht Hydrodynamics | total_order-exp-4 | `total_order:exponential:4.00` | 0.572 | 0.575 | 487 | 0.01468 | 0.9967 | +0.0072 | 27.61 | 81 | 1.78 | 4 | 25.72 | 81 |
| OpenML Yacht Hydrodynamics | total_order-exp-6 | `total_order:exponential:6.00` | 0.921 | 0.954 | 495 | 0.02154 | 0.9963 | +0.0069 | 26.13 | 83 | 1.65 | 4 | 24.20 | 83 |
| OpenML Yacht Hydrodynamics | total_order-exp-8 | `total_order:exponential:8.00` | 0.919 | 0.888 | 520 | 0.01468 | 0.9959 | +0.0065 | 30.01 | 84 | 1.60 | 4 | 28.39 | 84 |
| OpenML Yacht Hydrodynamics | total_order-exp-10 | `total_order:exponential:10.00` | 1.854 | 1.806 | 644 | 0.00003 | 0.9983 | +0.0089 | 34.83 | 108 | 1.61 | 4 | 32.87 | 108 |
| OpenML Yacht Hydrodynamics | sparsity-exp-4 | `sparsity:exponential:4.00` | 2.600 | 2.576 | 640 | 0.01468 | 0.9962 | +0.0068 | 17.65 | 107 | 2.39 | 4 | 13.63 | 107 |
| OpenML Yacht Hydrodynamics | sparsity-exp-6 | `sparsity:exponential:6.00` | 0.255 | 0.252 | 260 | 0.00032 | 0.9958 | +0.0063 | 9.38 | 44 | 2.48 | 4 | 6.50 | 44 |
| OpenML Yacht Hydrodynamics | sparsity-exp-8 | `sparsity:exponential:8.00` | 0.873 | 0.859 | 465 | 0.00002 | 0.9985 | +0.0091 | 13.29 | 78 | 2.45 | 4 | 9.61 | 78 |
| OpenML Yacht Hydrodynamics | sparsity-exp-10 | `sparsity:exponential:10.00` | 0.285 | 0.286 | 301 | 0.00001 | 0.9978 | +0.0083 | 8.53 | 50 | 2.66 | 4 | 5.02 | 50 |
