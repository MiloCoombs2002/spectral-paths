# Complexity-Aware Ridge Benchmark Results

- mode: `overgrow`
- dataset selection: `uci:energy-efficiency`

| Dataset | Case | Regularization | Cold (s) | Warm (s) | Paths | Lambda | Test R2 | Delta vs Baseline | Mean Total Order | Max Total Order | Mean Sparsity | Max Sparsity | Mean Harmonic Order | Max Harmonic Order |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UCI Energy Efficiency | baseline-uniform | `uniform` | 0.962 | 0.887 | 295 | 0.00068 | 0.9977 | +0.0000 | 8.91 | 37 | 2.29 | 4 | 6.74 | 37 |
| UCI Energy Efficiency | total_order-exp-3 | `total_order:exponential:3.00` | 0.266 | 0.237 | 297 | 0.00046 | 0.9977 | +0.0001 | 11.44 | 37 | 2.03 | 4 | 9.71 | 37 |
| UCI Energy Efficiency | total_order-exp-4 | `total_order:exponential:4.00` | 0.499 | 0.479 | 349 | 0.00046 | 0.9978 | +0.0001 | 11.30 | 43 | 2.07 | 4 | 9.35 | 43 |
| UCI Energy Efficiency | total_order-exp-6 | `total_order:exponential:6.00` | 0.254 | 0.228 | 299 | 0.00068 | 0.9976 | -0.0000 | 9.29 | 36 | 1.96 | 4 | 7.45 | 36 |
| UCI Energy Efficiency | total_order-exp-8 | `total_order:exponential:8.00` | 0.486 | 0.465 | 347 | 0.00068 | 0.9974 | -0.0002 | 11.91 | 42 | 1.84 | 4 | 10.21 | 42 |
| UCI Energy Efficiency | total_order-exp-10 | `total_order:exponential:10.00` | 0.514 | 0.484 | 376 | 0.00068 | 0.9974 | -0.0002 | 13.88 | 47 | 1.62 | 4 | 12.32 | 47 |
| UCI Energy Efficiency | sparsity-exp-4 | `sparsity:exponential:4.00` | 0.352 | 0.332 | 295 | 0.00022 | 0.9975 | -0.0001 | 8.88 | 37 | 2.23 | 4 | 6.74 | 37 |
| UCI Energy Efficiency | sparsity-exp-6 | `sparsity:exponential:6.00` | 0.404 | 0.377 | 352 | 0.00022 | 0.9976 | -0.0000 | 8.98 | 44 | 2.29 | 4 | 6.60 | 44 |
| UCI Energy Efficiency | sparsity-exp-8 | `sparsity:exponential:8.00` | 0.365 | 0.340 | 355 | 0.00022 | 0.9978 | +0.0001 | 9.25 | 44 | 2.07 | 4 | 6.89 | 44 |
| UCI Energy Efficiency | sparsity-exp-10 | `sparsity:exponential:10.00` | 0.338 | 0.311 | 373 | 0.00032 | 0.9978 | +0.0001 | 8.41 | 47 | 2.03 | 4 | 5.89 | 47 |
