# Complexity-Aware Ridge Benchmark Results

- mode: `overgrow`
- dataset selection: `openml:concrete-slump`

| Dataset | Case | Regularization | Cold (s) | Warm (s) | Paths | Lambda | Test R2 | Delta vs Baseline | Mean Total Order | Max Total Order | Mean Sparsity | Max Sparsity | Mean Harmonic Order | Max Harmonic Order |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenML Concrete Slump | baseline-uniform | `uniform` | 0.680 | 0.578 | 290 | 0.04642 | -0.7066 | +0.0000 | 6.62 | 42 | 3.08 | 4 | 3.30 | 42 |
| OpenML Concrete Slump | total_order-exp-3 | `total_order:exponential:3.00` | 0.040 | 0.038 | 61 | 0.10000 | 0.4169 | +1.1235 | 3.08 | 4 | 2.54 | 3 | 1.00 | 1 |
| OpenML Concrete Slump | total_order-exp-4 | `total_order:exponential:4.00` | 0.043 | 0.042 | 74 | 0.10000 | 0.3218 | +1.0285 | 2.96 | 4 | 2.26 | 3 | 1.08 | 2 |
| OpenML Concrete Slump | total_order-exp-6 | `total_order:exponential:6.00` | 0.080 | 0.080 | 160 | 0.10000 | 0.3980 | +1.1046 | 3.82 | 5 | 2.73 | 4 | 1.04 | 2 |
| OpenML Concrete Slump | total_order-exp-8 | `total_order:exponential:8.00` | 0.077 | 0.077 | 163 | 0.06813 | 0.3398 | +1.0465 | 3.82 | 5 | 2.64 | 4 | 1.06 | 2 |
| OpenML Concrete Slump | total_order-exp-10 | `total_order:exponential:10.00` | 0.128 | 0.127 | 255 | 0.04642 | 0.3273 | +1.0339 | 4.35 | 5 | 2.75 | 4 | 1.03 | 2 |
| OpenML Concrete Slump | sparsity-exp-4 | `sparsity:exponential:4.00` | 0.061 | 0.061 | 114 | 0.06813 | 0.0297 | +0.7363 | 3.68 | 17 | 2.75 | 3 | 1.20 | 17 |
| OpenML Concrete Slump | sparsity-exp-6 | `sparsity:exponential:6.00` | 0.092 | 0.092 | 180 | 0.06813 | 0.3200 | +1.0266 | 4.38 | 6 | 3.32 | 4 | 1.04 | 2 |
| OpenML Concrete Slump | sparsity-exp-8 | `sparsity:exponential:8.00` | 3.602 | 3.461 | 768 | 0.06813 | 0.2309 | +0.9376 | 5.93 | 10 | 3.58 | 4 | 1.07 | 5 |
| OpenML Concrete Slump | sparsity-exp-10 | `sparsity:exponential:10.00` | 2.919 | 2.892 | 768 | 0.06813 | 0.1547 | +0.8614 | 5.90 | 10 | 3.52 | 4 | 1.05 | 5 |
