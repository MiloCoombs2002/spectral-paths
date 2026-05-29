# Focused Overgrow Sweep Results

- mode: `overgrow`
- case set: `focused-overgrow`
- completed datasets:
  - `openml:concrete-slump`
  - `openml:yacht-hydrodynamics`
  - `uci:energy-efficiency`
- fetches failed during this session due DNS/network lookup issues:
  - `pmlb:echo-months`
  - `pmlb:wind-speed`
  - `uci:concrete-strength`

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
