# Complexity-Aware Ridge Benchmark Results

- mode: `overgrow`
- dataset selection: `all-uci`

| Dataset | Case | Regularization | Cold (s) | Warm (s) | Paths | Lambda | Test R2 | Delta vs Baseline | Mean Total Order | Max Total Order | Mean Sparsity | Max Sparsity | Mean Harmonic Order | Max Harmonic Order |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UCI Energy Efficiency | baseline-uniform | `uniform` | 1.068 | 0.890 | 295 | 0.00068 | 0.9977 | +0.0000 | 8.91 | 37 | 2.29 | 4 | 6.74 | 37 |
| UCI Energy Efficiency | total_order-exp-3 | `total_order:exponential:3.00` | 0.255 | 0.236 | 297 | 0.00046 | 0.9977 | +0.0001 | 11.44 | 37 | 2.03 | 4 | 9.71 | 37 |
| UCI Energy Efficiency | total_order-exp-4 | `total_order:exponential:4.00` | 0.505 | 0.478 | 349 | 0.00046 | 0.9978 | +0.0001 | 11.30 | 43 | 2.07 | 4 | 9.35 | 43 |
| UCI Energy Efficiency | total_order-exp-6 | `total_order:exponential:6.00` | 0.254 | 0.237 | 299 | 0.00068 | 0.9976 | -0.0000 | 9.29 | 36 | 1.96 | 4 | 7.45 | 36 |
| UCI Energy Efficiency | total_order-exp-8 | `total_order:exponential:8.00` | 0.498 | 0.477 | 347 | 0.00068 | 0.9974 | -0.0002 | 11.91 | 42 | 1.84 | 4 | 10.21 | 42 |
| UCI Energy Efficiency | total_order-exp-10 | `total_order:exponential:10.00` | 0.504 | 0.482 | 376 | 0.00068 | 0.9974 | -0.0002 | 13.88 | 47 | 1.62 | 4 | 12.32 | 47 |
| UCI Energy Efficiency | sparsity-exp-4 | `sparsity:exponential:4.00` | 0.352 | 0.331 | 295 | 0.00022 | 0.9975 | -0.0001 | 8.88 | 37 | 2.23 | 4 | 6.74 | 37 |
| UCI Energy Efficiency | sparsity-exp-6 | `sparsity:exponential:6.00` | 0.406 | 0.377 | 352 | 0.00022 | 0.9976 | -0.0000 | 8.98 | 44 | 2.29 | 4 | 6.60 | 44 |
| UCI Energy Efficiency | sparsity-exp-8 | `sparsity:exponential:8.00` | 0.380 | 0.352 | 355 | 0.00022 | 0.9978 | +0.0001 | 9.25 | 44 | 2.07 | 4 | 6.89 | 44 |
| UCI Energy Efficiency | sparsity-exp-10 | `sparsity:exponential:10.00` | 0.338 | 0.312 | 373 | 0.00032 | 0.9978 | +0.0001 | 8.41 | 47 | 2.03 | 4 | 5.89 | 47 |
| UCI Concrete Compressive Strength | baseline-uniform | `uniform` | 0.568 | 0.540 | 308 | 0.00001 | 0.8775 | +0.0000 | 9.37 | 39 | 2.43 | 4 | 7.13 | 39 |
| UCI Concrete Compressive Strength | total_order-exp-3 | `total_order:exponential:3.00` | 0.375 | 0.347 | 300 | 0.00002 | 0.8884 | +0.0109 | 7.36 | 38 | 2.61 | 4 | 4.87 | 38 |
| UCI Concrete Compressive Strength | total_order-exp-4 | `total_order:exponential:4.00` | 0.376 | 0.338 | 306 | 0.00005 | 0.8635 | -0.0140 | 8.12 | 38 | 2.47 | 4 | 5.69 | 38 |
| UCI Concrete Compressive Strength | total_order-exp-6 | `total_order:exponential:6.00` | 0.208 | 0.183 | 250 | 0.00046 | 0.9088 | +0.0313 | 7.45 | 31 | 2.51 | 4 | 5.15 | 31 |
| UCI Concrete Compressive Strength | total_order-exp-8 | `total_order:exponential:8.00` | 0.283 | 0.220 | 279 | 0.00005 | 0.8828 | +0.0053 | 8.06 | 27 | 2.24 | 4 | 5.96 | 27 |
| UCI Concrete Compressive Strength | total_order-exp-10 | `total_order:exponential:10.00` | 0.492 | 0.464 | 358 | 0.00002 | 0.8825 | +0.0050 | 12.08 | 39 | 1.96 | 4 | 10.25 | 39 |
| UCI Concrete Compressive Strength | sparsity-exp-4 | `sparsity:exponential:4.00` | 0.604 | 0.554 | 395 | 0.00002 | 0.8782 | +0.0007 | 8.08 | 49 | 2.76 | 4 | 5.21 | 49 |
| UCI Concrete Compressive Strength | sparsity-exp-6 | `sparsity:exponential:6.00` | 1.698 | 1.672 | 538 | 0.00001 | 0.8946 | +0.0171 | 11.22 | 68 | 2.67 | 4 | 8.15 | 68 |
| UCI Concrete Compressive Strength | sparsity-exp-8 | `sparsity:exponential:8.00` | 1.166 | 1.091 | 612 | 0.00001 | 0.9237 | +0.0462 | 10.95 | 77 | 2.77 | 4 | 7.74 | 77 |
| UCI Concrete Compressive Strength | sparsity-exp-10 | `sparsity:exponential:10.00` | 1.110 | 1.135 | 660 | 0.00001 | 0.8800 | +0.0025 | 8.60 | 69 | 2.93 | 4 | 5.09 | 69 |
| UCI Wine Quality | baseline-uniform | `uniform` | 1.352 | 1.330 | 378 | 0.00100 | 0.3534 | +0.0000 | 7.64 | 35 | 2.24 | 4 | 5.79 | 35 |
| UCI Wine Quality | total_order-exp-3 | `total_order:exponential:3.00` | 0.978 | 0.969 | 378 | 0.00100 | 0.3541 | +0.0007 | 7.64 | 35 | 2.24 | 4 | 5.79 | 35 |
| UCI Wine Quality | total_order-exp-4 | `total_order:exponential:4.00` | 0.961 | 0.978 | 378 | 0.00100 | 0.3546 | +0.0013 | 7.64 | 35 | 2.24 | 4 | 5.79 | 35 |
| UCI Wine Quality | total_order-exp-6 | `total_order:exponential:6.00` | 0.964 | 0.954 | 378 | 0.00100 | 0.3566 | +0.0032 | 7.64 | 35 | 2.24 | 4 | 5.79 | 35 |
| UCI Wine Quality | total_order-exp-8 | `total_order:exponential:8.00` | 0.814 | 0.788 | 347 | 0.00068 | 0.3347 | -0.0187 | 6.44 | 29 | 2.37 | 4 | 4.51 | 29 |
| UCI Wine Quality | total_order-exp-10 | `total_order:exponential:10.00` | 0.820 | 0.802 | 352 | 0.00022 | 0.3445 | -0.0089 | 7.07 | 31 | 2.20 | 4 | 5.34 | 31 |
| UCI Wine Quality | sparsity-exp-4 | `sparsity:exponential:4.00` | 1.562 | 1.497 | 465 | 0.00015 | 0.3654 | +0.0120 | 8.82 | 42 | 2.23 | 4 | 6.88 | 42 |
| UCI Wine Quality | sparsity-exp-6 | `sparsity:exponential:6.00` | 3.462 | 3.477 | 560 | 0.00001 | 0.3552 | +0.0018 | 11.52 | 51 | 2.13 | 4 | 9.56 | 51 |
| UCI Wine Quality | sparsity-exp-8 | `sparsity:exponential:8.00` | 0.918 | 0.896 | 369 | 0.00003 | 0.3508 | -0.0026 | 8.28 | 33 | 2.13 | 4 | 6.56 | 33 |
| UCI Wine Quality | sparsity-exp-10 | `sparsity:exponential:10.00` | 3.240 | 3.137 | 768 | 0.00001 | 0.3669 | +0.0135 | 9.33 | 69 | 2.73 | 4 | 6.61 | 69 |
| UCI Phishing Websites | baseline-uniform | `uniform` | 2.261 | 2.237 | 768 | 0.00010 | 0.8156 | +0.0000 | 3.33 | 8 | 2.80 | 4 | 1.31 | 8 |
| UCI Phishing Websites | total_order-exp-3 | `total_order:exponential:3.00` | 1.560 | 1.565 | 768 | 0.00215 | 0.8179 | +0.0023 | 4.70 | 24 | 2.80 | 4 | 2.80 | 24 |
| UCI Phishing Websites | total_order-exp-4 | `total_order:exponential:4.00` | 1.562 | 1.560 | 768 | 0.00100 | 0.8151 | -0.0005 | 4.96 | 24 | 2.71 | 4 | 3.15 | 24 |
| UCI Phishing Websites | total_order-exp-6 | `total_order:exponential:6.00` | 1.569 | 1.568 | 768 | 0.00100 | 0.8158 | +0.0003 | 4.88 | 24 | 2.63 | 4 | 3.19 | 24 |
| UCI Phishing Websites | total_order-exp-8 | `total_order:exponential:8.00` | 1.568 | 1.537 | 768 | 0.00100 | 0.8114 | -0.0042 | 5.81 | 24 | 2.45 | 4 | 4.32 | 24 |
| UCI Phishing Websites | total_order-exp-10 | `total_order:exponential:10.00` | 1.541 | 1.557 | 768 | 0.00046 | 0.8117 | -0.0039 | 5.82 | 22 | 2.32 | 4 | 4.44 | 22 |
| UCI Phishing Websites | sparsity-exp-4 | `sparsity:exponential:4.00` | 1.578 | 1.550 | 768 | 0.00022 | 0.8177 | +0.0021 | 3.68 | 20 | 2.73 | 4 | 1.78 | 20 |
| UCI Phishing Websites | sparsity-exp-6 | `sparsity:exponential:6.00` | 1.576 | 1.532 | 768 | 0.00003 | 0.8162 | +0.0007 | 4.00 | 23 | 2.49 | 4 | 2.33 | 23 |
| UCI Phishing Websites | sparsity-exp-8 | `sparsity:exponential:8.00` | 1.547 | 1.537 | 768 | 0.00001 | 0.8123 | -0.0033 | 2.63 | 4 | 2.22 | 4 | 1.12 | 3 |
| UCI Phishing Websites | sparsity-exp-10 | `sparsity:exponential:10.00` | 1.512 | 1.512 | 768 | 0.00001 | 0.8088 | -0.0068 | 2.59 | 4 | 2.10 | 4 | 1.12 | 3 |
| UCI Superconductivity | baseline-uniform | `uniform` | 2.254 | 2.260 | 768 | 0.00032 | 0.8480 | +0.0000 | 4.27 | 8 | 2.26 | 4 | 3.00 | 8 |
| UCI Superconductivity | total_order-exp-3 | `total_order:exponential:3.00` | 1.984 | 1.981 | 768 | 0.00015 | 0.8475 | -0.0005 | 4.05 | 8 | 2.58 | 4 | 2.48 | 8 |
| UCI Superconductivity | total_order-exp-4 | `total_order:exponential:4.00` | 2.010 | 2.004 | 768 | 0.00022 | 0.8476 | -0.0004 | 4.27 | 8 | 2.26 | 4 | 3.00 | 8 |
| UCI Superconductivity | total_order-exp-6 | `total_order:exponential:6.00` | 1.975 | 1.964 | 768 | 0.00001 | 0.8449 | -0.0031 | 3.05 | 5 | 2.52 | 4 | 1.53 | 5 |
| UCI Superconductivity | total_order-exp-8 | `total_order:exponential:8.00` | 1.995 | 2.005 | 768 | 0.00005 | 0.8462 | -0.0018 | 3.69 | 7 | 2.32 | 4 | 2.37 | 7 |
| UCI Superconductivity | total_order-exp-10 | `total_order:exponential:10.00` | 2.007 | 1.973 | 768 | 0.00003 | 0.8442 | -0.0039 | 3.58 | 7 | 2.21 | 3 | 2.37 | 7 |
| UCI Superconductivity | sparsity-exp-4 | `sparsity:exponential:4.00` | 1.986 | 1.965 | 768 | 0.00001 | 0.8461 | -0.0019 | 4.53 | 9 | 2.42 | 4 | 3.11 | 9 |
| UCI Superconductivity | sparsity-exp-6 | `sparsity:exponential:6.00` | 1.980 | 1.976 | 768 | 0.00001 | 0.8454 | -0.0026 | 4.42 | 9 | 2.31 | 4 | 3.11 | 9 |
| UCI Superconductivity | sparsity-exp-8 | `sparsity:exponential:8.00` | 1.856 | 1.858 | 768 | 0.00001 | 0.8424 | -0.0056 | 2.63 | 4 | 2.63 | 4 | 1.00 | 1 |
| UCI Superconductivity | sparsity-exp-10 | `sparsity:exponential:10.00` | 1.865 | 1.879 | 768 | 0.00001 | 0.8406 | -0.0074 | 2.63 | 4 | 2.63 | 4 | 1.00 | 1 |
