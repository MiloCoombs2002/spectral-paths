# Complexity-Aware Ridge Benchmark Results

- mode: `fast`
- dataset selection: `all-broad`

| Dataset | Case | Regularization | Cold (s) | Warm (s) | Paths | Lambda | Test R2 | Delta vs Baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenML Concrete Slump | baseline-uniform | `uniform` | 0.078 | 0.011 | 32 | 0.10000 | 0.3747 | +0.0000 |
| OpenML Concrete Slump | total_order-linear | `total_order:linear:1.00` | 0.012 | 0.011 | 32 | 0.10000 | 0.3731 | -0.0016 |
| OpenML Concrete Slump | total_order-exponential | `total_order:exponential:1.00` | 0.011 | 0.011 | 32 | 0.10000 | 0.3672 | -0.0075 |
| OpenML Concrete Slump | sparsity-linear | `sparsity:linear:1.00` | 0.011 | 0.011 | 32 | 0.10000 | 0.3811 | +0.0065 |
| OpenML Concrete Slump | sparsity-exponential | `sparsity:exponential:1.00` | 0.011 | 0.011 | 32 | 0.10000 | 0.3824 | +0.0077 |
| OpenML Concrete Slump | harmonic_order-linear | `harmonic_order:linear:1.00` | 0.011 | 0.011 | 32 | 0.10000 | 0.3714 | -0.0033 |
| OpenML Concrete Slump | harmonic_order-exponential | `harmonic_order:exponential:1.00` | 0.011 | 0.011 | 32 | 0.10000 | 0.3714 | -0.0033 |
| OpenML Yacht Hydrodynamics | baseline-uniform | `uniform` | 0.015 | 0.015 | 50 | 0.00010 | 0.9862 | +0.0000 |
| OpenML Yacht Hydrodynamics | total_order-linear | `total_order:linear:1.00` | 0.020 | 0.020 | 75 | 0.01389 | 0.9870 | +0.0008 |
| OpenML Yacht Hydrodynamics | total_order-exponential | `total_order:exponential:1.00` | 0.022 | 0.021 | 74 | 0.00518 | 0.9886 | +0.0024 |
| OpenML Yacht Hydrodynamics | sparsity-linear | `sparsity:linear:1.00` | 0.016 | 0.016 | 50 | 0.00010 | 0.9862 | -0.0000 |
| OpenML Yacht Hydrodynamics | sparsity-exponential | `sparsity:exponential:1.00` | 0.016 | 0.015 | 50 | 0.00010 | 0.9862 | -0.0000 |
| OpenML Yacht Hydrodynamics | harmonic_order-linear | `harmonic_order:linear:1.00` | 0.023 | 0.022 | 75 | 0.01389 | 0.9880 | +0.0018 |
| OpenML Yacht Hydrodynamics | harmonic_order-exponential | `harmonic_order:exponential:1.00` | 0.023 | 0.021 | 74 | 0.01389 | 0.9878 | +0.0016 |
| PMLB Echo Cardiogram | baseline-uniform | `uniform` | 0.054 | 0.039 | 42 | 0.03728 | 0.4430 | +0.0000 |
| PMLB Echo Cardiogram | total_order-linear | `total_order:linear:1.00` | 0.053 | 0.039 | 42 | 0.01389 | 0.4431 | +0.0001 |
| PMLB Echo Cardiogram | total_order-exponential | `total_order:exponential:1.00` | 0.053 | 0.040 | 42 | 0.01389 | 0.4431 | +0.0001 |
| PMLB Echo Cardiogram | sparsity-linear | `sparsity:linear:1.00` | 0.052 | 0.039 | 42 | 0.03728 | 0.4433 | +0.0002 |
| PMLB Echo Cardiogram | sparsity-exponential | `sparsity:exponential:1.00` | 0.052 | 0.040 | 42 | 0.03728 | 0.4434 | +0.0003 |
| PMLB Echo Cardiogram | harmonic_order-linear | `harmonic_order:linear:1.00` | 0.053 | 0.039 | 42 | 0.01389 | 0.4430 | +0.0000 |
| PMLB Echo Cardiogram | harmonic_order-exponential | `harmonic_order:exponential:1.00` | 0.054 | 0.041 | 42 | 0.01389 | 0.4431 | +0.0000 |
| PMLB Wind Speed | baseline-uniform | `uniform` | 0.097 | 0.044 | 95 | 0.01389 | 0.7947 | +0.0000 |
| PMLB Wind Speed | total_order-linear | `total_order:linear:1.00` | 0.061 | 0.055 | 95 | 0.01389 | 0.7947 | +0.0000 |
| PMLB Wind Speed | total_order-exponential | `total_order:exponential:1.00` | 0.061 | 0.041 | 95 | 0.01389 | 0.7947 | +0.0000 |
| PMLB Wind Speed | sparsity-linear | `sparsity:linear:1.00` | 0.061 | 0.042 | 95 | 0.01389 | 0.7946 | -0.0001 |
| PMLB Wind Speed | sparsity-exponential | `sparsity:exponential:1.00` | 0.058 | 0.043 | 95 | 0.01389 | 0.7946 | -0.0001 |
| PMLB Wind Speed | harmonic_order-linear | `harmonic_order:linear:1.00` | 0.061 | 0.043 | 95 | 0.01389 | 0.7947 | +0.0000 |
| PMLB Wind Speed | harmonic_order-exponential | `harmonic_order:exponential:1.00` | 0.062 | 0.042 | 95 | 0.01389 | 0.7947 | +0.0000 |
| UCI Energy Efficiency | baseline-uniform | `uniform` | 0.067 | 0.041 | 128 | 0.00027 | 0.9976 | +0.0000 |
| UCI Energy Efficiency | total_order-linear | `total_order:linear:1.00` | 0.056 | 0.033 | 128 | 0.00010 | 0.9974 | -0.0001 |
| UCI Energy Efficiency | total_order-exponential | `total_order:exponential:1.00` | 0.055 | 0.033 | 128 | 0.00010 | 0.9974 | -0.0001 |
| UCI Energy Efficiency | sparsity-linear | `sparsity:linear:1.00` | 0.055 | 0.032 | 128 | 0.00027 | 0.9976 | -0.0000 |
| UCI Energy Efficiency | sparsity-exponential | `sparsity:exponential:1.00` | 0.056 | 0.033 | 128 | 0.00027 | 0.9975 | -0.0000 |
| UCI Energy Efficiency | harmonic_order-linear | `harmonic_order:linear:1.00` | 0.060 | 0.036 | 128 | 0.00027 | 0.9975 | -0.0001 |
| UCI Energy Efficiency | harmonic_order-exponential | `harmonic_order:exponential:1.00` | 0.058 | 0.036 | 128 | 0.00027 | 0.9974 | -0.0002 |
| UCI Concrete Compressive Strength | baseline-uniform | `uniform` | 0.068 | 0.042 | 117 | 0.00010 | 0.8855 | +0.0000 |
| UCI Concrete Compressive Strength | total_order-linear | `total_order:linear:1.00` | 0.056 | 0.033 | 117 | 0.00010 | 0.8855 | -0.0001 |
| UCI Concrete Compressive Strength | total_order-exponential | `total_order:exponential:1.00` | 0.058 | 0.034 | 117 | 0.00010 | 0.8855 | -0.0001 |
| UCI Concrete Compressive Strength | sparsity-linear | `sparsity:linear:1.00` | 0.056 | 0.033 | 117 | 0.00010 | 0.8855 | +0.0000 |
| UCI Concrete Compressive Strength | sparsity-exponential | `sparsity:exponential:1.00` | 0.056 | 0.033 | 117 | 0.00010 | 0.8856 | +0.0001 |
| UCI Concrete Compressive Strength | harmonic_order-linear | `harmonic_order:linear:1.00` | 0.058 | 0.035 | 117 | 0.00010 | 0.8854 | -0.0001 |
| UCI Concrete Compressive Strength | harmonic_order-exponential | `harmonic_order:exponential:1.00` | 0.058 | 0.035 | 117 | 0.00010 | 0.8854 | -0.0001 |
