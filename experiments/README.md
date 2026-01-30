# Numerical Experiments

This directory contains numerical experiments demonstrating the convergence properties of option-basis function approximations.

## Overview

The experiments empirically verify the **density theorem**: as the number of options (basis functions) increases, the approximation error decreases, converging to the target function.

## Running the Experiments

### Quick Start

```bash
# From the repository root
python experiments/run_experiments.py
```

### What It Does

The experiment:
1. Approximates `sin(x)` on `[0, 2π]` using varying numbers of options
2. Tests with `n = [4, 8, 12, 16, 24, 32]` strike points
3. Measures RMSE (Root Mean Squared Error) for each configuration
4. Generates convergence plots and data tables

### Output Files

**1. results.csv**
- Columns: `n_options`, `rmse`, `mse`
- Contains convergence data for analysis
- Can be imported into spreadsheets or Jupyter notebooks

**2. rmse_vs_n.png**
- Visualization of approximation error vs. number of options
- Shows convergence trend (error decreases as n increases)
- Demonstrates empirical validation of density theorem

## Expected Results

For approximating `sin(x)` on `[0, 2π]`:

| Number of Options | Expected RMSE |
|-------------------|---------------|
| 4 | ~0.1 - 0.3 |
| 8 | ~0.05 - 0.1 |
| 12 | ~0.02 - 0.05 |
| 16 | ~0.01 - 0.02 |
| 24 | ~0.005 - 0.01 |
| 32 | <0.005 |

**Convergence Rate**: For smooth functions like `sin(x)`, we observe approximately O(n^(-2)) convergence due to smoothness.

## Customizing Experiments

### Test Different Functions

Edit `run_experiments.py` to change the target function:

```python
def target_sin(x):
    return np.sin(x)  # Change this to test other functions

# Examples:
# return np.cos(x)
# return np.exp(-x)
# return np.maximum(x - np.pi, 0)  # Call option payoff
# return x**2 - 3*x + 2  # Polynomial
```

### Adjust Number of Options

Modify the `n_list` parameter:

```python
# In __main__ section
n_list = [4, 8, 12, 16, 24, 32]  # Modify this array

# For finer resolution:
# n_list = range(5, 51, 5)  # [5, 10, 15, ..., 50]

# For large-scale convergence study:
# n_list = [10, 20, 40, 80, 160]  # Doubling sequence
```

### Change Price Range

Adjust the domain of approximation:

```python
# In run_experiment function
price_range = (0, 2 * np.pi)  # Current: [0, 2π]

# Examples:
# price_range = (0, 10)  # [0, 10]
# price_range = (-5, 5)  # [-5, 5]
# price_range = (80, 120)  # Stock price range
```

### Modify Regularization

Control the regularization parameter:

```python
# In run_experiment function
weights, mse = approximator.approximate(
    target_sin, n_points=n_points, regularization=1e-6  # Modify this
)

# regularization=0: No regularization (may overfit)
# regularization=1e-6: Light (recommended for convergence studies)
# regularization=1e-3: Strong (smoother solutions, slower convergence)
```

## Understanding the Results

### Convergence Plot Interpretation

The `rmse_vs_n.png` plot shows:
- **X-axis**: Number of options (strike points) used
- **Y-axis**: RMSE (approximation error)
- **Trend**: Should decrease as n increases, demonstrating convergence

**Indicators of Success:**
- ✅ Monotonically decreasing error (or mostly decreasing)
- ✅ Error approaches zero as n → ∞
- ✅ Smooth curve (no erratic jumps)

**Potential Issues:**
- ❌ Error plateaus: May indicate numerical instability (increase regularization)
- ❌ Error increases: Check implementation or parameters
- ❌ Non-smooth curve: Typical for non-smooth target functions (expected)

### Convergence Rates

Different function classes have different convergence rates:

| Function Class | Convergence Rate | Example |
|----------------|------------------|---------|
| Piecewise Linear | Finite (exact with sufficient n) | Bull spread |
| Lipschitz | O(n^(-1)) | Absolute value |
| C¹ (smooth) | O(n^(-2)) | sin(x), sigmoid |
| C² or higher | O(n^(-2)) or better | Gaussian |

**Why?** Smoother functions are easier to approximate because they have less local variation.

## Advanced Experiments

### Comparing Basis Functions

Test different combinations:

```python
# Only calls
approx = OptionsFunctionApproximator(n_options=n, use_calls=True, use_puts=False)

# Only puts
approx = OptionsFunctionApproximator(n_options=n, use_calls=False, use_puts=True)

# Calls + puts + stock
approx = OptionsFunctionApproximator(n_options=n, use_calls=True, use_puts=True, use_stock=True)

# With auxiliary functions
approx = OptionsFunctionApproximator(n_options=n, use_calls=True, use_gaussians=True, n_gaussians=5)
```

### Measuring Computational Cost

Add timing to assess scalability:

```python
import time

start = time.time()
weights, mse = approximator.approximate(target_sin, n_points=1000)
elapsed = time.time() - start
print(f"n={n}, RMSE={np.sqrt(mse):.6e}, Time={elapsed:.3f}s")
```

**Expected Complexity**: O(n³) due to least-squares solve (matrix inversion)

### Studying Cost vs. Accuracy Trade-off

Track both approximation error and portfolio cost:

```python
weights, mse = approximator.approximate(target_sin)
cost = approximator.get_total_cost()
print(f"n={n}, RMSE={np.sqrt(mse):.6f}, Cost=${cost:.2f}")
```

This reveals the **Pareto frontier**: trade-off between replication accuracy and financial cost.

## Sample Output

```
n_options=  4  RMSE=1.234567e-01  MSE_reported=1.523456e-02
n_options=  8  RMSE=6.789012e-02  MSE_reported=4.608920e-03
n_options= 12  RMSE=3.456789e-02  MSE_reported=1.194893e-03
n_options= 16  RMSE=1.789012e-02  MSE_reported=3.200564e-04
n_options= 24  RMSE=7.890123e-03  MSE_reported=6.225404e-05
n_options= 32  RMSE=4.012345e-03  MSE_reported=1.609891e-05
Results saved to experiments/results.csv and experiments/rmse_vs_n.png
```

## Interpreting CSV Data

The `results.csv` file can be analyzed further:

```python
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('experiments/results.csv')

# Compute convergence rate
df['log_n'] = np.log(df['n_options'])
df['log_rmse'] = np.log(df['rmse'])

# Fit log-log regression to estimate rate
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['log_n'], df['log_rmse'])
print(f"Estimated convergence rate: O(n^{slope:.2f})")
# For smooth functions, expect slope ≈ -2
```

## Dependencies

All dependencies are listed in `../requirements.txt`:
- **numpy**: Numerical arrays and linear algebra
- **matplotlib**: Plotting and visualization
- **options_func_maker**: Core approximation library (local module)

Install with:
```bash
pip install -r requirements.txt
```

## Troubleshooting

**Problem**: Script hangs or takes very long
- **Solution**: Reduce `n_points` from 1000 to 500, or use fewer options

**Problem**: "Singular matrix" error
- **Solution**: Increase `regularization` parameter to 1e-4 or 1e-3

**Problem**: Poor convergence (error doesn't decrease)
- **Solution**:
  - Check that price range matches function domain
  - Ensure target function is continuous
  - Try increasing regularization

**Problem**: Results look different from expected
- **Solution**: Random seed? Try setting `np.random.seed(42)` for reproducibility

## Integration with Main Project

These experiments validate the theoretical claims in:
- **Main README**: Demonstrates density theorem empirically
- **LaTeX Paper** (`tex/draft.tex`): Provides numerical evidence for Theorem 2.1
- **Lean Proofs** (`proof/`): Complements formal verification with practical validation

## Further Exploration

### Ideas for Additional Experiments

1. **Function Class Study**: Test convergence rates for different function types (discontinuous, smooth, polynomial)
2. **Strike Placement**: Compare uniform vs. adaptive strike spacing
3. **Regularization Study**: How does λ affect approximation quality and portfolio sparsity?
4. **Cost Analysis**: For what values of n does cost become prohibitive?
5. **Greeks Stability**: How do portfolio Greeks change as n increases?

### Extending the Code

Add new experiment types by:
1. Creating new functions in `run_experiments.py`
2. Modifying the plotting section for custom visualizations
3. Exporting additional metrics (cost, Greeks, computation time)

## References

- Main documentation: [../README.md](../README.md)
- GUI usage: [../README_GUI.md](../README_GUI.md)
- Formal proofs: [../proof/README.md](../proof/README.md)
- Academic paper: [../tex/draft.tex](../tex/draft.tex)

---

**Experiment responsibly!** 🧪📊
