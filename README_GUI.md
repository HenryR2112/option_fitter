# Options Function Approximator GUI

A professional desktop application for decomposing complex financial payoffs into vanilla options with comprehensive pricing and risk analytics.

![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)

## Overview

The GUI provides an interactive interface to the Options Function Approximator library, enabling users to visually explore how complex payoff functions can be replicated using portfolios of vanilla call and put options. The application combines mathematical function approximation with real-world financial pricing through the Black-Scholes model.

## Features

### Core Functionality

- **Function Library**: Choose from 7+ predefined functions or define custom payoffs
- **Real-time Visualization**: Interactive matplotlib charts with zoom, pan, and save capabilities
- **Flexible Basis Selection**: Mix calls, puts, stock positions, and auxiliary functions (Gaussians, sigmoids, polynomials)
- **Cost Analysis**: Detailed breakdown of portfolio cost and composition
- **Risk Metrics**: Complete portfolio Greeks (Delta, Gamma, Vega, Theta) with interpretations
- **Export Capability**: Save results to CSV for further analysis

### User Interface

- **Modern Dark Theme**: Professional, eye-friendly design optimized for extended use
- **Tabbed Results**: Organized display of summary, options breakdown, Greeks, and detailed output
- **Parameter Controls**: Intuitive sliders and input fields for all configuration options
- **Keyboard Shortcuts**: Streamlined workflow with hotkeys for common operations
- **Error Handling**: Clear feedback and validation for all inputs

## Requirements

- Python 3.7 or higher
- NumPy (≥2.0)
- Matplotlib (≥3.0)
- Tkinter (usually pre-installed with Python)

## Installation and Launch

### Quick Start

```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Launch the GUI
python options_gui.py
```

### Verify Tkinter Installation

```bash
python -m tkinter
# Should open a small test window
```

If Tkinter is missing:
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **macOS**: Usually pre-installed; if not, reinstall Python from python.org
- **Windows**: Usually pre-installed with official Python installer

## Using the Application

### Step-by-Step Guide

#### 1. Select a Target Function

The left sidebar provides several options:

**Predefined Functions:**
- **sin(x)**: Smooth oscillation, tests approximation of smooth functions
- **cos(x)**: Shifted sine wave
- **Sigmoid**: S-shaped curve, models transitions
- **Gaussian PDF**: Bell curve, models probability distributions
- **Polynomial (x³ - 2x² + x)**: Tests polynomial approximation
- **Bull Spread**: Classic option strategy (long lower strike call, short higher strike call)
- **Complex Payoff**: Multi-component structured product

**Custom Functions:**
- Enter Python expressions using NumPy
- Variable `x` represents the stock price array
- Examples:
  ```python
  np.sin(x) + 0.5 * np.cos(2*x)          # Multi-frequency
  np.maximum(x - 95, 0) - np.maximum(x - 105, 0)  # Call spread
  x**2 - 5*x + 10                         # Polynomial
  np.exp(-((x-100)**2)/50)                # Gaussian bump
  np.where(x > 100, x - 100, 0)           # Conditional payoff
  ```

#### 2. Configure Price Range

- **Min Stock Price**: Lower bound of approximation domain (typically 0 or lower strike)
- **Max Stock Price**: Upper bound of approximation domain
- **Number of Options**: How many strike prices to use (more = better approximation but higher cost)
  - **4-8 options**: Fast, rough approximation
  - **10-20 options**: Good balance
  - **25+ options**: High accuracy, demonstrates convergence

#### 3. Set Financial Parameters

**Black-Scholes Model Inputs:**
- **Current Stock Price (S₀)**: Spot price, default 100.0
- **Risk-free Rate (r)**: Annualized rate (e.g., 0.05 = 5%), affects discounting
- **Time to Expiration (T)**: Years until maturity (e.g., 0.25 = 3 months)
- **Volatility (σ)**: Annualized volatility (e.g., 0.20 = 20%)

**Effects on Pricing:**
- Higher volatility → higher option premiums
- Longer time → higher premiums (more time value)
- Higher rate → lower put values, higher call values

#### 4. Choose Basis Functions

**Core Options:**
- **Use Call Options**: Include (x - K)₊ basis functions
- **Use Put Options**: Include (K - x)₊ basis functions
- **Use Stock Position**: Include linear term (Δx)

**Auxiliary Functions** (optional, for enhanced approximation):
- **Gaussians**: Smooth bumps for local features
- **Sigmoids**: Smooth transitions
- **Polynomials**: Global polynomial terms (x², x³, etc.)

**Recommendations:**
- For piecewise linear payoffs: Calls + Puts sufficient
- For smooth functions: Add Gaussians/Sigmoids
- For polynomial functions: Enable Polynomials

#### 5. Approximation Parameters

- **Regularization (λ)**: Penalty for large weights, prevents overfitting
  - **0**: No regularization (may overfit)
  - **1e-6 to 1e-4**: Light regularization (recommended)
  - **1e-3+**: Strong regularization (smoother, less accurate)
- **Sample Points**: Number of evaluation points (default 1000, higher = slower but more accurate)

#### 6. Calculate and Analyze

Click **"Calculate Approximation"** (or press `Ctrl+R`/`⌘R`)

**Output Tabs:**

1. **Summary Tab**:
   - Approximation error (MSE, RMSE)
   - Total portfolio cost
   - Portfolio composition (number of calls, puts, stock position)
   - Cost efficiency metrics

2. **Options Breakdown Tab**:
   - **Call Options**: Table with Strike, Weight, Premium, Cost
   - **Put Options**: Table with Strike, Weight, Premium, Cost
   - Sortable columns for analysis

3. **Portfolio Greeks Tab**:
   - **Delta (Δ)**: Portfolio sensitivity to $1 change in stock price
   - **Gamma (Γ)**: Rate of change of Delta (convexity)
   - **Vega (ν)**: Sensitivity to 1% change in volatility
   - **Theta (Θ)**: Time decay ($ per day)
   - Interpretations and risk implications

4. **Details Tab**:
   - Complete text output
   - Detailed cost breakdown
   - Technical details

**Visualization**:
- **Blue line**: Target function
- **Orange line**: Options approximation
- **Interactive tools**: Zoom, pan, save figure
- **Tooltips**: Hover for coordinate values

### Advanced Usage

#### Custom Strike Placement

By default, strikes are uniformly spaced. For advanced users:
1. Modify `options_func_maker.py` to accept custom strike array
2. Use denser strikes near points of interest (kinks, discontinuities)

#### Optimization

For large approximations:
- Reduce `n_points` (sample points) from 1000 to 500
- Use fewer options initially, increase iteratively
- Enable regularization to avoid numerical instability

#### Batch Analysis

For systematic studies:
```python
from options_func_maker import OptionsFunctionApproximator
import numpy as np

for n in [5, 10, 15, 20, 25]:
    approx = OptionsFunctionApproximator(n_options=n, ...)
    weights, mse = approx.approximate(target_func)
    print(f"n={n}, RMSE={np.sqrt(mse):.6f}, Cost={approx.get_total_cost():.2f}")
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` / `⌘R` | Calculate approximation |
| `Ctrl+E` / `⌘E` | Export results to CSV |
| `Ctrl+Q` / `⌘Q` | Quit application |

## Output Files

**CSV Export** (Ctrl+E):
- Filename: `options_approximation_YYYYMMDD_HHMMSS.csv`
- Contents: Strike prices, option types, weights, premiums, costs
- Location: Current working directory

## Understanding the Results

### Approximation Quality

- **MSE (Mean Squared Error)**: Average squared deviation, lower is better
- **RMSE (Root Mean Squared Error)**: More interpretable, same units as target function
  - RMSE < 0.01: Excellent approximation
  - RMSE < 0.1: Good approximation
  - RMSE > 0.5: Poor approximation (increase options or adjust parameters)

### Cost Analysis

- **Individual Option Cost**: |weight| × Black-Scholes premium
- **Total Cost**: Sum of all option costs + stock position
- **Long Position**: Positive weight (pay premium)
- **Short Position**: Negative weight (receive premium, but counted as cost in analysis)

### Portfolio Greeks

| Greek | Interpretation | Typical Range |
|-------|---------------|---------------|
| Delta (Δ) | $ change per $1 stock move | -∞ to +∞ |
| Gamma (Γ) | Δ change per $1 stock move | 0 to +∞ |
| Vega (ν) | $ change per 1% vol increase | 0 to +∞ |
| Theta (Θ) | $ decay per day | -∞ to 0 (usually negative) |

**Risk Management:**
- High |Δ|: Large directional exposure
- High Γ: Position changes rapidly with stock price
- High Vega: Sensitive to volatility changes
- Large negative Θ: Position loses value over time

## Troubleshooting

### Common Issues

**Problem**: "ImportError: No module named tkinter"
- **Solution**: Install tkinter for your OS (see Installation section)

**Problem**: GUI launches but window is tiny/distorted
- **Solution**: Check display scaling settings; try different Python version

**Problem**: "numpy.linalg.LinAlgError: Singular matrix"
- **Solution**:
  - Reduce number of options
  - Increase regularization parameter
  - Check that price range covers target function support

**Problem**: High approximation error
- **Solution**:
  - Increase number of options
  - Adjust price range to tightly fit function domain
  - Try different basis function combinations
  - Reduce regularization

**Problem**: Negative or unrealistic costs
- **Solution**: Check Black-Scholes parameters (S₀, r, T, σ); ensure they're reasonable

**Problem**: GUI freezes during calculation
- **Solution**: Reduce `n_points` or number of options; computation is O(n³) for n options

### Performance Tips

- **Large approximations**: Use `n_points=500` instead of 1000
- **Real-time experimentation**: Start with 5-10 options, scale up
- **Memory constraints**: Limit to <50 options for most systems

## Technical Details

### Algorithm

1. **Basis Construction**: Creates φᵢ(x) basis functions at specified strikes
2. **Matrix Formation**: Builds design matrix Φ where Φᵢⱼ = φⱼ(xᵢ)
3. **Least Squares**: Solves min ||Φw - f||² + λ||w||² using NumPy
4. **Pricing**: Applies Black-Scholes formula to each option
5. **Greeks Calculation**: Uses closed-form Black-Scholes Greeks

### Approximation Theory

The GUI demonstrates the density theorem: span{1, x, (x-K)₊} is dense in C([a,b]). This means any continuous function can be approximated arbitrarily well using options.

### Financial Interpretation

Each approximation corresponds to a **static replication portfolio**:
- Buy/sell calls and puts at various strikes
- Hold/short stock position
- Portfolio payoff ≈ target payoff at expiration

## Examples

### Example 1: Replicating a Bull Spread

1. Select "Bull Spread" from presets
2. Set price range: [80, 120], S₀ = 100
3. Use 5 options, calls + puts
4. Calculate
5. **Expected result**: Should decompose into approximately 1 long call at K=90, 1 short call at K=110

### Example 2: Approximating sin(x)

1. Select "sin(x)"
2. Price range: [0, 2π] ≈ [0, 6.28]
3. Use 15-20 options
4. Enable calls and puts
5. **Expected result**: RMSE < 0.05, cost depends on parameters

### Example 3: Custom Structured Product

1. Select "Custom Function"
2. Enter: `np.maximum(x - 95, 0) - 2*np.maximum(x - 100, 0) + np.maximum(x - 105, 0)`
3. Price range: [90, 110]
4. Calculate
5. **Expected result**: Butterfly spread (should identify three strikes near 95, 100, 105)

## Notes and Limitations

- **European Options Only**: Black-Scholes applies to European-style options
- **Single Underlying**: Currently limited to univariate functions
- **Static Replication**: No dynamic hedging or time evolution
- **Transaction Costs Ignored**: Real markets have bid-ask spreads and fees
- **Discrete Strikes**: Real markets have finite strike availability
- **Model Assumptions**: Black-Scholes assumes constant volatility, no dividends, continuous trading

## Educational Use

This GUI is ideal for:
- **Finance Students**: Understanding option strategies and payoff decomposition
- **Quantitative Finance**: Exploring replication and hedging
- **Machine Learning**: Seeing connections between ReLU networks and option portfolios
- **Approximation Theory**: Visualizing spline approximation with hinge functions

## Further Reading

- [Main README](README.md): Complete project documentation
- [Experiments](experiments/README.md): Convergence studies
- [Lean Proofs](proof/README.md): Formal mathematical verification
- [LaTeX Manuscript](tex/draft.tex): Academic paper on theoretical foundations

## Support

- **Issues**: Report bugs at [GitHub Issues](https://github.com/HenryR2112/option_fitter/issues)
- **Questions**: Use [GitHub Discussions](https://github.com/HenryR2112/option_fitter/discussions)

---

**Happy approximating!** 📈
