# Options Function Approximator

A comprehensive Python library and GUI application for approximating arbitrary functions using combinations of financial options and other basis functions. The system decomposes complex payoffs into vanilla options with full Black-Scholes pricing and risk metrics (Greeks).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Mathematical Foundation](#mathematical-foundation)
- [Installation](#installation)
- [Core Library Usage](#core-library-usage)
- [GUI Application](#gui-application)
- [API Reference](#api-reference)
- [Basis Functions](#basis-functions)
- [Pricing and Risk Metrics](#pricing-and-risk-metrics)
- [Examples](#examples)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)

## Overview

The Options Function Approximator uses options (calls and puts) as basis functions to approximate arbitrary mathematical functions. This approach is particularly useful in quantitative finance for:

- **Structured Product Decomposition**: Breaking down complex financial products into vanilla options
- **Payoff Replication**: Creating portfolios that replicate target payoff functions
- **Risk Analysis**: Understanding the Greeks and cost structure of complex payoffs
- **Function Approximation**: Using options as a basis set for general function approximation

### Key Features

- **Multiple Basis Functions**: Options (calls/puts), stock positions, Gaussians, sigmoids, polynomials, and more
- **Black-Scholes Pricing**: Full European option pricing with customizable parameters
- **Portfolio Greeks**: Delta, Gamma, Vega, and Theta calculations
- **Cost Analysis**: Detailed breakdown of replication costs
- **Interactive GUI**: User-friendly interface for exploration and analysis
- **Custom Functions**: Support for user-defined target functions

## Architecture

### Core Components

1. **`OptionsFunctionApproximator` Class**: Main approximator engine
   - Manages basis function collection
   - Performs least squares optimization
   - Calculates pricing and risk metrics

2. **Basis Function Library**: Collection of mathematical functions
   - Option payoffs (call/put)
   - Statistical functions (Gaussian PDF/CDF, sigmoid)
   - Financial functions (Black-Scholes delta, volatility smile)
   - Polynomial terms

3. **Pricing Engine**: Black-Scholes implementation
   - European call/put pricing
   - Greeks calculation (Delta, Gamma, Vega, Theta)
   - Portfolio-level aggregation

4. **GUI Application**: Tkinter-based interface
   - Function selection and customization
   - Real-time visualization
   - Results display and export

### Design Patterns

- **Strategy Pattern**: Different optimization methods (least squares, scipy minimize)
- **Factory Pattern**: Dynamic basis function creation
- **Observer Pattern**: GUI updates based on calculation results

## Mathematical Foundation

### Function Approximation Problem

Given a target function $f(x)$ over domain $[x_{min}, x_{max}]$, find weights $w_i$ such that:

$$\hat{f}(x) = \sum_{i=1}^{n} w_i \phi_i(x) \approx f(x)$$

where $\phi_i(x)$ are basis functions (options, Gaussians, etc.).

### Optimization Objective

Minimize the mean squared error (MSE) with optional L2 regularization:

$$\min_{\mathbf{w}} \frac{1}{N} \sum_{j=1}^{N} \left( \sum_{i=1}^{n} w_i \phi_i(x_j) - f(x_j) \right)^2 + \lambda \|\mathbf{w}\|^2$$

where:
- $N$ is the number of sample points
- $\lambda$ is the regularization parameter
- $\mathbf{w} = [w_1, w_2, ..., w_n]^T$ is the weight vector

### Solution Methods

1. **Least Squares (Normal Equations)**:
   $$\mathbf{w} = (\mathbf{A}^T \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^T \mathbf{b}$$
   
   where $\mathbf{A}_{ij} = \phi_i(x_j)$ and $\mathbf{b}_j = f(x_j)$

2. **Scipy Minimize**: Uses L-BFGS-B algorithm for constrained optimization

### Option Payoff Functions

- **Call Option**: $\max(S - K, 0)$
- **Put Option**: $\max(K - S, 0)$

where $S$ is stock price and $K$ is strike price.

### Black-Scholes Pricing

**Call Option Price**:
$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

**Put Option Price**:
$$P = K e^{-rT} N(-d_2) - S_0 N(-d_1)$$

where:
- $d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
- $d_2 = d_1 - \sigma\sqrt{T}$
- $N(\cdot)$ is the cumulative normal distribution
- $S_0$ is current stock price
- $K$ is strike price
- $r$ is risk-free rate
- $T$ is time to expiration
- $\sigma$ is volatility

### Portfolio Greeks

For a portfolio with weights $w_i$:

- **Delta**: $\Delta_{portfolio} = \sum_i w_i \Delta_i$
- **Gamma**: $\Gamma_{portfolio} = \sum_i w_i \Gamma_i$
- **Vega**: $\nu_{portfolio} = \sum_i w_i \nu_i$
- **Theta**: $\Theta_{portfolio} = \sum_i w_i \Theta_i$

## Installation

### Requirements

- Python 3.7 or higher
- NumPy
- Matplotlib
- Tkinter (usually included with Python)
- SciPy (optional, for advanced optimization)

### Install Dependencies

```bash
pip install numpy matplotlib scipy
```

### Verify Installation

```python
python -c "import numpy, matplotlib; print('Installation successful')"
```

## Core Library Usage

### Basic Example

```python
from options_func_maker import OptionsFunctionApproximator
import numpy as np

# Define target function
def sin_function(x):
    return np.sin(x)

# Create approximator
approximator = OptionsFunctionApproximator(
    n_options=15,
    price_range=(0, 2*np.pi),
    use_calls=True,
    use_puts=True,
    use_stock=True,
    S0=100.0,      # Current stock price
    r=0.05,        # Risk-free rate (5%)
    T=0.25,        # Time to expiration (3 months)
    sigma=0.2      # Volatility (20%)
)

# Find approximation
weights, mse = approximator.approximate(
    sin_function,
    n_points=1000,
    regularization=0.001,
    method="least_squares"
)

print(f"MSE: {mse:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")

# Evaluate approximation
stock_prices = np.linspace(0, 2*np.pi, 100)
approx_values = approximator.evaluate(stock_prices)

# Calculate cost breakdown
approximator.print_cost_breakdown()

# Get portfolio Greeks
greeks = approximator.calculate_portfolio_greeks()
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Vega: {greeks['vega']:.2f}")
print(f"Theta: {greeks['theta']:.4f}")
```

### Advanced Configuration

```python
approximator = OptionsFunctionApproximator(
    n_options=10,
    price_range=(0, 10),
    use_calls=True,
    use_puts=True,
    use_stock=True,
    # Additional basis functions
    use_gaussians=True,
    n_gaussians=5,
    use_sigmoids=True,
    n_sigmoids=5,
    use_cumulative_normal=True,
    n_cumulative_normal=3,
    use_polynomials=True,
    max_polynomial_power=3,
    # Pricing parameters
    S0=100.0,
    r=0.05,
    T=0.25,
    sigma=0.2
)
```

## GUI Application

### Launching the GUI

**Windows**:
```bash
run_gui.bat
```

**Linux/Mac**:
```bash
chmod +x run_gui.sh
./run_gui.sh
```

**Direct Python**:
```bash
python options_gui.py
```

### GUI Workflow

1. **Select Function**: Choose from predefined functions or enter custom Python code
2. **Set Parameters**: Configure price range, number of options, and pricing parameters
3. **Choose Basis Functions**: Select which basis functions to use
4. **Calculate**: Click "Calculate Approximation" to generate results
5. **Review Results**: Examine plot, cost breakdown, and portfolio Greeks

### Custom Functions in GUI

Enter Python expressions using numpy. The variable `x` represents the stock price array.

Examples:
- `np.sin(x) + 0.5*x`
- `np.maximum(x - 100, 0)`
- `x**2 - 5*x + 10`
- `np.exp(-x/10)`

## API Reference

### `OptionsFunctionApproximator`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_options` | int | 10 | Number of options (calls/puts) |
| `price_range` | tuple | (0, 10) | (min_price, max_price) domain |
| `use_calls` | bool | True | Include call options |
| `use_puts` | bool | True | Include put options |
| `use_stock` | bool | True | Include stock position |
| `use_gaussians` | bool | False | Include Gaussian PDFs |
| `n_gaussians` | int | 5 | Number of Gaussian functions |
| `use_sigmoids` | bool | False | Include sigmoid functions |
| `n_sigmoids` | int | 5 | Number of sigmoid functions |
| `use_cumulative_normal` | bool | False | Include cumulative normal CDFs |
| `n_cumulative_normal` | int | 5 | Number of CDF functions |
| `use_exponential_decay` | bool | False | Include exponential decay |
| `n_exponential_decay` | int | 5 | Number of decay functions |
| `use_polynomials` | bool | False | Include polynomial terms |
| `max_polynomial_power` | int | 3 | Maximum polynomial power |
| `use_log_normal` | bool | False | Include log-normal functions |
| `n_log_normal` | int | 3 | Number of log-normal functions |
| `use_black_scholes_delta` | bool | False | Include BS delta functions |
| `n_black_scholes_delta` | int | 5 | Number of BS delta functions |
| `use_volatility_smile` | bool | False | Include volatility smile |
| `n_volatility_smile` | int | 3 | Number of smile functions |
| `S0` | float | 100.0 | Current stock price |
| `r` | float | 0.05 | Risk-free rate |
| `T` | float | 0.25 | Time to expiration (years) |
| `sigma` | float | 0.2 | Volatility |

#### Methods

##### `approximate(target_function, n_points=1000, regularization=0.0, method="least_squares")`

Find optimal weights to approximate the target function.

**Parameters**:
- `target_function`: Callable function f(x) → array
- `n_points`: Number of sample points for optimization
- `regularization`: L2 regularization strength
- `method`: "least_squares" or "minimize"

**Returns**: `(weights, mse_error)` tuple

##### `evaluate(stock_prices)`

Evaluate the approximated function at given stock prices.

**Parameters**:
- `stock_prices`: NumPy array of stock prices

**Returns**: NumPy array of approximated function values

##### `calculate_premiums()`

Calculate Black-Scholes premiums for all options.

**Returns**: `(premiums_array, premium_dict)` tuple

##### `calculate_portfolio_greeks()`

Calculate portfolio-level Greeks.

**Returns**: Dictionary with keys: `delta`, `gamma`, `vega`, `theta`

##### `get_total_cost()`

Calculate total cost of the option portfolio.

**Returns**: Total cost (float)

##### `print_cost_breakdown()`

Print detailed cost breakdown to console.

##### `plot_approximation(target_function, n_points=1000, title=None, save_path=None)`

Plot target function vs. approximation.

**Parameters**:
- `target_function`: Original function to plot
- `n_points`: Number of points for plotting
- `title`: Plot title (optional)
- `save_path`: Path to save plot (optional)

##### `print_weights()`

Print weights for each basis function.

### Standalone Functions

#### Option Payoffs

- `call_payoff(stock_price, strike)` → `max(S - K, 0)`
- `put_payoff(stock_price, strike)` → `max(K - S, 0)`

#### Black-Scholes Pricing

- `black_scholes_call(S, K, T, r, sigma)` → Call option price
- `black_scholes_put(S, K, T, r, sigma)` → Put option price
- `black_scholes_delta_call(S, K, T, r, sigma)` → Call delta
- `black_scholes_delta_put(S, K, T, r, sigma)` → Put delta
- `black_scholes_gamma(S, K, T, r, sigma)` → Gamma
- `black_scholes_vega(S, K, T, r, sigma)` → Vega
- `black_scholes_theta_call(S, K, T, r, sigma)` → Call theta
- `black_scholes_theta_put(S, K, T, r, sigma)` → Put theta

#### Basis Functions

- `gaussian_pdf(x, mu, sigma)` → Gaussian PDF
- `sigmoid(x, center, scale)` → Sigmoid function
- `cumulative_normal(x, mu, sigma)` → Normal CDF
- `exponential_decay(x, center, decay_rate)` → Exponential decay
- `polynomial_term(x, power)` → x^power
- `log_normal(x, mu, sigma)` → Log-normal function
- `volatility_smile(x, center, vol_atm, vol_skew)` → Volatility smile

## Basis Functions

### Option-Based

- **Call Options**: `max(S - K, 0)` - Provide upward exposure
- **Put Options**: `max(K - S, 0)` - Provide downward exposure
- **Stock Position**: `S` - Linear term

### Statistical Functions

- **Gaussian PDF**: Normal probability density function
- **Cumulative Normal**: Normal CDF (used in Black-Scholes)
- **Sigmoid**: Smooth S-shaped transition function

### Financial Functions

- **Black-Scholes Delta**: Option price sensitivity
- **Volatility Smile**: Implied volatility as function of strike

### Other Functions

- **Polynomials**: `x^n` terms
- **Exponential Decay**: Time decay modeling
- **Log-Normal**: Asset price modeling

## Pricing and Risk Metrics

### Cost Calculation

The total cost of replicating a function is:

$$Cost = \sum_{i \in calls} |w_i| \cdot C_i + \sum_{i \in puts} |w_i| \cdot P_i + |w_{stock}| \cdot S_0$$

where:
- $C_i$ is the call option premium
- $P_i$ is the put option premium
- $w_i$ are the weights from optimization

### Portfolio Greeks

- **Delta ($\Delta$)**: Sensitivity to stock price changes
  - Units: $ per $1 stock move
  
- **Gamma ($\Gamma$)**: Sensitivity of delta to stock price
  - Units: Delta change per $1 stock move
  
- **Vega ($\nu$)**: Sensitivity to volatility changes
  - Units: $ per 1% volatility change
  
- **Theta ($\Theta$)**: Time decay
  - Units: $ per day

## Examples

### Example 1: Approximating sin(x)

```python
from options_func_maker import OptionsFunctionApproximator
import numpy as np

def sin_function(x):
    return np.sin(x)

approximator = OptionsFunctionApproximator(
    n_options=15,
    price_range=(0, 2*np.pi),
    use_calls=True,
    use_puts=True,
    use_stock=True
)

weights, mse = approximator.approximate(sin_function)
approximator.plot_approximation(sin_function, title="sin(x) Approximation")
```

### Example 2: Complex Payoff Decomposition

```python
def complex_payoff(x):
    strike1, strike2 = 90.0, 110.0
    bull_spread = np.maximum(x - strike1, 0) - np.maximum(x - strike2, 0)
    vol_component = 5.0 * np.exp(-((x - 100.0)**2) / (2 * 10.0**2))
    return bull_spread + vol_component

approximator = OptionsFunctionApproximator(
    n_options=15,
    price_range=(80, 120),
    S0=100.0,
    r=0.05,
    T=0.25,
    sigma=0.2
)

weights, mse = approximator.approximate(complex_payoff)
approximator.print_cost_breakdown()
greeks = approximator.calculate_portfolio_greeks()
```

### Example 3: Using Advanced Basis Functions

```python
approximator = OptionsFunctionApproximator(
    n_options=10,
    price_range=(0, 10),
    use_calls=True,
    use_puts=True,
    use_stock=True,
    use_gaussians=True,
    n_gaussians=5,
    use_sigmoids=True,
    n_sigmoids=5,
    use_polynomials=True,
    max_polynomial_power=3
)

weights, mse = approximator.approximate(target_function)
```

## File Structure

```
option_fitter/
├── options_func_maker.py    # Core library and approximator class
├── options_gui.py           # GUI application
├── run_gui.bat             # Windows launcher script
├── run_gui.sh              # Linux/Mac launcher script
├── README.md               # This file
└── README_GUI.md          # GUI-specific documentation
```

### File Descriptions

- **`options_func_maker.py`**: Contains the `OptionsFunctionApproximator` class, all basis functions, Black-Scholes pricing, and example functions
- **`options_gui.py`**: Tkinter-based GUI application for interactive use
- **`run_gui.bat`**: Windows batch script to launch GUI
- **`run_gui.sh`**: Unix shell script to launch GUI
- **`README_GUI.md`**: Quick reference for GUI usage

## Dependencies

### Required

- **NumPy** (>=1.17.0): Numerical computations and array operations
- **Matplotlib** (>=3.1.0): Plotting and visualization

### Optional

- **SciPy** (>=1.3.0): Advanced optimization methods (L-BFGS-B)

### Python Version

- Python 3.7 or higher

## Performance Considerations

### Optimization Tips

1. **Sample Points**: More points improve accuracy but increase computation time
   - Default: 1000 points
   - Range: 100-10000 recommended

2. **Number of Options**: More options improve approximation but increase:
   - Computation time (O(n²) for least squares)
   - Memory usage
   - Cost (more options to purchase)

3. **Regularization**: Helps prevent overfitting
   - Typical range: 0.001 - 0.1
   - Higher values: smoother but less accurate
   - Lower values: more accurate but potentially unstable

4. **Basis Function Selection**: Choose basis functions appropriate for target
   - Options: Good for piecewise linear/payoff functions
   - Gaussians: Good for smooth, localized features
   - Polynomials: Good for smooth, global functions

### Computational Complexity

- **Basis Evaluation**: O(n × m) where n = basis functions, m = sample points
- **Least Squares**: O(n² × m) for matrix multiplication
- **Matrix Solve**: O(n³) for Cholesky decomposition
- **Overall**: O(n² × m + n³) per approximation

### Memory Usage

- Basis matrix: n × m floats (typically 4-8 bytes each)
- Example: 100 basis functions × 1000 points = 800 KB

## Contributing

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Document all public functions and classes
- Include docstrings for all methods

### Testing

Before submitting changes:

1. Test with various target functions
2. Verify pricing calculations match known values
3. Check that Greeks are calculated correctly
4. Ensure GUI remains responsive

### Adding New Basis Functions

1. Implement the function in `options_func_maker.py`
2. Add configuration parameters to `OptionsFunctionApproximator.__init__`
3. Add basis function creation logic in `__init__`
4. Update documentation
5. Add example usage

### Reporting Issues

Include:
- Python version
- Operating system
- Error messages or unexpected behavior
- Minimal code example to reproduce

## License

This project is provided as-is for educational and research purposes.

## References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Hull, J. C. (2017). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Press, W. H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.

## Acknowledgments

This project demonstrates the mathematical relationship between options and function approximation, showing how financial derivatives can be used as basis functions for general mathematical problems.
