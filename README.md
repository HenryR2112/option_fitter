# Options Function Approximator

A comprehensive Python library and GUI application for approximating arbitrary functions using combinations of financial options. The system decomposes complex payoffs into vanilla options with full Black-Scholes pricing and risk metrics (Greeks).

**Includes Lean 4 formal proofs** of the underlying mathematical theory.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [GUI Application](#gui-application)
- [Core Library Usage](#core-library-usage)
- [Mathematical Foundation](#mathematical-foundation)
- [Lean 4 Formal Proofs](#lean-4-formal-proofs)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [References](#references)

## Overview

The Options Function Approximator uses options (calls and puts) as basis functions to approximate arbitrary mathematical functions. This approach leverages the density theorem: **the span of {1, x, (x-K)₊ : K ∈ [a,b]} is dense in C([a,b])**.

### Use Cases

- **Structured Product Decomposition**: Break down complex financial products into vanilla options
- **Payoff Replication**: Create portfolios that replicate target payoff functions
- **Risk Analysis**: Understand the Greeks and cost structure of complex payoffs
- **Education**: Explore the relationship between options and function approximation

## Features

- **Multiple Basis Functions**: Calls, puts, stock positions, Gaussians, sigmoids, polynomials
- **Black-Scholes Pricing**: Full European option pricing with customizable parameters
- **Portfolio Greeks**: Delta, Gamma, Vega, and Theta calculations
- **Cost Analysis**: Detailed breakdown of replication costs
- **Professional GUI**: Modern dark-themed interface with interactive charts
- **Export Capabilities**: Save results to CSV
- **Formal Proofs**: Lean 4 formalization of the mathematical foundations

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy (optional)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd option_fitter

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib scipy
```

### Verify Installation

```bash
python -c "from options_func_maker import OptionsFunctionApproximator; print('OK')"
```

## Quick Start

```python
from options_func_maker import OptionsFunctionApproximator
import numpy as np

# Create approximator
approx = OptionsFunctionApproximator(
    n_options=15,
    price_range=(0, 2*np.pi),
    use_calls=True,
    use_puts=True,
    S0=100.0,
    r=0.05,
    T=0.25,
    sigma=0.2
)

# Approximate sin(x)
weights, mse = approx.approximate(np.sin, n_points=1000)
print(f"RMSE: {np.sqrt(mse):.6f}")

# Get cost and Greeks
print(f"Total Cost: ${approx.get_total_cost():,.2f}")
greeks = approx.calculate_portfolio_greeks()
print(f"Delta: {greeks['delta']:.4f}")
```

## GUI Application

Launch the professional GUI interface:

```bash
python options_gui.py
```

### GUI Features

- **Function Selection**: Choose from presets (sin, cos, sigmoid, Gaussian, bull spread, butterfly) or define custom functions
- **Real-time Visualization**: Interactive matplotlib charts with zoom/pan
- **Parameter Controls**: Adjust price range, number of options, and Black-Scholes parameters
- **Results Tabs**:
  - **Summary**: MSE, RMSE, total cost, portfolio composition
  - **Options Breakdown**: Detailed tables of call and put positions
  - **Portfolio Greeks**: Delta, Gamma, Vega, Theta
  - **Details**: Full text output
- **Export**: Save results to CSV (Ctrl+E)

### Keyboard Shortcuts

- `Ctrl+R`: Calculate approximation
- `Ctrl+E`: Export results

## Core Library Usage

### Basic Approximation

```python
from options_func_maker import OptionsFunctionApproximator
import numpy as np

# Define target function
def target(x):
    return np.sin(x) + 0.5 * np.cos(2*x)

# Create approximator
approx = OptionsFunctionApproximator(
    n_options=20,
    price_range=(0, 10),
    use_calls=True,
    use_puts=True,
    use_stock=True
)

# Find optimal weights
weights, mse = approx.approximate(
    target,
    n_points=1000,
    regularization=0.001
)

# Evaluate at new points
x = np.linspace(0, 10, 100)
y_approx = approx.evaluate(x)

# Plot comparison
approx.plot_approximation(target, title="Function Approximation")
```

### Complex Payoff Decomposition

```python
def butterfly_spread(x):
    """Butterfly spread centered at 100."""
    return (np.maximum(x - 95, 0)
            - 2 * np.maximum(x - 100, 0)
            + np.maximum(x - 105, 0))

approx = OptionsFunctionApproximator(
    n_options=15,
    price_range=(85, 115),
    S0=100.0,
    r=0.05,
    T=0.25,
    sigma=0.2
)

weights, mse = approx.approximate(butterfly_spread)
approx.print_cost_breakdown()
```

### Using Additional Basis Functions

```python
approx = OptionsFunctionApproximator(
    n_options=10,
    price_range=(0, 10),
    use_calls=True,
    use_puts=True,
    use_stock=True,
    # Statistical functions
    use_gaussians=True,
    n_gaussians=5,
    use_sigmoids=True,
    n_sigmoids=5,
    # Polynomial terms
    use_polynomials=True,
    max_polynomial_power=3
)
```

## Mathematical Foundation

### Density Theorem

The core mathematical result is that **call options form a dense basis** for continuous functions. Given any continuous function f on [a,b] and ε > 0, there exist constants α, β and weights wᵢ such that:

```
|f(x) - (α + βx + Σᵢ wᵢ(x - Kᵢ)₊)| < ε  for all x ∈ [a,b]
```

where (x - K)₊ = max(x - K, 0) is the call option payoff.

### Approximation Method

The system solves a least-squares optimization problem:

```
minimize  Σⱼ |f(xⱼ) - Σᵢ wᵢ φᵢ(xⱼ)|² + λ||w||²
```

where:
- φᵢ(x) are basis functions (calls, puts, stock)
- λ is the regularization parameter
- xⱼ are sample points

### Black-Scholes Pricing

Call and put options are priced using the Black-Scholes formula:

```
C = S₀ N(d₁) - K e^(-rT) N(d₂)
P = K e^(-rT) N(-d₂) - S₀ N(-d₁)

where d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
      d₂ = d₁ - σ√T
```

### Portfolio Greeks

| Greek | Definition | Interpretation |
|-------|-----------|----------------|
| Delta (Δ) | ∂V/∂S | $ change per $1 stock move |
| Gamma (Γ) | ∂²V/∂S² | Delta change per $1 stock move |
| Vega (ν) | ∂V/∂σ | $ change per 1% vol change |
| Theta (Θ) | ∂V/∂t | $ change per day |

## Lean 4 Formal Proofs

The `proof/` directory contains Lean 4 formalizations of the mathematical foundations.

### Building the Proofs

```bash
# Install Lean 4 (if not already installed)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build proofs
cd proof
lake build
```

### What's Proven

The Lean formalization in `proof/Proof/Basic.lean` includes:

**Fully Proven:**
- `hinge_continuous`: The hinge function (x - K)₊ is continuous
- `hinge_lipschitz`: Hinge is Lipschitz with constant 1
- `hinge_nonneg`: Hinge is non-negative
- `option_portfolio_continuous`: Option portfolios are continuous
- `hinge_bounded_on_interval`: Hinge is bounded on compact intervals

**Theorem Statements (with `sorry`):**
- `density_of_call_spans_statement`: Main density theorem
- `integral_representation_C2_statement`: Integral representation for C² functions

See [proof/README.md](proof/README.md) for detailed documentation.

## API Reference

### OptionsFunctionApproximator

#### Constructor

```python
OptionsFunctionApproximator(
    n_options: int = 10,
    price_range: tuple = (0, 10),
    use_calls: bool = True,
    use_puts: bool = True,
    use_stock: bool = True,
    S0: float = 100.0,
    r: float = 0.05,
    T: float = 0.25,
    sigma: float = 0.2,
    # Optional basis functions
    use_gaussians: bool = False,
    use_sigmoids: bool = False,
    use_polynomials: bool = False,
    ...
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `approximate(func, n_points, regularization)` | Find optimal weights |
| `evaluate(stock_prices)` | Evaluate approximation |
| `calculate_premiums()` | Get Black-Scholes premiums |
| `calculate_portfolio_greeks()` | Get portfolio Greeks |
| `get_total_cost()` | Get total portfolio cost |
| `print_cost_breakdown()` | Print detailed breakdown |
| `plot_approximation(func)` | Plot comparison |

### Standalone Functions

```python
# Option payoffs
call_payoff(stock_price, strike)
put_payoff(stock_price, strike)

# Black-Scholes pricing
black_scholes_call(S, K, T, r, sigma)
black_scholes_put(S, K, T, r, sigma)

# Greeks
black_scholes_delta_call(S, K, T, r, sigma)
black_scholes_gamma(S, K, T, r, sigma)
black_scholes_vega(S, K, T, r, sigma)
black_scholes_theta_call(S, K, T, r, sigma)

# Basis functions
gaussian_pdf(x, mu, sigma)
sigmoid(x, center, scale)
cumulative_normal(x)
```

## Examples

### Example 1: Sine Wave Approximation

```python
approx = OptionsFunctionApproximator(
    n_options=15,
    price_range=(0, 2*np.pi)
)
weights, mse = approx.approximate(np.sin)
print(f"RMSE: {np.sqrt(mse):.6f}")
```

### Example 2: Bull Call Spread

```python
def bull_spread(x):
    return np.maximum(x - 90, 0) - np.maximum(x - 110, 0)

approx = OptionsFunctionApproximator(
    n_options=10,
    price_range=(80, 120),
    S0=100.0
)
weights, mse = approx.approximate(bull_spread)
approx.print_cost_breakdown()
```

### Example 3: Custom Payoff with Gaussian Component

```python
def custom_payoff(x):
    call_spread = np.maximum(x - 95, 0) - np.maximum(x - 105, 0)
    gaussian_bump = 3.0 * np.exp(-((x - 100)**2) / 50)
    return call_spread + gaussian_bump

approx = OptionsFunctionApproximator(
    n_options=20,
    price_range=(85, 115),
    use_gaussians=True,
    n_gaussians=5
)
weights, mse = approx.approximate(custom_payoff)
```

## Project Structure

```
option_fitter/
├── options_func_maker.py   # Core library
├── options_gui.py          # GUI application
├── README.md               # This file
├── LEAN_PROOFS_README.md   # Lean proofs overview
│
└── proof/                  # Lean 4 formalization
    ├── Proof/
    │   └── Basic.lean      # Main proofs
    ├── Proof.lean          # Root import
    ├── lakefile.toml       # Build config
    ├── lean-toolchain      # Lean version
    └── README.md           # Proof documentation
```

## Contributing

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Document public functions with docstrings

### Adding New Basis Functions

1. Implement function in `options_func_maker.py`
2. Add constructor parameters
3. Update basis function creation in `__init__`
4. Update documentation

### Extending the Lean Proofs

1. Open `proof/Proof/Basic.lean` in VS Code with Lean 4 extension
2. Replace `sorry` placeholders with actual proofs
3. Run `lake build` to verify

## References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Hull, J. C. (2017). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Carr, P., & Madan, D. (2001). Optimal positioning in derivative securities. *Quantitative Finance*, 1(1), 19-37.

## License

This project is provided for educational and research purposes.
