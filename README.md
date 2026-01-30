# Options Function Approximator

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Lean 4](https://img.shields.io/badge/Lean-4-purple.svg)](https://leanprover.github.io/)

A comprehensive Python library and GUI application for approximating arbitrary functions using combinations of financial options. The system decomposes complex payoffs into vanilla options with full Black-Scholes pricing and risk metrics (Greeks).

**Includes Lean 4 formal proofs** of the underlying mathematical theory and a complete LaTeX manuscript detailing the mathematical foundations connecting options, splines, and neural networks.

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

- Python 3.7 or higher
- NumPy (≥2.0)
- Matplotlib (≥3.0)
- SciPy (≥1.0, optional for advanced optimization)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/HenryR2112/option_fitter.git
cd option_fitter

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from options_func_maker import OptionsFunctionApproximator; print('✓ Installation successful')"
```

### Optional: Lean 4 Proofs

To build and verify the formal proofs:

```bash
# Install Lean 4 (macOS/Linux)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build the proofs
cd proof
lake build
```

See [proof/README.md](proof/README.md) for detailed instructions.

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
- **Real-time Visualization**: Interactive matplotlib charts with zoom/pan capabilities
- **Parameter Controls**: Adjust price range, number of options, and Black-Scholes parameters (S₀, r, T, σ)
- **Results Dashboard**:
  - **Summary**: MSE, RMSE, total cost, portfolio composition
  - **Options Breakdown**: Detailed tables of call and put positions with strikes and weights
  - **Portfolio Greeks**: Delta, Gamma, Vega, and Theta with interpretations
  - **Details**: Complete text output with cost breakdown
- **Export Capabilities**: Save results to CSV format (Ctrl+E)
- **Professional Dark Theme**: Modern, eye-friendly interface

### Keyboard Shortcuts

- `Ctrl+R` or `⌘R`: Calculate approximation
- `Ctrl+E` or `⌘E`: Export results to CSV

For detailed GUI documentation, see [README_GUI.md](README_GUI.md).

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
├── options_func_maker.py      # Core approximation library
├── options_gui.py             # Professional GUI application
├── README.md                  # Main documentation (this file)
├── README_GUI.md              # Detailed GUI documentation
├── CHANGES.md                 # Version history and changelog
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── experiments/               # Numerical convergence experiments
│   ├── run_experiments.py     # Experiment runner
│   ├── results.csv            # Convergence data
│   ├── rmse_vs_n.png          # Visualization
│   └── README.md              # Experiments documentation
│
├── proof/                     # Lean 4 formal verification
│   ├── Proof/
│   │   └── Basic.lean         # Formalized theorems and proofs
│   ├── Proof.lean             # Root module
│   ├── lakefile.toml          # Lake build configuration
│   ├── lean-toolchain         # Lean version pinning
│   └── README.md              # Proof system documentation
│
└── tex/                       # LaTeX manuscript
    └── draft.tex              # Academic paper connecting finance, splines, and ML
```

## Experiments and Validation

The `experiments/` directory contains numerical experiments demonstrating convergence properties.

```bash
cd experiments
python run_experiments.py
```

This generates:
- `results.csv`: Convergence data for different numbers of options
- `rmse_vs_n.png`: Visualization of approximation error vs. number of basis functions

See [experiments/README.md](experiments/README.md) for details.

## Academic Paper

The `tex/` directory contains a complete LaTeX manuscript:

**"Options as Basis Functions: A Unifying Perspective Connecting Finance, Splines, and Neural Networks"**

The paper provides:
- Unified mathematical framework connecting financial options, linear B-splines, and ReLU activations
- Constructive density proofs with error bounds
- Formal verification details
- Interdisciplinary connections and applications

To compile:
```bash
cd tex
pdflatex draft.tex
```

## Citation

If you use this software or build upon the ideas in your research, please cite:

```bibtex
@misc{ramstad2026options,
  author = {Ramstad, Henry James},
  title = {Options as Basis Functions: A Unifying Perspective},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HenryR2112/option_fitter}}
}
```

## Troubleshooting

### Common Issues

**Import Error**: Ensure you've activated the virtual environment and installed dependencies:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**GUI Won't Launch**: Check that tkinter is installed (usually included with Python):
```bash
python -m tkinter  # Should open a test window
```

**Lean Build Fails**: Ensure Lean 4 is properly installed and lake is in your PATH:
```bash
lake --version
```

**High Memory Usage**: Reduce `n_points` in approximation calls or use fewer basis functions.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** for Python code style
3. **Add tests** for new functionality
4. **Update documentation** for any changes
5. **Submit a pull request** with a clear description

### Areas for Contribution

- Additional basis functions (wavelets, Fourier terms, etc.)
- Performance optimizations
- Additional numerical experiments
- Completing Lean 4 proof gaps (replacing `sorry` statements)
- Examples and tutorials
- Bug fixes and improvements

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Document public functions with docstrings (Google style preferred)
- Keep functions focused and modular

### Extending the Lean Proofs

1. Install Lean 4 and VS Code with Lean extension
2. Open `proof/Proof/Basic.lean`
3. Replace `sorry` placeholders with proofs using Mathlib lemmas
4. Run `lake build` to verify correctness
5. Submit a PR with your completed proofs

## References

### Financial Mathematics
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Breeden, D. T., & Litzenberger, R. H. (1978). Prices of state-contingent claims implicit in option prices. *Journal of Business*, 51(4), 621-651.
- Carr, P., & Madan, D. (2001). Towards a theory of volatility trading. *Option Pricing, Interest Rates and Risk Management*, 458-476.
- Hull, J. C. (2017). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.

### Approximation Theory
- de Boor, C. (2001). *A Practical Guide to Splines*. Springer.
- Schumaker, L. L. (2007). *Spline Functions: Basic Theory* (3rd ed.). Cambridge University Press.

### Machine Learning
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303-314.
- Leshno, M., Lin, V. Y., Pinkus, A., & Schocken, S. (1993). Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. *Neural Networks*, 6(6), 861-867.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359-366.

### Formal Verification
- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib Documentation](https://leanprover-community.github.io/mathlib4_docs/)

## Related Projects

- **Neural Network Libraries**: TensorFlow, PyTorch (ReLU activations)
- **Spline Libraries**: scipy.interpolate, FITPACK
- **Options Pricing**: QuantLib, PyQL
- **Formal Verification**: Lean 4, Coq, Isabelle/HOL

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Henry James Ramstad

This project is provided for educational and research purposes.

## Acknowledgments

- Built with NumPy, Matplotlib, and SciPy
- Formal proofs use Lean 4 and Mathlib
- Inspired by connections between finance, approximation theory, and machine learning

## Contact and Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/HenryR2112/option_fitter/issues)
- **Discussions**: Join discussions in [GitHub Discussions](https://github.com/HenryR2112/option_fitter/discussions)

---

**Made with mathematical rigor and practical utility in mind.**
