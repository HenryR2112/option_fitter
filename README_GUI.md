# Options Function Approximator GUI

A lightweight desktop application for decomposing complex financial functions into vanilla options with full pricing and risk metrics.

## Features

- **Function Selection**: Choose from predefined functions (sin, cos, sigmoid, Gaussian, polynomial, bull spread, complex payoff) or define your own custom function
- **Interactive Visualization**: Real-time plot showing target function vs. options approximation
- **Cost Breakdown**: Detailed pricing breakdown showing:
  - Individual option costs (calls and puts)
  - Stock position costs
  - Total portfolio cost
- **Portfolio Greeks**: Risk sensitivities (Delta, Gamma, Vega, Theta)
- **Cost Efficiency Metrics**: Analysis of portfolio composition

## Requirements

- Python 3.7+
- numpy
- matplotlib
- tkinter (usually included with Python)

## Usage

### Launch the GUI

```bash
python options_gui.py
```

### Using the Application

1. **Select a Function**:
   - Choose from predefined functions (sin, cos, sigmoid, etc.)
   - Or enter a custom Python function using numpy (e.g., `np.sin(x) + 0.5*x`)

2. **Set Price Range**:
   - Enter minimum and maximum stock prices
   - Set the number of options to use for approximation

3. **Configure Pricing Parameters**:
   - Current Stock Price (S0)
   - Risk-free Rate (r)
   - Time to Expiration (T, in years)
   - Volatility (σ)

4. **Select Basis Functions**:
   - Choose whether to use call options, put options, and/or stock positions

5. **Calculate**:
   - Click "Calculate Approximation" to generate the decomposition
   - View the plot showing target vs. approximation
   - Review the detailed cost breakdown and portfolio Greeks in the results panel

## Example Functions

- **sin(x)**: Trigonometric sine function
- **cos(x)**: Trigonometric cosine function
- **Sigmoid**: Smooth S-shaped transition function
- **Gaussian PDF**: Normal probability density function
- **x³ - 2x² + x**: Polynomial function
- **Bull Spread**: Option strategy payoff
- **Complex Payoff**: Multi-component financial payoff

## Custom Functions

Enter Python code using numpy. The variable `x` represents the stock price array.

Examples:
- `np.sin(x) + 0.5*x`
- `np.maximum(x - 100, 0)`
- `x**2 - 5*x + 10`
- `np.exp(-x/10)`

## Output

The application displays:

1. **Plot**: Visual comparison of target function and approximation
2. **Approximation Quality**: MSE and RMSE error metrics
3. **Cost Breakdown**: 
   - Individual option costs with strikes, weights, and premiums
   - Stock position cost
   - Total portfolio cost
4. **Portfolio Greeks**:
   - Delta: Price sensitivity
   - Gamma: Delta sensitivity
   - Vega: Volatility sensitivity
   - Theta: Time decay
5. **Cost Efficiency Metrics**: Portfolio composition analysis

## Notes

- The application uses Black-Scholes pricing for European options
- All costs are calculated based on absolute weights (long positions)
- The approximation uses least squares optimization with L2 regularization
- Non-option basis functions (Gaussians, sigmoids, etc.) contribute to approximation but have no direct cost
