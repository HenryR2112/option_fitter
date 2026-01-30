"""
Options-based Function Approximator

This script uses call and put options at different strike prices to approximate
arbitrary functions. The idea is that options act as basis functions, and by
combining them with appropriate weights, we can approximate any function.

Example: Given sin(x), the script finds a combination of n options that best
approximates sin(x) over a specified range.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
import warnings

try:
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def call_payoff(stock_price: np.ndarray, strike: float) -> np.ndarray:
    """
    Call option payoff: max(S - K, 0)

    Args:
        stock_price: Array of stock prices
        strike: Strike price K

    Returns:
        Payoff array
    """
    return np.maximum(stock_price - strike, 0.0)


def put_payoff(stock_price: np.ndarray, strike: float) -> np.ndarray:
    """
    Put option payoff: max(K - S, 0)

    Args:
        stock_price: Array of stock prices
        strike: Strike price K

    Returns:
        Payoff array
    """
    return np.maximum(strike - stock_price, 0.0)


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Gaussian (normal) probability density function.
    Used for modeling probability distributions and volatility.

    Args:
        x: Input values
        mu: Mean
        sigma: Standard deviation

    Returns:
        PDF values
    """
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sigmoid(x: np.ndarray, center: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """
    Sigmoid (logistic) function: 1 / (1 + exp(-scale * (x - center)))
    Used for smooth transitions, probability modeling, and activation functions.

    Args:
        x: Input values
        center: Center point of sigmoid
        scale: Steepness parameter

    Returns:
        Sigmoid values in [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-scale * (x - center)))


def cumulative_normal(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """
    Cumulative normal distribution (CDF).
    Used in Black-Scholes option pricing and risk calculations.

    Args:
        x: Input values
        mu: Mean
        sigma: Standard deviation

    Returns:
        CDF values in [0, 1]
    """
    # Using error function (erf) for CDF: Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
    # Approximation: erf(x) ≈ tanh(1.128379167 * x) for small x
    # Better: use Abramowitz and Stegun approximation
    z = (x - mu) / (sigma + 1e-10)  # Avoid division by zero

    # Abramowitz and Stegun approximation (accurate to ~7 decimal places)
    # IMPORTANT: Need to divide by sqrt(2) for the erf transformation
    z_erf = z / np.sqrt(2.0)
    sign = np.sign(z_erf)
    z_abs = np.abs(z_erf)

    # Constants for approximation
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911

    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-(z_abs**2))

    return 0.5 * (1.0 + sign * y)


def exponential_decay(x: np.ndarray, center: float, decay_rate: float) -> np.ndarray:
    """
    Exponential decay function: exp(-decay_rate * |x - center|)
    Used for discounting, time decay, and volatility modeling.

    Args:
        x: Input values
        center: Center point
        decay_rate: Decay rate (positive)

    Returns:
        Decay values
    """
    return np.exp(-decay_rate * np.abs(x - center))


def polynomial_term(x: np.ndarray, power: int) -> np.ndarray:
    """
    Polynomial term: x^power
    Used for modeling nonlinear relationships.

    Args:
        x: Input values
        power: Power of polynomial

    Returns:
        x raised to power
    """
    return np.power(x, power)


def log_normal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Log-normal function: exp(mu + sigma * log(x)) for x > 0
    Used for modeling asset prices and returns.

    Args:
        x: Input values (should be positive)
        mu: Mean parameter
        sigma: Volatility parameter

    Returns:
        Log-normal values
    """
    x_safe = np.maximum(x, 1e-10)  # Avoid log(0)
    return np.exp(mu + sigma * np.log(x_safe))


def black_scholes_delta_like(
    x: np.ndarray, strike: float, volatility: float = 0.2
) -> np.ndarray:
    """
    Black-Scholes delta-like function (normal CDF approximation).
    Represents option sensitivity to underlying price.

    Args:
        x: Stock price
        strike: Strike price
        volatility: Volatility parameter

    Returns:
        Delta-like values
    """
    # Avoid division by zero or very small strikes
    strike_safe = np.maximum(strike, 1e-6)
    x_safe = np.maximum(x, 1e-6)

    # Calculate d1 for Black-Scholes
    d1 = (np.log(x_safe / strike_safe) + 0.5 * volatility**2) / (volatility + 1e-10)
    return cumulative_normal(d1)


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black-Scholes formula for European call option price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility (annualized)

    Returns:
        Call option premium
    """
    if T <= 0:
        return max(S - K, 0.0)
    if K <= 0:
        return max(S - K * np.exp(-r * T), 0.0)  # Deep ITM call

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Use cumulative_normal for N(d1) and N(d2)
    N_d1 = cumulative_normal(np.array([d1]))[0]
    N_d2 = cumulative_normal(np.array([d2]))[0]

    call_price = S * N_d1 - K * np.exp(-r * T) * N_d2
    return max(call_price, 0.0)


def black_scholes_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black-Scholes formula for European put option price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility (annualized)

    Returns:
        Put option premium
    """
    if T <= 0:
        return max(K - S, 0.0)
    if K <= 0:
        return 0.0  # Put with K=0 is worthless

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Use cumulative_normal for N(-d1) and N(-d2)
    N_neg_d1 = cumulative_normal(np.array([-d1]))[0]
    N_neg_d2 = cumulative_normal(np.array([-d2]))[0]

    put_price = K * np.exp(-r * T) * N_neg_d2 - S * N_neg_d1
    return max(put_price, 0.0)


def volatility_smile(
    x: np.ndarray, center: float, vol_atm: float, vol_skew: float
) -> np.ndarray:
    """
    Volatility smile function: vol_atm + vol_skew * (x - center)^2
    Models implied volatility as a function of strike price.

    Args:
        x: Strike prices
        center: At-the-money strike
        vol_atm: ATM volatility
        vol_skew: Volatility skew parameter

    Returns:
        Volatility values
    """
    return vol_atm + vol_skew * (x - center) ** 2


def black_scholes_delta_call(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Calculate delta (price sensitivity) for a call option."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    if K <= 0:
        return 1.0  # Deep ITM call has delta ~1
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return cumulative_normal(np.array([d1]))[0]


def black_scholes_delta_put(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Calculate delta (price sensitivity) for a put option."""
    if T <= 0:
        return -1.0 if S < K else 0.0
    if K <= 0:
        return 0.0  # Put with K=0 has delta 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return cumulative_normal(np.array([d1]))[0] - 1.0


def black_scholes_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate gamma (delta sensitivity) for an option."""
    if T <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    # Gamma is the same for calls and puts
    return gaussian_pdf(np.array([d1]), 0.0, 1.0)[0] / (S * sigma * np.sqrt(T))


def black_scholes_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate vega (volatility sensitivity) for an option.

    Returns the change in option value per 0.01 (1 percentage point) change in volatility.
    E.g., if vega = 15.3 and volatility changes from 20% to 21% (Δσ = 0.01),
    the option value changes by approximately $15.30.
    """
    if T <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    # Vega is the same for calls and puts
    # Standard vega formula: S * φ(d1) * √T
    # This gives change per unit change in σ (i.e., per 0.01 when σ is in decimal form)
    return S * gaussian_pdf(np.array([d1]), 0.0, 1.0)[0] * np.sqrt(T)


def black_scholes_theta_call(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Calculate theta (time decay) for a call option."""
    if T <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = cumulative_normal(np.array([d1]))[0]
    N_d2 = cumulative_normal(np.array([d2]))[0]
    pdf_d1 = gaussian_pdf(np.array([d1]), 0.0, 1.0)[0]

    theta = (
        -S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2
    ) / 365.0  # Per day
    return theta


def black_scholes_theta_put(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Calculate theta (time decay) for a put option."""
    if T <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_neg_d2 = cumulative_normal(np.array([-d2]))[0]
    pdf_d1 = gaussian_pdf(np.array([d1]), 0.0, 1.0)[0]

    theta = (
        -S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N_neg_d2
    ) / 365.0  # Per day
    return theta


class OptionsFunctionApproximator:
    """
    Approximates arbitrary functions using combinations of call and put options.

    The approximator uses n options (calls and puts) at different strike prices
    and finds optimal weights to minimize the approximation error.
    """

    def __init__(
        self,
        n_options: int = 10,
        price_range: Tuple[float, float] = (0, 10),
        use_calls: bool = True,
        use_puts: bool = True,
        use_stock: bool = True,
        use_gaussians: bool = False,
        n_gaussians: int = 5,
        use_sigmoids: bool = False,
        n_sigmoids: int = 5,
        use_cumulative_normal: bool = False,
        n_cumulative_normal: int = 5,
        use_exponential_decay: bool = False,
        n_exponential_decay: int = 5,
        use_polynomials: bool = False,
        max_polynomial_power: int = 3,
        use_log_normal: bool = False,
        n_log_normal: int = 3,
        use_black_scholes_delta: bool = False,
        n_black_scholes_delta: int = 5,
        use_volatility_smile: bool = False,
        n_volatility_smile: int = 3,
        # Black-Scholes pricing parameters
        S0: float = 100.0,  # Current stock price
        r: float = 0.05,  # Risk-free interest rate
        T: float = 0.25,  # Time to expiration (years)
        sigma: float = 0.2,  # Volatility
    ):
        """
        Initialize the approximator with various basis functions.

        Args:
            n_options: Number of options to use for approximation
            price_range: (min_price, max_price) range for stock prices
            use_calls: Whether to include call options
            use_puts: Whether to include put options
            use_stock: Whether to include stock position (linear term)
            use_gaussians: Whether to include Gaussian PDFs
            n_gaussians: Number of Gaussian basis functions
            use_sigmoids: Whether to include sigmoid functions
            n_sigmoids: Number of sigmoid basis functions
            use_cumulative_normal: Whether to include cumulative normal CDFs
            n_cumulative_normal: Number of cumulative normal basis functions
            use_exponential_decay: Whether to include exponential decay functions
            n_exponential_decay: Number of exponential decay basis functions
            use_polynomials: Whether to include polynomial terms
            max_polynomial_power: Maximum power for polynomial terms (e.g., 3 -> x, x^2, x^3)
            use_log_normal: Whether to include log-normal functions
            n_log_normal: Number of log-normal basis functions
            use_black_scholes_delta: Whether to include Black-Scholes delta-like functions
            n_black_scholes_delta: Number of Black-Scholes delta basis functions
            use_volatility_smile: Whether to include volatility smile functions
            n_volatility_smile: Number of volatility smile basis functions
            S0: Current stock price for Black-Scholes pricing
            r: Risk-free interest rate (annualized)
            T: Time to expiration in years
            sigma: Volatility (annualized)
        """
        self.n_options = n_options
        self.price_range = price_range
        self.use_calls = use_calls
        self.use_puts = use_puts
        self.use_stock = use_stock

        # Black-Scholes pricing parameters
        self.S0 = S0
        self.r = r
        self.T = T
        self.sigma = sigma

        # Determine strike prices (evenly spaced across range)
        self.strikes = np.linspace(price_range[0], price_range[1], n_options)

        # Center of price range for basis functions
        self.center = (price_range[0] + price_range[1]) / 2
        self.range_width = price_range[1] - price_range[0]

        # Build basis functions list
        self.basis_functions = []
        self.basis_names = []
        # Track option metadata: (type, strike) for each basis function
        # type: 'call', 'put', 'stock', or None for non-option basis functions
        self.option_metadata = []

        # Stock position (linear term)
        if use_stock:
            self.basis_functions.append(lambda S: S)
            self.basis_names.append("Stock")
            self.option_metadata.append(("stock", None))

        # Call options
        if use_calls:
            for strike in self.strikes:
                self.basis_functions.append(lambda S, K=strike: call_payoff(S, K))
                self.basis_names.append(f"Call(K={strike:.2f})")
                self.option_metadata.append(("call", strike))

        # Put options
        if use_puts:
            for strike in self.strikes:
                self.basis_functions.append(lambda S, K=strike: put_payoff(S, K))
                self.basis_names.append(f"Put(K={strike:.2f})")
                self.option_metadata.append(("put", strike))

        # Gaussian PDFs
        if use_gaussians:
            gaussian_centers = np.linspace(price_range[0], price_range[1], n_gaussians)
            gaussian_sigmas = np.linspace(
                self.range_width / 10, self.range_width / 2, n_gaussians
            )
            for mu, sigma in zip(gaussian_centers, gaussian_sigmas):
                self.basis_functions.append(
                    lambda S, m=mu, s=sigma: gaussian_pdf(S, m, s)
                )
                self.basis_names.append(f"Gaussian(mu={mu:.2f}, sigma={sigma:.2f})")
                self.option_metadata.append((None, None))

        # Sigmoid functions
        if use_sigmoids:
            sigmoid_centers = np.linspace(price_range[0], price_range[1], n_sigmoids)
            sigmoid_scales = np.linspace(0.5, 5.0, n_sigmoids)
            for center, scale in zip(sigmoid_centers, sigmoid_scales):
                self.basis_functions.append(
                    lambda S, c=center, s=scale: sigmoid(S, c, s)
                )
                self.basis_names.append(f"Sigmoid(c={center:.2f}, scale={scale:.2f})")
                self.option_metadata.append((None, None))

        # Cumulative normal (CDF)
        if use_cumulative_normal:
            cdf_centers = np.linspace(
                price_range[0], price_range[1], n_cumulative_normal
            )
            cdf_sigmas = np.linspace(
                self.range_width / 10, self.range_width / 3, n_cumulative_normal
            )
            for mu, sigma in zip(cdf_centers, cdf_sigmas):
                self.basis_functions.append(
                    lambda S, m=mu, s=sigma: cumulative_normal(S, m, s)
                )
                self.basis_names.append(f"CumNorm(mu={mu:.2f}, sigma={sigma:.2f})")
                self.option_metadata.append((None, None))

        # Exponential decay
        if use_exponential_decay:
            decay_centers = np.linspace(
                price_range[0], price_range[1], n_exponential_decay
            )
            decay_rates = np.linspace(0.1, 2.0, n_exponential_decay)
            for center, rate in zip(decay_centers, decay_rates):
                self.basis_functions.append(
                    lambda S, c=center, r=rate: exponential_decay(S, c, r)
                )
                self.basis_names.append(f"ExpDecay(c={center:.2f}, rate={rate:.2f})")
                self.option_metadata.append((None, None))

        # Polynomial terms
        if use_polynomials:
            for power in range(1, max_polynomial_power + 1):
                self.basis_functions.append(lambda S, p=power: polynomial_term(S, p))
                self.basis_names.append(f"x^{power}")
                self.option_metadata.append((None, None))

        # Log-normal functions
        if use_log_normal:
            log_mus = np.linspace(-1, 1, n_log_normal)
            log_sigmas = np.linspace(0.1, 0.5, n_log_normal)
            for mu, sigma in zip(log_mus, log_sigmas):
                self.basis_functions.append(
                    lambda S, m=mu, s=sigma: log_normal(S, m, s)
                )
                self.basis_names.append(f"LogNormal(mu={mu:.2f}, sigma={sigma:.2f})")
                self.option_metadata.append((None, None))

        # Black-Scholes delta-like functions
        if use_black_scholes_delta:
            delta_strikes = np.linspace(
                price_range[0], price_range[1], n_black_scholes_delta
            )
            delta_vols = np.linspace(0.1, 0.5, n_black_scholes_delta)
            for strike, vol in zip(delta_strikes, delta_vols):
                self.basis_functions.append(
                    lambda S, K=strike, v=vol: black_scholes_delta_like(S, K, v)
                )
                self.basis_names.append(f"BS_Delta(K={strike:.2f}, vol={vol:.2f})")
                self.option_metadata.append((None, None))

        # Volatility smile functions
        if use_volatility_smile:
            smile_centers = np.linspace(
                price_range[0], price_range[1], n_volatility_smile
            )
            smile_atm_vols = np.linspace(0.15, 0.35, n_volatility_smile)
            smile_skews = np.linspace(-0.1, 0.1, n_volatility_smile)
            for center, atm_vol, skew in zip(
                smile_centers, smile_atm_vols, smile_skews
            ):
                self.basis_functions.append(
                    lambda S, c=center, v=atm_vol, s=skew: volatility_smile(S, c, v, s)
                )
                self.basis_names.append(
                    f"VolSmile(c={center:.2f}, ATM={atm_vol:.2f}, skew={skew:.2f})"
                )
                self.option_metadata.append((None, None))

        self.n_basis = len(self.basis_functions)
        self.weights = None

    def evaluate_basis(self, stock_prices: np.ndarray) -> np.ndarray:
        """
        Evaluate all basis functions at given stock prices.

        Args:
            stock_prices: Array of stock prices

        Returns:
            Matrix of shape (n_prices, n_basis) with basis function values
        """
        basis_matrix = np.zeros((len(stock_prices), self.n_basis))
        for i, basis_func in enumerate(self.basis_functions):
            basis_matrix[:, i] = basis_func(stock_prices)
        return basis_matrix

    def approximate(
        self,
        target_function: Callable,
        n_points: int = 1000,
        regularization: float = 0.0,
        method: str = "least_squares",
    ) -> Tuple[np.ndarray, float]:
        """
        Find optimal weights to approximate the target function.

        Args:
            target_function: Function f(S) to approximate
            n_points: Number of sample points for optimization
            regularization: L2 regularization strength
            method: Optimization method ("least_squares" or "minimize")

        Returns:
            Tuple of (weights, mse_error)
        """
        # Sample points across price range
        stock_prices = np.linspace(self.price_range[0], self.price_range[1], n_points)
        target_values = target_function(stock_prices)

        # Evaluate basis functions
        basis_matrix = self.evaluate_basis(stock_prices)

        if method == "least_squares":
            # Solve: min ||basis_matrix @ weights - target_values||^2
            # Using normal equations with regularization
            A = basis_matrix.T @ basis_matrix
            b = basis_matrix.T @ target_values

            # Add regularization
            if regularization > 0:
                A += regularization * np.eye(self.n_basis)

            # Check matrix conditioning for numerical stability
            cond_number = np.linalg.cond(A)
            if cond_number > 1e12:
                warnings.warn(
                    f"Matrix is ill-conditioned (condition number: {cond_number:.2e}). "
                    f"Results may be numerically unstable. Consider:\n"
                    f"  1. Increasing regularization (current: {regularization})\n"
                    f"  2. Reducing number of options\n"
                    f"  3. Checking for redundant basis functions",
                    UserWarning
                )
            elif cond_number > 1e9:
                warnings.warn(
                    f"Matrix conditioning is marginal (condition number: {cond_number:.2e}). "
                    f"Consider increasing regularization.",
                    UserWarning
                )

            self.weights = np.linalg.solve(A, b)

        elif method == "minimize":
            if not HAS_SCIPY:
                raise ImportError(
                    "scipy is required for 'minimize' method. Install with: pip install scipy"
                )

            # Use scipy minimize for more flexibility
            def objective(weights):
                approximation = basis_matrix @ weights
                error = approximation - target_values
                mse = np.mean(error**2)
                reg_term = regularization * np.sum(weights**2)
                return mse + reg_term

            # Initial guess: zeros
            initial_weights = np.zeros(self.n_basis)

            result = minimize(objective, initial_weights, method="L-BFGS-B")
            self.weights = result.x
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate final error
        approximation = basis_matrix @ self.weights
        mse = np.mean((approximation - target_values) ** 2)

        return self.weights, mse

    def evaluate(self, stock_prices: np.ndarray) -> np.ndarray:
        """
        Evaluate the approximated function at given stock prices.

        Args:
            stock_prices: Array of stock prices

        Returns:
            Approximated function values
        """
        if self.weights is None:
            raise ValueError("Must call approximate() first")

        basis_matrix = self.evaluate_basis(stock_prices)
        return basis_matrix @ self.weights

    def calculate_premiums(self) -> Tuple[np.ndarray, dict]:
        """
        Calculate Black-Scholes premiums for all options in the approximation.

        Returns:
            Tuple of (premiums_array, premium_dict)
            - premiums_array: Array of net costs for each basis function
              (positive = cash outflow for long, negative = cash inflow for short)
            - premium_dict: Dictionary with detailed premium information including:
              * Individual option details (calls, puts)
              * net_cost: Total net cash flow (outflow - inflow)
              * gross_long_cost: Total premium paid for long positions
              * gross_short_credit: Total premium received from short positions
              * gross_notional: Sum of absolute values (for capital requirements)
        """
        if self.weights is None:
            raise ValueError("Must call approximate() first")

        premiums = np.zeros(self.n_basis)
        premium_details = {
            "calls": [],
            "puts": [],
            "stock_net_cost": 0.0,
            "non_option_functions": [],
            "gross_long_cost": 0.0,
            "gross_short_credit": 0.0,
            "gross_notional": 0.0,
        }

        for i, (weight, (opt_type, strike), name) in enumerate(
            zip(self.weights, self.option_metadata, self.basis_names)
        ):
            if opt_type == "call":
                premium = black_scholes_call(
                    self.S0, strike, self.T, self.r, self.sigma
                )
                net_cost = weight * premium  # Positive = pay, negative = receive
                gross_cost = abs(weight) * premium
                premiums[i] = net_cost

                if weight > 0:
                    premium_details["gross_long_cost"] += gross_cost
                else:
                    premium_details["gross_short_credit"] += gross_cost

                premium_details["gross_notional"] += gross_cost
                premium_details["calls"].append(
                    {
                        "name": name,
                        "strike": strike,
                        "weight": weight,
                        "premium_per_unit": premium,
                        "net_cost": net_cost,
                        "position": "LONG" if weight > 0 else "SHORT",
                    }
                )
            elif opt_type == "put":
                premium = black_scholes_put(self.S0, strike, self.T, self.r, self.sigma)
                net_cost = weight * premium
                gross_cost = abs(weight) * premium
                premiums[i] = net_cost

                if weight > 0:
                    premium_details["gross_long_cost"] += gross_cost
                else:
                    premium_details["gross_short_credit"] += gross_cost

                premium_details["gross_notional"] += gross_cost
                premium_details["puts"].append(
                    {
                        "name": name,
                        "strike": strike,
                        "weight": weight,
                        "premium_per_unit": premium,
                        "net_cost": net_cost,
                        "position": "LONG" if weight > 0 else "SHORT",
                    }
                )
            elif opt_type == "stock":
                # Stock cost: positive weight = long (pay), negative = short (receive)
                net_cost = weight * self.S0
                premiums[i] = net_cost
                premium_details["stock_net_cost"] += net_cost
                if weight > 0:
                    premium_details["gross_long_cost"] += abs(net_cost)
                else:
                    premium_details["gross_short_credit"] += abs(net_cost)
                premium_details["gross_notional"] += abs(net_cost)
            else:
                # Non-option basis functions have no direct cost
                premium_details["non_option_functions"].append(
                    {"name": name, "weight": weight}
                )

        # Calculate net cost (total cash outflow)
        premium_details["net_cost"] = premiums.sum()

        return premiums, premium_details

    def calculate_portfolio_greeks(self) -> dict:
        """
        Calculate portfolio-level Greeks (delta, gamma, vega, theta).

        Returns:
            Dictionary with portfolio Greeks
        """
        if self.weights is None:
            raise ValueError("Must call approximate() first")

        portfolio_delta = 0.0
        portfolio_gamma = 0.0
        portfolio_vega = 0.0
        portfolio_theta = 0.0

        for i, (weight, (opt_type, strike), name) in enumerate(
            zip(self.weights, self.option_metadata, self.basis_names)
        ):
            if opt_type == "call":
                delta = black_scholes_delta_call(
                    self.S0, strike, self.T, self.r, self.sigma
                )
                gamma = black_scholes_gamma(self.S0, strike, self.T, self.r, self.sigma)
                vega = black_scholes_vega(self.S0, strike, self.T, self.r, self.sigma)
                theta = black_scholes_theta_call(
                    self.S0, strike, self.T, self.r, self.sigma
                )

                portfolio_delta += weight * delta
                portfolio_gamma += weight * gamma
                portfolio_vega += weight * vega
                portfolio_theta += weight * theta

            elif opt_type == "put":
                delta = black_scholes_delta_put(
                    self.S0, strike, self.T, self.r, self.sigma
                )
                gamma = black_scholes_gamma(self.S0, strike, self.T, self.r, self.sigma)
                vega = black_scholes_vega(self.S0, strike, self.T, self.r, self.sigma)
                theta = black_scholes_theta_put(
                    self.S0, strike, self.T, self.r, self.sigma
                )

                portfolio_delta += weight * delta
                portfolio_gamma += weight * gamma
                portfolio_vega += weight * vega
                portfolio_theta += weight * theta

            elif opt_type == "stock":
                # Stock has delta = 1, no other Greeks
                portfolio_delta += weight * 1.0

        return {
            "delta": portfolio_delta,
            "gamma": portfolio_gamma,
            "vega": portfolio_vega,
            "theta": portfolio_theta,
        }

    def get_total_cost(self) -> float:
        """
        Calculate the total net cost of the option portfolio.

        Returns:
            Net cost in dollars (positive = cash outflow, negative = cash inflow/credit)
            This represents: (Premium Paid for Longs) - (Premium Received from Shorts)
        """
        premiums, details = self.calculate_premiums()
        return details["net_cost"]

    def print_cost_breakdown(self):
        """
        Print a detailed breakdown of the cost of the approximation.
        """
        if self.weights is None:
            print("No weights computed yet. Call approximate() first.")
            return

        premiums, details = self.calculate_premiums()
        net_cost = details["net_cost"]
        gross_long = details["gross_long_cost"]
        gross_short = details["gross_short_credit"]
        gross_notional = details["gross_notional"]

        print("\n" + "=" * 70)
        print("COST BREAKDOWN - Decomposition into Vanilla Options")
        print("=" * 70)
        print(f"\nPricing Parameters:")
        print(f"  Current Stock Price (S0): ${self.S0:.2f}")
        print(f"  Risk-free Rate (r): {self.r*100:.2f}%")
        print(f"  Time to Expiration (T): {self.T:.3f} years ({self.T*365:.1f} days)")
        print(f"  Volatility (sigma): {self.sigma*100:.2f}%")

        print(f"\n{'='*70}")
        print(f"PORTFOLIO COST SUMMARY")
        print(f"{'='*70}")
        print(f"  Gross Long Positions (cash paid):     ${gross_long:>12.2f}")
        print(f"  Gross Short Positions (credit received): ${gross_short:>12.2f}")
        print(f"  {'─'*70}")
        print(f"  NET COST (outflow - inflow):          ${net_cost:>12.2f}")
        print(f"  {'─'*70}")
        print(f"  Gross Notional (capital required):    ${gross_notional:>12.2f}")
        print(f"{'='*70}")

        if details["calls"]:
            print(f"\nCALL OPTIONS ({len(details['calls'])} total):")
            print("-" * 80)
            print(
                f"{'Position':<8} {'Strike':<10} {'Weight':<12} {'Premium/Unit':<15} {'Net Cost':<15}"
            )
            print("-" * 80)
            call_net = 0.0
            for call in sorted(details["calls"], key=lambda x: x["strike"]):
                pos = call['position']
                sign = '+' if call['net_cost'] >= 0 else ''
                print(
                    f"{pos:<8} ${call['strike']:<9.2f} {call['weight']:>11.4f} "
                    f"${call['premium_per_unit']:<14.2f} {sign}${call['net_cost']:<14.2f}"
                )
                call_net += call["net_cost"]
            print("-" * 80)
            print(f"Call Options Net Cost: ${call_net:.2f}")

        if details["puts"]:
            print(f"\nPUT OPTIONS ({len(details['puts'])} total):")
            print("-" * 80)
            print(
                f"{'Position':<8} {'Strike':<10} {'Weight':<12} {'Premium/Unit':<15} {'Net Cost':<15}"
            )
            print("-" * 80)
            put_net = 0.0
            for put in sorted(details["puts"], key=lambda x: x["strike"]):
                pos = put['position']
                sign = '+' if put['net_cost'] >= 0 else ''
                print(
                    f"{pos:<8} ${put['strike']:<9.2f} {put['weight']:>11.4f} "
                    f"${put['premium_per_unit']:<14.2f} {sign}${put['net_cost']:<14.2f}"
                )
                put_net += put["net_cost"]
            print("-" * 80)
            print(f"Put Options Net Cost: ${put_net:.2f}")

        if details["stock_net_cost"] != 0:
            print(f"\nSTOCK POSITION:")
            sign = '+' if details["stock_net_cost"] >= 0 else ''
            print(f"  Net Cost: {sign}${details['stock_net_cost']:.2f}")

        if details["non_option_functions"]:
            print(
                f"\nNON-OPTION BASIS FUNCTIONS ({len(details['non_option_functions'])} total):"
            )
            print(
                "  (These functions have no direct cost but contribute to the approximation)"
            )
            for func in details["non_option_functions"][:5]:  # Show first 5
                print(f"    - {func['name']}: weight = {func['weight']:.4f}")
            if len(details["non_option_functions"]) > 5:
                print(f"    ... and {len(details['non_option_functions'])-5} more")

        # Calculate portfolio Greeks
        greeks = self.calculate_portfolio_greeks()

        print("\n" + "=" * 70)
        print("PORTFOLIO GREEKS (Risk Sensitivities)")
        print("-" * 70)
        print(f"  Delta (price sensitivity):  {greeks['delta']:>10.4f}")
        print(
            f"    -> Portfolio value changes by ${greeks['delta']:.2f} per $1 stock move"
        )
        print(f"  Gamma (delta sensitivity):  {greeks['gamma']:>10.4f}")
        print(f"    -> Delta changes by {greeks['gamma']:.4f} per $1 stock move")
        print(f"  Vega (volatility sensitivity): {greeks['vega']:>10.2f}")
        print(
            f"    -> Portfolio value changes by ${greeks['vega']:.2f} per 1% vol change"
        )
        print(f"  Theta (time decay):         {greeks['theta']:>10.4f}")
        print(f"    -> Portfolio loses ${abs(greeks['theta']):.4f} per day")
        print("-" * 70)

        # Calculate cost efficiency metrics
        print("\n" + "=" * 70)
        print("COST EFFICIENCY METRICS")
        print("-" * 70)
        num_long = sum(1 for c in details["calls"] if c["weight"] > 0) + sum(1 for p in details["puts"] if p["weight"] > 0)
        num_short = sum(1 for c in details["calls"] if c["weight"] < 0) + sum(1 for p in details["puts"] if p["weight"] < 0)
        total_opts = len(details['calls']) + len(details['puts'])

        print(f"  Total Options: {total_opts}")
        print(f"    - Long positions: {num_long}")
        print(f"    - Short positions: {num_short}")

        if gross_notional > 0:
            print(f"  Net Cost / Gross Notional: {net_cost / gross_notional * 100:.1f}%")
            if details['stock_net_cost'] != 0:
                stock_pct = abs(details['stock_net_cost']) / gross_notional * 100
                opts_pct = 100 - stock_pct
                print(f"  Options as % of Notional: {opts_pct:.1f}%")
                print(f"  Stock as % of Notional: {stock_pct:.1f}%")
        print("-" * 70)

        print("\n" + "=" * 70)
        print(
            f"SUMMARY: Complex product decomposed into {len(details['calls']) + len(details['puts'])} vanilla options"
        )
        print("=" * 70 + "\n")

    def plot_approximation(
        self,
        target_function: Callable,
        n_points: int = 1000,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Plot the target function and its approximation.

        Args:
            target_function: Original function to approximate
            n_points: Number of points for plotting
            title: Plot title
            save_path: Optional path to save the plot
        """
        stock_prices = np.linspace(self.price_range[0], self.price_range[1], n_points)
        target_values = target_function(stock_prices)
        approx_values = self.evaluate(stock_prices)

        plt.figure(figsize=(12, 6))
        plt.plot(
            stock_prices, target_values, "b-", label="Target Function", linewidth=2
        )
        plt.plot(
            stock_prices,
            approx_values,
            "r--",
            label="Options Approximation",
            linewidth=2,
        )
        plt.xlabel("Stock Price")
        plt.ylabel("Function Value")
        plt.title(title or f"Function Approximation using {self.n_options} Options")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add error info
        mse = np.mean((target_values - approx_values) ** 2)
        rmse = np.sqrt(mse)
        plt.text(
            0.02,
            0.98,
            f"RMSE: {rmse:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def print_weights(self):
        """Print the weights for each basis function."""
        if self.weights is None:
            print("No weights computed yet. Call approximate() first.")
            return

        print("\nOption Weights:")
        print("-" * 50)
        for i, (name, weight) in enumerate(zip(self.basis_names, self.weights)):
            print(f"{i+1:3d}. {name:20s}: {weight:10.4f}")
        print("-" * 50)
        print(f"Total basis functions: {self.n_basis}")


def example_sin_approximation():
    """Example: Approximate sin(x) using options."""
    print("Example: Approximating sin(x) using options\n")

    # Define target function
    def sin_function(x):
        return np.sin(x)

    # Create approximator
    approximator = OptionsFunctionApproximator(
        n_options=15,
        price_range=(0, 2 * np.pi),
        use_calls=True,
        use_puts=True,
        use_stock=True,
    )

    # Find approximation
    weights, mse = approximator.approximate(
        sin_function, n_points=1000, regularization=0.001, method="least_squares"
    )

    print(f"MSE Error: {mse:.6f}")
    print(f"RMSE Error: {np.sqrt(mse):.6f}\n")

    # Print weights
    approximator.print_weights()

    # Plot
    approximator.plot_approximation(
        sin_function, title="Approximating sin(x) with Options"
    )


def example_polynomial_approximation():
    """Example: Approximate a polynomial function."""
    print("\nExample: Approximating x^3 - 2x^2 + x using options\n")

    def poly_function(x):
        return x**3 - 2 * x**2 + x

    approximator = OptionsFunctionApproximator(
        n_options=20,
        price_range=(0, 5),
        use_calls=True,
        use_puts=True,
        use_stock=True,
    )

    weights, mse = approximator.approximate(
        poly_function,
        n_points=1000,
        regularization=0.001,
    )

    print(f"MSE Error: {mse:.6f}")
    print(f"RMSE Error: {np.sqrt(mse):.6f}\n")

    approximator.print_weights()
    approximator.plot_approximation(
        poly_function, title="Approximating Polynomial with Options"
    )


def example_gaussian_approximation():
    """Example: Approximate a Gaussian function using Gaussian and other basis functions."""
    print("\nExample: Approximating Gaussian PDF using advanced basis functions\n")

    def target_gaussian(x):
        mu = 5.0
        sigma = 1.0
        return gaussian_pdf(x, mu, sigma)

    approximator = OptionsFunctionApproximator(
        n_options=10,
        price_range=(0, 10),
        use_calls=True,
        use_puts=True,
        use_stock=True,
        use_gaussians=True,
        n_gaussians=5,
        use_sigmoids=True,
        n_sigmoids=3,
        use_cumulative_normal=True,
        n_cumulative_normal=3,
    )

    weights, mse = approximator.approximate(
        target_gaussian,
        n_points=1000,
        regularization=0.001,
    )

    print(f"MSE Error: {mse:.6f}")
    print(f"RMSE Error: {np.sqrt(mse):.6f}\n")

    approximator.print_weights()
    approximator.plot_approximation(
        target_gaussian,
        title="Approximating Gaussian PDF with Advanced Basis Functions",
    )


def example_sigmoid_approximation():
    """Example: Approximate a sigmoid function."""
    print("\nExample: Approximating sigmoid function\n")

    def target_sigmoid(x):
        return sigmoid(x, center=5.0, scale=2.0)

    approximator = OptionsFunctionApproximator(
        n_options=8,
        price_range=(0, 10),
        use_calls=True,
        use_puts=True,
        use_stock=True,
        use_sigmoids=True,
        n_sigmoids=5,
        use_cumulative_normal=True,
        n_cumulative_normal=5,
        use_exponential_decay=True,
        n_exponential_decay=3,
    )

    weights, mse = approximator.approximate(
        target_sigmoid,
        n_points=1000,
        regularization=0.001,
    )

    print(f"MSE Error: {mse:.6f}")
    print(f"RMSE Error: {np.sqrt(mse):.6f}\n")

    approximator.print_weights()
    approximator.plot_approximation(
        target_sigmoid, title="Approximating Sigmoid with Advanced Basis Functions"
    )


def example_black_scholes_payoff():
    """Example: Approximate a Black-Scholes-like payoff function."""
    print("\nExample: Approximating Black-Scholes-like payoff\n")

    def bs_payoff(x):
        # Simulate a call option payoff with some volatility effects
        strike = 5.0
        payoff = np.maximum(x - strike, 0)
        # Add some smoothness (like Black-Scholes would have before expiration)
        volatility_effect = 0.1 * np.exp(-((x - strike) ** 2) / (2 * 0.5**2))
        return payoff + volatility_effect

    approximator = OptionsFunctionApproximator(
        n_options=10,
        price_range=(0, 10),
        use_calls=True,
        use_puts=True,
        use_stock=True,
        use_black_scholes_delta=True,
        n_black_scholes_delta=5,
        use_gaussians=True,
        n_gaussians=5,
        use_cumulative_normal=True,
        n_cumulative_normal=3,
    )

    weights, mse = approximator.approximate(
        bs_payoff,
        n_points=1000,
        regularization=0.001,
    )

    print(f"MSE Error: {mse:.6f}")
    print(f"RMSE Error: {np.sqrt(mse):.6f}\n")

    approximator.print_weights()
    approximator.plot_approximation(
        bs_payoff, title="Approximating Black-Scholes-like Payoff"
    )


def example_volatility_smile_approximation():
    """Example: Approximate a volatility smile function."""
    print("\nExample: Approximating volatility smile\n")

    def target_vol_smile(x):
        center = 5.0
        vol_atm = 0.2
        vol_skew = 0.05
        return volatility_smile(x, center, vol_atm, vol_skew)

    approximator = OptionsFunctionApproximator(
        n_options=8,
        price_range=(0, 10),
        use_calls=True,
        use_puts=True,
        use_stock=True,
        use_volatility_smile=True,
        n_volatility_smile=5,
        use_polynomials=True,
        max_polynomial_power=3,
        use_gaussians=True,
        n_gaussians=3,
    )

    weights, mse = approximator.approximate(
        target_vol_smile,
        n_points=1000,
        regularization=0.001,
    )

    print(f"MSE Error: {mse:.6f}")
    print(f"RMSE Error: {np.sqrt(mse):.6f}\n")

    approximator.print_weights()
    approximator.plot_approximation(
        target_vol_smile, title="Approximating Volatility Smile"
    )


def example_complex_financial_function():
    """Example: Approximate a complex financial function combining multiple effects."""
    print("\nExample: Approximating complex financial function\n")

    def complex_function(x):
        # Combine multiple financial effects:
        # 1. Option-like payoff
        strike = 5.0
        option_payoff = np.maximum(x - strike, 0) * 0.5

        # 2. Volatility smile effect
        vol_effect = 0.1 * (x - strike) ** 2

        # 3. Time decay (exponential)
        decay_effect = 0.3 * np.exp(-0.2 * np.abs(x - strike))

        # 4. Probability weighting (sigmoid)
        prob_weight = 0.2 * sigmoid(x, center=strike, scale=1.0)

        return option_payoff + vol_effect - decay_effect + prob_weight

    approximator = OptionsFunctionApproximator(
        n_options=12,
        price_range=(0, 10),
        use_calls=True,
        use_puts=True,
        use_stock=True,
        use_gaussians=True,
        n_gaussians=5,
        use_sigmoids=True,
        n_sigmoids=5,
        use_cumulative_normal=True,
        n_cumulative_normal=3,
        use_exponential_decay=True,
        n_exponential_decay=3,
        use_volatility_smile=True,
        n_volatility_smile=3,
        use_black_scholes_delta=True,
        n_black_scholes_delta=3,
    )

    weights, mse = approximator.approximate(
        complex_function,
        n_points=1000,
        regularization=0.001,
    )

    print(f"MSE Error: {mse:.6f}")
    print(f"RMSE Error: {np.sqrt(mse):.6f}\n")

    approximator.print_weights()
    approximator.plot_approximation(
        complex_function, title="Approximating Complex Financial Function"
    )


def example_pricing_decomposition():
    """Example: Decompose a complex payoff into vanilla options with pricing."""
    print("\n" + "=" * 70)
    print("Example: Decomposing Complex Payoff into Vanilla Options with Pricing")
    print("=" * 70 + "\n")

    # Define a complex payoff function (e.g., a structured product)
    def complex_payoff(x):
        """
        Complex payoff: combination of option-like features
        - Bull spread component
        - Volatility-dependent component
        - Smooth transition
        """
        strike1 = 90.0
        strike2 = 110.0

        # Bull spread: long call at K1, short call at K2
        bull_spread = np.maximum(x - strike1, 0) - np.maximum(x - strike2, 0)

        # Add some smooth volatility-like component
        vol_component = 5.0 * np.exp(-((x - 100.0) ** 2) / (2 * 10.0**2))

        # Smooth transition
        transition = 2.0 * sigmoid(x, center=100.0, scale=0.1)

        return bull_spread + vol_component + transition

    # Create approximator with pricing parameters
    approximator = OptionsFunctionApproximator(
        n_options=15,
        price_range=(80, 120),
        use_calls=True,
        use_puts=True,
        use_stock=True,
        # Pricing parameters
        S0=100.0,  # Current stock price
        r=0.05,  # 5% risk-free rate
        T=0.25,  # 3 months to expiration
        sigma=0.2,  # 20% volatility
    )

    # Find approximation
    weights, mse = approximator.approximate(
        complex_payoff,
        n_points=1000,
        regularization=0.001,
    )

    print(f"Approximation Quality:")
    print(f"  MSE Error: {mse:.6f}")
    print(f"  RMSE Error: {np.sqrt(mse):.6f}\n")

    # Print cost breakdown
    approximator.print_cost_breakdown()

    # Show the decomposition
    print("\nDecomposition Summary:")
    premiums, details = approximator.calculate_premiums()
    total_cost = premiums.sum()

    num_calls = len(details["calls"])
    num_puts = len(details["puts"])

    print(f"  Complex product decomposed into:")
    print(f"    - {num_calls} call options")
    print(f"    - {num_puts} put options")
    if details["stock_cost"] > 0:
        print(f"    - Stock position")
    print(f"  Total replication cost: ${total_cost:.2f}")

    # Plot the approximation
    approximator.plot_approximation(
        complex_payoff,
        title="Complex Payoff Decomposed into Vanilla Options",
    )


if __name__ == "__main__":
    # Run pricing example
    example_pricing_decomposition()

    # Uncomment to run other examples:
    # example_sin_approximation()
    # example_polynomial_approximation()
    # example_gaussian_approximation()
    # example_sigmoid_approximation()
    # example_black_scholes_payoff()
    # example_volatility_smile_approximation()
    # example_complex_financial_function()
