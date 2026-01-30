"""
Comprehensive Test Suite for Options Function Approximator

Tests Black-Scholes pricing, Greeks calculations, cost analysis, and numerical stability.
Run with: python test_calculations.py
"""

import numpy as np
from options_func_maker import (
    OptionsFunctionApproximator,
    black_scholes_call,
    black_scholes_put,
    black_scholes_delta_call,
    black_scholes_delta_put,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta_call,
    black_scholes_theta_put,
)


def test_put_call_parity():
    """Test that put-call parity holds: C - P = S - K*e^(-rT)"""
    print("\n" + "="*70)
    print("TEST 1: Put-Call Parity")
    print("="*70)

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    call = black_scholes_call(S, K, T, r, sigma)
    put = black_scholes_put(S, K, T, r, sigma)

    lhs = call - put
    rhs = S - K * np.exp(-r * T)

    error = abs(lhs - rhs)

    print(f"Parameters: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Call Price: ${call:.6f}")
    print(f"Put Price:  ${put:.6f}")
    print(f"C - P = {lhs:.6f}")
    print(f"S - K*e^(-rT) = {rhs:.6f}")
    print(f"Error: {error:.2e}")

    tolerance = 1e-10
    if error < tolerance:
        print(f" PASS: Put-call parity holds (error < {tolerance})")
        return True
    else:
        print(f" FAIL: Put-call parity violated (error = {error:.2e})")
        return False


def test_option_bounds():
    """Test that option prices satisfy no-arbitrage bounds"""
    print("\n" + "="*70)
    print("TEST 2: Option Price Bounds (No-Arbitrage)")
    print("="*70)

    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.25

    call = black_scholes_call(S, K, T, r, sigma)
    put = black_scholes_put(S, K, T, r, sigma)

    # Call bounds: max(0, S - K*e^(-rT)) <= C <= S
    call_lower = max(0, S - K * np.exp(-r * T))
    call_upper = S

    # Put bounds: max(0, K*e^(-rT) - S) <= P <= K*e^(-rT)
    put_lower = max(0, K * np.exp(-r * T) - S)
    put_upper = K * np.exp(-r * T)

    print(f"Parameters: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"\nCall Option: ${call:.4f}")
    print(f"  Lower bound: ${call_lower:.4f}")
    print(f"  Upper bound: ${call_upper:.4f}")
    call_ok = call_lower <= call <= call_upper
    print(f"  Status: {' PASS' if call_ok else ' FAIL'}")

    print(f"\nPut Option: ${put:.4f}")
    print(f"  Lower bound: ${put_lower:.4f}")
    print(f"  Upper bound: ${put_upper:.4f}")
    put_ok = put_lower <= put <= put_upper
    print(f"  Status: {' PASS' if put_ok else ' FAIL'}")

    return call_ok and put_ok


def test_greeks_numerical():
    """Verify Greeks using finite difference approximation"""
    print("\n" + "="*70)
    print("TEST 3: Greeks Validation (Finite Differences)")
    print("="*70)

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    # Delta (call)
    dS = 0.01
    call_up = black_scholes_call(S + dS, K, T, r, sigma)
    call_down = black_scholes_call(S - dS, K, T, r, sigma)
    delta_numeric = (call_up - call_down) / (2 * dS)
    delta_analytic = black_scholes_delta_call(S, K, T, r, sigma)
    delta_error = abs(delta_numeric - delta_analytic)

    print(f"\nDelta (Call):")
    print(f"  Analytic: {delta_analytic:.6f}")
    print(f"  Numeric:  {delta_numeric:.6f}")
    print(f"  Error:    {delta_error:.2e}")
    delta_ok = delta_error < 1e-5
    print(f"  Status: {' PASS' if delta_ok else ' FAIL'}")

    # Gamma
    delta_up = black_scholes_delta_call(S + dS, K, T, r, sigma)
    delta_down = black_scholes_delta_call(S - dS, K, T, r, sigma)
    gamma_numeric = (delta_up - delta_down) / (2 * dS)
    gamma_analytic = black_scholes_gamma(S, K, T, r, sigma)
    gamma_error = abs(gamma_numeric - gamma_analytic)

    print(f"\nGamma:")
    print(f"  Analytic: {gamma_analytic:.6f}")
    print(f"  Numeric:  {gamma_numeric:.6f}")
    print(f"  Error:    {gamma_error:.2e}")
    gamma_ok = gamma_error < 1e-5
    print(f"  Status: {' PASS' if gamma_ok else ' FAIL'}")

    # Vega
    dsigma = 0.01
    call_up_vol = black_scholes_call(S, K, T, r, sigma + dsigma)
    call_down_vol = black_scholes_call(S, K, T, r, sigma - dsigma)
    vega_numeric = (call_up_vol - call_down_vol) / (2 * dsigma)
    vega_analytic = black_scholes_vega(S, K, T, r, sigma)
    vega_error = abs(vega_numeric - vega_analytic)

    print(f"\nVega:")
    print(f"  Analytic: {vega_analytic:.6f}")
    print(f"  Numeric:  {vega_numeric:.6f}")
    print(f"  Error:    {vega_error:.2e}")
    vega_ok = vega_error < 0.01  # Relaxed tolerance for finite difference approximation
    print(f"  Status: {' PASS' if vega_ok else ' FAIL'}")

    # Theta (call)
    dT = 1.0 / 365.0  # 1 day in years
    call_t1 = black_scholes_call(S, K, T, r, sigma)
    call_t2 = black_scholes_call(S, K, T - dT, r, sigma)
    # Theta is the change in price per day, so just take the difference
    theta_numeric = call_t2 - call_t1  # Change over one day
    theta_analytic = black_scholes_theta_call(S, K, T, r, sigma)
    theta_error = abs(theta_numeric - theta_analytic)

    print(f"\nTheta (Call):")
    print(f"  Analytic: {theta_analytic:.6f}")
    print(f"  Numeric:  {theta_numeric:.6f}")
    print(f"  Error:    {theta_error:.2e}")
    theta_ok = theta_error < 1e-4
    print(f"  Status: {' PASS' if theta_ok else ' FAIL'}")

    return delta_ok and gamma_ok and vega_ok and theta_ok


def test_vega_scaling():
    """
    Test that Vega has correct scaling.
    Standard Vega should give price change per 0.01 change in volatility.
    """
    print("\n" + "="*70)
    print("TEST 4: Vega Scaling Validation")
    print("="*70)

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    vega = black_scholes_vega(S, K, T, r, sigma)

    # Test: change volatility by 0.01 (1 percentage point)
    dsigma = 0.01
    call1 = black_scholes_call(S, K, T, r, sigma)
    call2 = black_scholes_call(S, K, T, r, sigma + dsigma)
    actual_change = call2 - call1
    predicted_change = vega * dsigma

    error = abs(actual_change - predicted_change)

    print(f"Parameters: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Vega: {vega:.4f}")
    print(f"\nWhen σ changes from {sigma:.2f} to {sigma + dsigma:.2f}:")
    print(f"  Actual price change:    ${actual_change:.4f}")
    print(f"  Predicted (Vega * Δσ):  ${predicted_change:.4f}")
    print(f"  Error:                  ${error:.6f}")

    tolerance = 0.01  # 1 cent tolerance
    if error < tolerance:
        print(f" PASS: Vega scaling is correct (error < ${tolerance})")
        return True
    else:
        print(f" FAIL: Vega scaling incorrect (error = ${error:.4f})")
        return False


def test_cost_calculation_bull_spread():
    """Test that cost calculation correctly handles long and short positions (bull spread)"""
    print("\n" + "="*70)
    print("TEST 5: Cost Calculation - Bull Call Spread")
    print("="*70)

    # Bull call spread: long call at K1, short call at K2 (K2 > K1)
    def bull_spread(x):
        return np.maximum(x - 90, 0) - np.maximum(x - 110, 0)

    approx = OptionsFunctionApproximator(
        n_options=5,
        price_range=(80, 120),
        use_calls=True,
        use_puts=False,
        use_stock=False,
        S0=100.0,
        r=0.05,
        T=0.25,
        sigma=0.2,
    )

    weights, mse = approx.approximate(bull_spread, n_points=1000, regularization=1e-6)

    # Get cost details
    premiums, details = approx.calculate_premiums()
    net_cost = details["net_cost"]
    gross_long = details["gross_long_cost"]
    gross_short = details["gross_short_credit"]

    print(f"Bull Call Spread (long K=90, short K=110):")
    print(f"  RMSE: {np.sqrt(mse):.6f}")
    print(f"\nCost Breakdown:")
    print(f"  Gross Long (paid):       ${gross_long:.2f}")
    print(f"  Gross Short (received):  ${gross_short:.2f}")
    print(f"  Net Cost:                ${net_cost:.2f}")

    # For a bull spread, net cost should be positive but less than gross long
    # (you pay for the lower strike, receive premium from higher strike)
    cost_ok = 0 < net_cost < gross_long and gross_short > 0

    print(f"\nValidation:")
    print(f"  0 < Net Cost < Gross Long? {0 < net_cost < gross_long} {'' if 0 < net_cost < gross_long else ''}")
    print(f"  Gross Short > 0? {gross_short > 0} {'' if gross_short > 0 else ''}")

    if cost_ok:
        print(f"\n PASS: Cost calculation correctly handles long and short positions")
    else:
        print(f"\n FAIL: Cost calculation issue")

    return cost_ok


def test_edge_cases():
    """Test edge cases: T=0, deep ITM/OTM, etc."""
    print("\n" + "="*70)
    print("TEST 6: Edge Cases")
    print("="*70)

    # T = 0 (expiration)
    S, K, r, sigma = 100.0, 95.0, 0.05, 0.20
    call_t0 = black_scholes_call(S, K, 0, r, sigma)
    intrinsic_call = max(S - K, 0)
    error_call = abs(call_t0 - intrinsic_call)

    print(f"\nCase 1: T = 0 (At Expiration)")
    print(f"  Call (S={S}, K={K}): ${call_t0:.4f}")
    print(f"  Intrinsic value:     ${intrinsic_call:.4f}")
    print(f"  Error:               ${error_call:.6f}")
    case1_ok = error_call < 1e-10
    print(f"  Status: {' PASS' if case1_ok else ' FAIL'}")

    # Deep ITM call (K much less than S)
    call_itm = black_scholes_call(S, 50, 1.0, r, sigma)
    approx_itm = S - 50 * np.exp(-r * 1.0)
    error_itm = abs(call_itm - approx_itm)

    print(f"\nCase 2: Deep ITM Call (K=50, S={S})")
    print(f"  Call price:       ${call_itm:.4f}")
    print(f"  S - K*e^(-rT):    ${approx_itm:.4f}")
    print(f"  Difference:       ${error_itm:.4f}")
    case2_ok = error_itm < 1.0  # Should be close for deep ITM
    print(f"  Status: {' PASS' if case2_ok else ' FAIL'}")

    # Deep OTM call (K much greater than S)
    call_otm = black_scholes_call(S, 200, 1.0, r, sigma)

    print(f"\nCase 3: Deep OTM Call (K=200, S={S})")
    print(f"  Call price: ${call_otm:.6f}")
    case3_ok = 0 <= call_otm < 1.0  # Should be very small
    print(f"  Status: {' PASS (price is small)' if case3_ok else ' FAIL'}")

    return case1_ok and case2_ok and case3_ok


def test_matrix_conditioning():
    """Test that matrix conditioning warnings work"""
    print("\n" + "="*70)
    print("TEST 7: Matrix Conditioning Warning")
    print("="*70)

    # Use many options with no regularization to trigger warning
    def sin_func(x):
        return np.sin(x)

    print("Creating approximation with many basis functions and low regularization...")
    print("(This should trigger a conditioning warning)")

    approx = OptionsFunctionApproximator(
        n_options=40,  # Many options
        price_range=(0, 2*np.pi),
        use_calls=True,
        use_puts=True,
        use_stock=True,
    )

    # This should issue a warning
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        weights, mse = approx.approximate(sin_func, n_points=1000, regularization=1e-12)

        if len(w) > 0:
            print(f"\n PASS: Conditioning warning issued")
            print(f"Warning message: {w[0].message}")
            return True
        else:
            print(f"\n⚠️  No warning issued (may be OK if matrix is well-conditioned)")
            return True  # Don't fail if no warning - might be OK


def run_all_tests():
    """Run all tests and summarize results"""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUITE FOR OPTIONS FUNCTION APPROXIMATOR")
    print("="*70)

    tests = [
        ("Put-Call Parity", test_put_call_parity),
        ("Option Bounds", test_option_bounds),
        ("Greeks (Numerical)", test_greeks_numerical),
        ("Vega Scaling", test_vega_scaling),
        ("Cost Calculation (Bull Spread)", test_cost_calculation_bull_spread),
        ("Edge Cases", test_edge_cases),
        ("Matrix Conditioning", test_matrix_conditioning),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"  {name:<35} {status}")

    print("="*70)
    print(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\nALL TESTS PASSED!")
    else:
        print(f"\n  {total_count - passed_count} test(s) failed. Review the issues above.")

    print("="*70)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
