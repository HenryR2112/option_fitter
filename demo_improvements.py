"""
Quick demo of the improvements to the option fitter
Run this to see the benefits of iterative fitting with auto-regularization
"""
import numpy as np
import warnings
from options_func_maker import OptionsFunctionApproximator

print("=" * 70)
print("OPTION FITTER IMPROVEMENTS DEMO")
print("=" * 70)
print()

# Define a target function (bull call spread with some smoothing)
def target_function(x):
    """Bull call spread: long 90 call, short 110 call, plus smoothing"""
    return (np.maximum(x - 90, 0) - np.maximum(x - 110, 0) +
            2 * np.exp(-((x - 100)**2) / 200))

print("Target Function: Bull Call Spread with Gaussian smoothing")
print()

# Method 1: Traditional approach (uses all options)
print("=" * 70)
print("METHOD 1: Traditional Fitting (uses all options)")
print("=" * 70)

approx1 = OptionsFunctionApproximator(
    n_options=20,
    price_range=(80, 120),
    use_calls=True,
    use_puts=True,
    use_stock=True,
    S0=100.0,
    r=0.05,
    T=0.25,
    sigma=0.2,
)

print("Fitting with 20 options...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    weights1, mse1 = approx1.approximate(
        target_function,
        n_points=1000,
        regularization=1e-6
    )

    if w:
        print(f"⚠️  WARNING: {w[0].message}")
    else:
        print("✓ No warnings")

print(f"\nResults:")
print(f"  Basis functions used: {approx1.n_basis}")
print(f"  RMSE: {np.sqrt(mse1):.6f}")
print(f"  Non-zero weights: {np.sum(np.abs(weights1) > 1e-6)}")

# Method 2: New iterative approach
print("\n" + "=" * 70)
print("METHOD 2: Iterative Fitting with Auto-Regularization")
print("=" * 70)

approx2 = OptionsFunctionApproximator(
    n_options=20,  # Maximum to try
    price_range=(80, 120),
    use_calls=True,
    use_puts=True,
    use_stock=True,
    S0=100.0,
    r=0.05,
    T=0.25,
    sigma=0.2,
)

print("Searching for optimal configuration...")
print("(This evaluates multiple options/regularization combinations)\n")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    weights2, mse2 = approx2.approximate(
        target_function,
        n_points=1000,
        iterative=True,
        auto_regularization=True,
        min_options=2,
        validation_split=0.2,
        verbose=False  # Set to True to see all trials
    )

    if w:
        print(f"⚠️  WARNING: {w[0].message}")
    else:
        print("✓ No warnings")

print(f"\nResults:")
print(f"  Basis functions used: {approx2.n_basis}")
print(f"  RMSE: {np.sqrt(mse2):.6f}")
print(f"  Non-zero weights: {np.sum(np.abs(weights2) > 1e-6)}")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"\n{'Metric':<30} {'Traditional':<20} {'Iterative':<20}")
print("-" * 70)
print(f"{'Basis Functions':<30} {approx1.n_basis:<20} {approx2.n_basis:<20}")
print(f"{'RMSE':<30} {np.sqrt(mse1):<20.6f} {np.sqrt(mse2):<20.6f}")
print(f"{'Reduction in complexity':<30} {'-':<20} {f'{(1 - approx2.n_basis/approx1.n_basis)*100:.1f}%':<20}")

if approx2.n_basis < approx1.n_basis:
    print(f"\n✅ Iterative fitting used {approx1.n_basis - approx2.n_basis} fewer basis functions!")
    print(f"   This reduces overfitting and improves numerical stability.")

print("\n" + "=" * 70)
print("BENEFITS OF ITERATIVE FITTING")
print("=" * 70)
print("""
✓ Automatically finds optimal number of options
✓ Prevents overfitting by using validation set
✓ Reduces matrix conditioning issues
✓ Auto-tunes regularization strength
✓ No manual parameter tuning needed
✓ Backward compatible (opt-in feature)

To use in your code:
    weights, mse = approx.approximate(
        target_func,
        iterative=True,
        auto_regularization=True
    )
""")

print("=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print("\nFor more details, see IMPROVEMENTS.md")
print("To use in GUI: Check the 'Optimization Settings' boxes")
