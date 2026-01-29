"""
Run numerical experiments to demonstrate RMSE convergence of option-basis approximations.
Produces a CSV and a PNG plot in the `experiments` folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from options_func_maker import OptionsFunctionApproximator


def target_sin(x):
    return np.sin(x)


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def run_experiment(n_options_list, price_range=(0, 2 * np.pi), n_points=1000):
    xs = np.linspace(price_range[0], price_range[1], n_points)
    y_true = target_sin(xs)

    results = []
    for n in n_options_list:
        approximator = OptionsFunctionApproximator(
            n_options=n,
            price_range=price_range,
            use_calls=True,
            use_puts=True,
            use_stock=True,
        )
        weights, mse = approximator.approximate(
            target_sin, n_points=n_points, regularization=1e-6
        )
        y_pred = approximator.evaluate(xs)
        error = rmse(y_true, y_pred)
        print(f"n_options={n:3d}  RMSE={error:.6e}  MSE_reported={mse:.6e}")
        results.append((n, error, mse))

    # Save CSV
    csv_path = "experiments/results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_options", "rmse", "mse"])
        for row in results:
            writer.writerow(row)

    # Plot
    ns = [r[0] for r in results]
    rmses = [r[1] for r in results]
    plt.figure(figsize=(8, 5))
    plt.plot(ns, rmses, "-o")
    plt.xlabel("Number of options (strikes)")
    plt.ylabel("RMSE vs sin(x)")
    plt.title("Convergence of Option-Basis Approximation (sin on [0,2pi])")
    plt.grid(True)
    plt.savefig("experiments/rmse_vs_n.png", dpi=150)
    plt.show()

    print("Results saved to experiments/results.csv and experiments/rmse_vs_n.png")


if __name__ == "__main__":
    n_list = [4, 8, 12, 16, 24, 32]
    run_experiment(n_list)
