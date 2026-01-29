"""
Lightweight GUI Client for Options-Based Function Approximation

A simple tkinter-based interface for decomposing complex functions into
vanilla options with pricing and risk metrics.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from options_func_maker import (
    OptionsFunctionApproximator,
    sigmoid,
    gaussian_pdf,
    volatility_smile,
)


class OptionsApproximatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Options Function Approximator")
        self.root.geometry("1400x900")

        # Store current approximator and results
        self.approximator = None
        self.current_function = None
        self.approximation_error = None

        self.setup_ui()

    def setup_ui(self):
        # Create main paned window for resizable sections
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Controls
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=1)

        # Right panel: Results (split vertically)
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=2)

        # Top right: Plot
        plot_frame = ttk.Frame(right_paned)
        right_paned.add(plot_frame, weight=2)

        # Bottom right: Text output
        text_frame = ttk.Frame(right_paned)
        right_paned.add(text_frame, weight=1)

        self.setup_controls(left_frame)
        self.setup_plot(plot_frame)
        self.setup_text_output(text_frame)

    def setup_controls(self, parent):
        # Title
        title_label = ttk.Label(
            parent, text="Options Function Approximator", font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)

        # Function selection
        func_frame = ttk.LabelFrame(parent, text="Function Selection", padding=10)
        func_frame.pack(fill=tk.X, padx=5, pady=5)

        self.function_var = tk.StringVar(value="sin")
        functions = {
            "sin": ("sin(x)", lambda x: np.sin(x)),
            "cos": ("cos(x)", lambda x: np.cos(x)),
            "sigmoid": ("Sigmoid", lambda x: sigmoid(x, center=5.0, scale=2.0)),
            "gaussian": ("Gaussian PDF", lambda x: gaussian_pdf(x, mu=5.0, sigma=1.0)),
            "polynomial": ("x³ - 2x² + x", lambda x: x**3 - 2 * x**2 + x),
            "bull_spread": ("Bull Spread", lambda x: np.maximum(x - 90, 0) - np.maximum(x - 110, 0)),
            "complex": ("Complex Payoff", self.complex_payoff_function),
        }

        self.function_dict = functions

        for key, (label, _) in functions.items():
            ttk.Radiobutton(
                func_frame,
                text=label,
                variable=self.function_var,
                value=key,
                command=self.on_function_change,
            ).pack(anchor=tk.W, pady=2)

        # Custom function input
        ttk.Label(func_frame, text="Custom Function (Python):").pack(anchor=tk.W, pady=(10, 2))
        self.custom_func_entry = tk.Text(func_frame, height=3, width=35)
        self.custom_func_entry.pack(fill=tk.X, pady=2)
        self.custom_func_entry.insert("1.0", "np.sin(x) + 0.5*x")

        ttk.Button(
            func_frame,
            text="Use Custom Function",
            command=self.use_custom_function,
        ).pack(pady=5)

        # Price range
        range_frame = ttk.LabelFrame(parent, text="Price Range", padding=10)
        range_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(range_frame, text="Min Price:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.min_price_var = tk.StringVar(value="0")
        ttk.Entry(range_frame, textvariable=self.min_price_var, width=15).grid(
            row=0, column=1, sticky=tk.W, padx=5
        )

        ttk.Label(range_frame, text="Max Price:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.max_price_var = tk.StringVar(value="10")
        ttk.Entry(range_frame, textvariable=self.max_price_var, width=15).grid(
            row=1, column=1, sticky=tk.W, padx=5
        )

        # Number of options
        ttk.Label(range_frame, text="Number of Options:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.n_options_var = tk.StringVar(value="15")
        ttk.Entry(range_frame, textvariable=self.n_options_var, width=15).grid(
            row=2, column=1, sticky=tk.W, padx=5
        )

        # Pricing parameters
        pricing_frame = ttk.LabelFrame(parent, text="Pricing Parameters", padding=10)
        pricing_frame.pack(fill=tk.X, padx=5, pady=5)

        params = [
            ("Current Stock Price (S0):", "S0", "100.0"),
            ("Risk-free Rate (r):", "r", "0.05"),
            ("Time to Expiration (T, years):", "T", "0.25"),
            ("Volatility (σ):", "sigma", "0.2"),
        ]

        self.pricing_vars = {}
        for i, (label, key, default) in enumerate(params):
            ttk.Label(pricing_frame, text=label).grid(
                row=i, column=0, sticky=tk.W, pady=2
            )
            var = tk.StringVar(value=default)
            self.pricing_vars[key] = var
            ttk.Entry(pricing_frame, textvariable=var, width=15).grid(
                row=i, column=1, sticky=tk.W, padx=5
            )

        # Basis function options
        basis_frame = ttk.LabelFrame(parent, text="Basis Functions", padding=10)
        basis_frame.pack(fill=tk.X, padx=5, pady=5)

        self.use_calls_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            basis_frame, text="Use Call Options", variable=self.use_calls_var
        ).pack(anchor=tk.W)

        self.use_puts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            basis_frame, text="Use Put Options", variable=self.use_puts_var
        ).pack(anchor=tk.W)

        self.use_stock_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            basis_frame, text="Use Stock Position", variable=self.use_stock_var
        ).pack(anchor=tk.W)

        # Approximate button
        ttk.Button(
            parent,
            text="Calculate Approximation",
            command=self.calculate_approximation,
            style="Accent.TButton",
        ).pack(pady=20)

        # Status label
        self.status_label = ttk.Label(parent, text="Ready", foreground="green")
        self.status_label.pack(pady=5)

    def setup_plot(self, parent):
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Stock Price")
        self.ax.set_ylabel("Function Value")
        self.ax.set_title("Function Approximation")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_text_output(self, parent):
        ttk.Label(parent, text="Results & Cost Breakdown", font=("Arial", 10, "bold")).pack(
            anchor=tk.W, padx=5, pady=2
        )
        self.output_text = scrolledtext.ScrolledText(
            parent, height=15, width=80, wrap=tk.WORD
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def complex_payoff_function(self, x):
        """Complex payoff example."""
        strike1 = 90.0
        strike2 = 110.0
        bull_spread = np.maximum(x - strike1, 0) - np.maximum(x - strike2, 0)
        vol_component = 5.0 * np.exp(-((x - 100.0) ** 2) / (2 * 10.0 ** 2))
        transition = 2.0 * sigmoid(x, center=100.0, scale=0.1)
        return bull_spread + vol_component + transition

    def on_function_change(self):
        """Update price range based on selected function."""
        func_key = self.function_var.get()
        if func_key == "sin" or func_key == "cos":
            self.min_price_var.set("0")
            self.max_price_var.set(str(2 * np.pi))
        elif func_key == "complex":
            self.min_price_var.set("80")
            self.max_price_var.set("120")
        else:
            self.min_price_var.set("0")
            self.max_price_var.set("10")

    def use_custom_function(self):
        """Use custom function from text entry."""
        try:
            code = self.custom_func_entry.get("1.0", tk.END).strip()
            if code.startswith("#") or not code:
                messagebox.showwarning("Warning", "Please enter a valid function expression")
                return

            # Create a function from the code that works with numpy arrays
            def custom_func(x):
                # x is a numpy array, so we can use it directly in the expression
                # Allow common numpy operations
                namespace = {
                    "np": np,
                    "x": x,
                    "sin": np.sin,
                    "cos": np.cos,
                    "exp": np.exp,
                    "log": np.log,
                    "sqrt": np.sqrt,
                    "abs": np.abs,
                    "maximum": np.maximum,
                    "minimum": np.minimum,
                    "__builtins__": {},
                }
                return eval(code, namespace)

            # Test the function
            test_x = np.array([1.0, 2.0, 3.0])
            test_result = custom_func(test_x)
            
            # Verify it returns an array
            if not isinstance(test_result, np.ndarray):
                raise ValueError("Function must return a numpy array")

            self.function_dict["custom"] = ("Custom Function", custom_func)
            self.function_var.set("custom")
            self.status_label.config(text="Custom function loaded", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid function: {str(e)}\n\nMake sure to use numpy functions (np.sin, np.exp, etc.)")
            self.status_label.config(text="Error loading custom function", foreground="red")

    def get_pricing_params(self):
        """Extract pricing parameters from UI."""
        try:
            return {
                "S0": float(self.pricing_vars["S0"].get()),
                "r": float(self.pricing_vars["r"].get()),
                "T": float(self.pricing_vars["T"].get()),
                "sigma": float(self.pricing_vars["sigma"].get()),
            }
        except ValueError:
            messagebox.showerror("Error", "Invalid pricing parameters")
            return None

    def calculate_approximation(self):
        """Calculate the approximation and update displays."""
        try:
            self.status_label.config(text="Calculating...", foreground="blue")

            # Get function
            func_key = self.function_var.get()
            if func_key not in self.function_dict:
                messagebox.showerror("Error", "Please select a function")
                return

            _, target_func = self.function_dict[func_key]
            self.current_function = target_func

            # Get parameters
            try:
                min_price = float(self.min_price_var.get())
                max_price = float(self.max_price_var.get())
                n_options = int(self.n_options_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid price range or number of options")
                return

            pricing_params = self.get_pricing_params()
            if pricing_params is None:
                return

            # Create approximator
            self.approximator = OptionsFunctionApproximator(
                n_options=n_options,
                price_range=(min_price, max_price),
                use_calls=self.use_calls_var.get(),
                use_puts=self.use_puts_var.get(),
                use_stock=self.use_stock_var.get(),
                **pricing_params,
            )

            # Calculate approximation
            weights, mse = self.approximator.approximate(
                target_func, n_points=1000, regularization=0.001
            )
            self.approximation_error = {"mse": mse, "rmse": np.sqrt(mse)}

            # Update plot
            self.update_plot(target_func, min_price, max_price)

            # Update text output
            self.update_text_output()

            self.status_label.config(text="Calculation complete!", foreground="green")

        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")
            self.status_label.config(text="Error", foreground="red")

    def update_plot(self, target_func, min_price, max_price):
        """Update the plot with approximation."""
        self.ax.clear()

        stock_prices = np.linspace(min_price, max_price, 1000)
        target_values = target_func(stock_prices)

        if self.approximator is not None:
            approx_values = self.approximator.evaluate(stock_prices)

            self.ax.plot(
                stock_prices,
                target_values,
                "b-",
                label="Target Function",
                linewidth=2,
            )
            self.ax.plot(
                stock_prices,
                approx_values,
                "r--",
                label="Options Approximation",
                linewidth=2,
            )
            self.ax.legend()
            self.ax.set_xlabel("Stock Price")
            self.ax.set_ylabel("Function Value")
            self.ax.set_title("Function Approximation")
            self.ax.grid(True, alpha=0.3)

            # Add error info
            if self.approximation_error:
                rmse = self.approximation_error["rmse"]
                self.ax.text(
                    0.02,
                    0.98,
                    f"RMSE: {rmse:.4f}",
                    transform=self.ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        self.canvas.draw()

    def update_text_output(self):
        """Update the text output with results."""
        self.output_text.delete("1.0", tk.END)

        if self.approximator is None:
            return

        # Capture print output
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            # Print approximation quality
            if self.approximation_error:
                self.output_text.insert(tk.END, "APPROXIMATION QUALITY\n")
                self.output_text.insert(tk.END, "=" * 70 + "\n")
                self.output_text.insert(
                    tk.END,
                    f"MSE Error: {self.approximation_error['mse']:.6f}\n",
                )
                self.output_text.insert(
                    tk.END,
                    f"RMSE Error: {self.approximation_error['rmse']:.6f}\n\n",
                )

            # Print cost breakdown
            self.approximator.print_cost_breakdown()

            # Get the output
            output = buffer.getvalue()
            self.output_text.insert(tk.END, output)

        finally:
            sys.stdout = old_stdout

        # Make text read-only but allow scrolling
        self.output_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = OptionsApproximatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
