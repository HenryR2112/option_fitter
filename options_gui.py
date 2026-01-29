"""
Enhanced GUI Client for Options-Based Function Approximation

A modern tkinter-based interface with improved error handling, performance,
and structured display boxes for decomposing complex functions into vanilla options.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import traceback
from typing import Optional, Dict, Any
from options_func_maker import (
    OptionsFunctionApproximator,
    sigmoid,
    gaussian_pdf,
    volatility_smile,
)


class ModernStyle:
    """Modern Professional Finance Tool Color Scheme - Bloomberg/TradingView inspired."""

    # Modern Dark Theme Backgrounds
    BG_PRIMARY = "#0f1419"  # Deep dark blue-gray (main background)
    BG_SECONDARY = "#1a1f2e"  # Dark blue-gray (panels/cards)
    BG_ACCENT = "#252b3b"  # Lighter blue-gray (hover states)
    BG_HIGHLIGHT = "#2d3447"  # Highlighted elements
    BG_CARD = "#1e2332"  # Card background
    BG_INPUT = "#151a25"  # Input field background

    # Professional Text Colors
    TEXT_PRIMARY = "#e4e6eb"  # Off-white (primary text)
    TEXT_SECONDARY = "#9ca3af"  # Medium gray (secondary text)
    TEXT_MUTED = "#6b7280"  # Muted gray (tertiary text)
    TEXT_DISABLED = "#4b5563"  # Disabled text

    # Finance Color Palette (Professional)
    ACCENT_PRIMARY = "#3b82f6"  # Professional blue (primary actions)
    ACCENT_SUCCESS = "#10b981"  # Professional green (positive/gains)
    ACCENT_DANGER = "#ef4444"  # Professional red (negative/losses)
    ACCENT_WARNING = "#f59e0b"  # Professional amber (warnings)
    ACCENT_INFO = "#06b6d4"  # Professional cyan (info)

    # Chart Colors
    CHART_LINE_1 = "#3b82f6"  # Blue for target function
    CHART_LINE_2 = "#ef4444"  # Red for approximation
    CHART_GRID = "#2d3447"  # Subtle grid lines
    CHART_AXIS = "#6b7280"  # Axis labels

    # UI Element Colors
    BORDER_COLOR = "#2d3447"  # Subtle borders
    BORDER_ACTIVE = "#3b82f6"  # Active border (focus)
    SELECTION_COLOR = "#3b82f6"  # Selection highlight
    SHADOW_COLOR = "#000000"  # Shadow color

    # Status Colors
    STATUS_SUCCESS = "#10b981"  # Success state
    STATUS_ERROR = "#ef4444"  # Error state
    STATUS_WARNING = "#f59e0b"  # Warning state
    STATUS_INFO = "#06b6d4"  # Info state

    # Plot colors (for matplotlib)
    PLOT_BG = "#0f1419"  # Plot background
    PLOT_FG = "#e4e6eb"  # Plot foreground/text
    PLOT_GRID = "#2d3447"  # Plot grid


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class OptionsApproximatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Options Function Approximator - MSDOS Terminal")

        # Auto-size to screen (fullscreen)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.state("zoomed")  # Maximize on Windows

        self.root.configure(bg=ModernStyle.BG_PRIMARY)

        # Store current approximator and results
        self.approximator: Optional[OptionsFunctionApproximator] = None
        self.current_function: Optional[callable] = None
        self.approximation_error: Optional[Dict[str, float]] = None
        self.premium_details: Optional[Dict[str, Any]] = None
        self.greeks: Optional[Dict[str, float]] = None

        # Threading control
        self.calculation_thread: Optional[threading.Thread] = None
        self.is_calculating = False

        # Configure styles
        self.setup_styles()
        self.setup_ui()

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """Configure Modern Professional Finance Tool styling."""
        style = ttk.Style()
        style.theme_use("clam")

        # Modern font family
        font_family = ("Segoe UI", "Helvetica Neue", "Arial", "sans-serif")
        font_mono = ("Consolas", "Courier New", "monospace")

        # Base frame styling
        style.configure(
            "TFrame",
            background=ModernStyle.BG_PRIMARY,
            borderwidth=0,
        )
        style.configure(
            "TLabel",
            background=ModernStyle.BG_PRIMARY,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=font_family,
        )

        # Modern button styling (compact)
        style.configure(
            "TButton",
            background=ModernStyle.BG_SECONDARY,
            foreground=ModernStyle.TEXT_PRIMARY,
            borderwidth=0,
            relief="flat",
            padding=(8, 6),
            font=(font_family[0], 8),
        )
        style.map(
            "TButton",
            background=[
                ("active", ModernStyle.BG_ACCENT),
                ("pressed", ModernStyle.BG_HIGHLIGHT),
            ],
        )

        # Primary action button (modern blue, compact)
        style.configure(
            "Accent.TButton",
            background=ModernStyle.ACCENT_PRIMARY,
            foreground="#ffffff",
            font=(font_family[0], 9, "bold"),
            padding=(16, 10),
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "Accent.TButton",
            background=[
                ("active", "#2563eb"),
                ("pressed", "#1d4ed8"),
            ],
        )

        # Modern card-style label frames (compact)
        style.configure(
            "TLabelframe",
            background=ModernStyle.BG_CARD,
            borderwidth=1,
            relief="flat",
            bordercolor=ModernStyle.BORDER_COLOR,
        )
        style.configure(
            "TLabelframe.Label",
            background=ModernStyle.BG_CARD,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=(font_family[0], 9, "bold"),
        )

        # Modern input fields (compact)
        style.configure(
            "TEntry",
            fieldbackground=ModernStyle.BG_INPUT,
            foreground=ModernStyle.TEXT_PRIMARY,
            borderwidth=1,
            padding=(6, 4),
            insertcolor=ModernStyle.ACCENT_PRIMARY,
            relief="flat",
            font=font_mono,
        )
        style.map(
            "TEntry",
            fieldbackground=[("focus", ModernStyle.BG_ACCENT)],
            bordercolor=[("focus", ModernStyle.BORDER_ACTIVE)],
        )

        # Modern radio buttons
        style.configure(
            "TRadiobutton",
            background=ModernStyle.BG_CARD,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=font_family,
        )
        style.map(
            "TRadiobutton",
            background=[("active", ModernStyle.BG_CARD)],
            foreground=[("selected", ModernStyle.ACCENT_PRIMARY)],
        )

        # Modern checkboxes
        style.configure(
            "TCheckbutton",
            background=ModernStyle.BG_CARD,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=font_family,
        )
        style.map(
            "TCheckbutton",
            background=[("active", ModernStyle.BG_CARD)],
            foreground=[("selected", ModernStyle.ACCENT_PRIMARY)],
        )

        # Modern tabs
        style.configure(
            "TNotebook",
            background=ModernStyle.BG_PRIMARY,
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background=ModernStyle.BG_SECONDARY,
            foreground=ModernStyle.TEXT_PRIMARY,
            padding=[12, 8],
            borderwidth=0,
            font=(font_family[0], 8),
        )
        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", ModernStyle.BG_CARD),
                ("active", ModernStyle.BG_ACCENT),
            ],
            foreground=[
                ("selected", ModernStyle.ACCENT_PRIMARY),
                ("active", ModernStyle.TEXT_PRIMARY),
            ],
            expand=[("selected", [1, 1, 1, 0])],
        )

        # Modern table styling
        style.configure(
            "Treeview",
            background=ModernStyle.BG_CARD,
            foreground=ModernStyle.TEXT_PRIMARY,
            fieldbackground=ModernStyle.BG_CARD,
            borderwidth=0,
            font=font_mono,
        )
        style.configure(
            "Treeview.Heading",
            background=ModernStyle.BG_SECONDARY,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=(font_family[0], 9, "bold"),
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Treeview",
            background=[("selected", ModernStyle.ACCENT_PRIMARY)],
            foreground=[("selected", "#ffffff")],
        )

        # Modern scrollbars
        style.configure(
            "TScrollbar",
            background=ModernStyle.BG_SECONDARY,
            troughcolor=ModernStyle.BG_PRIMARY,
            borderwidth=0,
            arrowcolor=ModernStyle.TEXT_SECONDARY,
            darkcolor=ModernStyle.BG_SECONDARY,
            lightcolor=ModernStyle.BG_SECONDARY,
            width=12,
        )

        # Professional matplotlib theme
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "figure.facecolor": ModernStyle.PLOT_BG,
                "axes.facecolor": ModernStyle.BG_CARD,
                "axes.edgecolor": ModernStyle.BORDER_COLOR,
                "axes.labelcolor": ModernStyle.TEXT_PRIMARY,
                "xtick.color": ModernStyle.CHART_AXIS,
                "ytick.color": ModernStyle.CHART_AXIS,
                "text.color": ModernStyle.TEXT_PRIMARY,
                "grid.color": ModernStyle.CHART_GRID,
                "grid.alpha": 0.3,
                "font.family": "sans-serif",
                "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial"],
            }
        )

    def setup_ui(self):
        """Set up the user interface."""
        # Create main paned window (compact)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left panel: Controls
        left_frame = ttk.Frame(main_paned, width=420)
        left_frame.configure(style="TFrame")
        main_paned.add(left_frame, weight=1)

        # Right panel: Results (split vertically)
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=3)

        # Top right: Plot (larger)
        plot_frame = ttk.Frame(right_paned)
        right_paned.add(plot_frame, weight=3)

        # Bottom right: Results display (smaller)
        results_frame = ttk.Frame(right_paned)
        right_paned.add(results_frame, weight=1)

        self.setup_controls(left_frame)
        self.setup_plot(plot_frame)
        self.setup_results_display(results_frame)

    def setup_controls(self, parent):
        """Set up control panel with modern professional styling."""
        # Modern header card (compact)
        title_frame = tk.Frame(parent, bg=ModernStyle.BG_CARD, relief="flat")
        title_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        title_label = tk.Label(
            title_frame,
            text="Options Function Approximator",
            font=("Segoe UI", 14, "bold"),
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
        )
        title_label.pack(pady=(8, 2))

        subtitle_label = tk.Label(
            title_frame,
            text="Decompose complex payoffs into vanilla options",
            font=("Segoe UI", 8),
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_SECONDARY,
        )
        subtitle_label.pack(pady=(0, 8))

        # Function selection - Modern card (compact)
        func_frame = ttk.LabelFrame(parent, text="Function Selection", padding=10)
        func_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        self.function_var = tk.StringVar(value="sin")
        functions = {
            "sin": ("sin(x)", lambda x: np.sin(x)),
            "cos": ("cos(x)", lambda x: np.cos(x)),
            "sigmoid": ("Sigmoid", lambda x: sigmoid(x, center=5.0, scale=2.0)),
            "gaussian": ("Gaussian PDF", lambda x: gaussian_pdf(x, mu=5.0, sigma=1.0)),
            "polynomial": ("x³ - 2x² + x", lambda x: x**3 - 2 * x**2 + x),
            "bull_spread": (
                "Bull Spread",
                lambda x: np.maximum(x - 90, 0) - np.maximum(x - 110, 0),
            ),
            "complex": ("Complex Payoff", self.complex_payoff_function),
        }

        self.function_dict = functions

        func_container = tk.Frame(func_frame, bg=ModernStyle.BG_CARD)
        func_container.pack(fill=tk.X)

        for key, (label, _) in functions.items():
            rb = ttk.Radiobutton(
                func_container,
                text=label,
                variable=self.function_var,
                value=key,
                command=self.on_function_change,
            )
            rb.pack(anchor=tk.W, pady=1)

        # Custom function input - Modern styling (compact)
        custom_label = tk.Label(
            func_frame,
            text="Custom Function (Python)",
            font=("Segoe UI", 8, "bold"),
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
        )
        custom_label.pack(anchor=tk.W, pady=(8, 4))

        self.custom_func_entry = tk.Text(
            func_frame,
            height=2,
            width=40,
            wrap=tk.WORD,
            bg=ModernStyle.BG_INPUT,
            fg=ModernStyle.TEXT_PRIMARY,
            insertbackground=ModernStyle.ACCENT_PRIMARY,
            relief="flat",
            borderwidth=1,
            selectbackground=ModernStyle.ACCENT_PRIMARY,
            selectforeground="#ffffff",
            font=("Consolas", 8),
            padx=6,
            pady=4,
        )
        self.custom_func_entry.pack(fill=tk.X, pady=(0, 6))
        self.custom_func_entry.insert("1.0", "np.sin(x) + 0.5*x")

        custom_btn = ttk.Button(
            func_frame,
            text="Use Custom Function",
            command=self.use_custom_function,
        )
        custom_btn.pack(pady=(0, 0))

        # Price range - Modern card (compact)
        range_frame = ttk.LabelFrame(parent, text="Price Range & Options", padding=10)
        range_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        self.create_labeled_entry(range_frame, "Min Price", "min_price", "0", 0)
        self.create_labeled_entry(range_frame, "Max Price", "max_price", "10", 1)
        self.create_labeled_entry(
            range_frame, "Number of Options", "n_options", "15", 2
        )

        # Pricing parameters - Modern card (compact)
        pricing_frame = ttk.LabelFrame(parent, text="Pricing Parameters", padding=10)
        pricing_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        params = [
            ("Current Stock Price (S₀)", "S0", "100.0"),
            ("Risk-free Rate (r)", "r", "0.05"),
            ("Time to Expiration (T, years)", "T", "0.25"),
            ("Volatility (σ)", "sigma", "0.2"),
        ]

        # Initialize pricing_vars before creating entries
        if not hasattr(self, "pricing_vars"):
            self.pricing_vars = {}

        for i, (label, key, default) in enumerate(params):
            self.create_labeled_entry(pricing_frame, label, key, default, i)

        # Basis function options - Modern card (compact)
        basis_frame = ttk.LabelFrame(parent, text="Basis Functions", padding=10)
        basis_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        self.use_calls_var = tk.BooleanVar(value=True)
        self.use_puts_var = tk.BooleanVar(value=True)
        self.use_stock_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            basis_frame, text="Use Call Options", variable=self.use_calls_var
        ).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(
            basis_frame, text="Use Put Options", variable=self.use_puts_var
        ).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(
            basis_frame, text="Use Stock Position", variable=self.use_stock_var
        ).pack(anchor=tk.W, pady=2)

        # Calculate button - Modern primary action (compact)
        button_frame = tk.Frame(parent, bg=ModernStyle.BG_PRIMARY)
        button_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        self.calculate_btn = ttk.Button(
            button_frame,
            text="Calculate Approximation",
            command=self.calculate_approximation,
            style="Accent.TButton",
        )
        self.calculate_btn.pack(fill=tk.X, pady=(0, 4))

        # Progress indicator
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            button_frame,
            textvariable=self.progress_var,
            font=("Segoe UI", 8),
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_SECONDARY,
        )
        self.progress_label.pack(pady=2)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(
            button_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9, "bold"),
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.STATUS_SUCCESS,
        )
        self.status_label.pack(pady=2)

    def create_labeled_entry(self, parent, label_text, var_key, default_value, row):
        """Helper to create modern labeled entry widgets (compact)."""
        label = tk.Label(
            parent,
            text=label_text,
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
            font=("Segoe UI", 8),
        )
        label.grid(row=row, column=0, sticky=tk.W, pady=3, padx=(0, 8))

        var = tk.StringVar(value=default_value)
        if var_key in ["S0", "r", "T", "sigma"]:
            self.pricing_vars[var_key] = var
        elif var_key == "min_price":
            self.min_price_var = var
        elif var_key == "max_price":
            self.max_price_var = var
        elif var_key == "n_options":
            self.n_options_var = var

        entry = ttk.Entry(parent, textvariable=var, width=18)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=(0, 0), pady=3)
        parent.columnconfigure(1, weight=1)

    def setup_plot(self, parent):
        """Set up the modern professional plot area (larger, compact padding)."""
        plot_container = tk.Frame(parent, bg=ModernStyle.BG_CARD, relief="flat")
        plot_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Modern plot header (compact)
        plot_header = tk.Frame(plot_container, bg=ModernStyle.BG_CARD)
        plot_header.pack(fill=tk.X, padx=8, pady=(8, 4))

        plot_title = tk.Label(
            plot_header,
            text="Function Approximation",
            font=("Segoe UI", 12, "bold"),
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
        )
        plot_title.pack(side=tk.LEFT)

        # Matplotlib figure (Modern professional theme - larger)
        self.fig = Figure(figsize=(12, 7), dpi=110, facecolor=ModernStyle.BG_CARD)
        self.ax = self.fig.add_subplot(111, facecolor=ModernStyle.BG_CARD)
        self.ax.set_xlabel("Stock Price", fontsize=11, color=ModernStyle.TEXT_PRIMARY)
        self.ax.set_ylabel(
            "Function Value", fontsize=11, color=ModernStyle.TEXT_PRIMARY
        )
        self.ax.set_title(
            "Target Function vs. Options Approximation",
            fontsize=12,
            fontweight="bold",
            color=ModernStyle.TEXT_PRIMARY,
            pad=12,
        )
        self.ax.tick_params(colors=ModernStyle.CHART_AXIS, labelsize=9)
        self.ax.grid(
            True, alpha=0.2, linestyle="-", color=ModernStyle.CHART_GRID, linewidth=0.5
        )
        self.ax.spines["bottom"].set_color(ModernStyle.BORDER_COLOR)
        self.ax.spines["top"].set_color(ModernStyle.BORDER_COLOR)
        self.ax.spines["right"].set_color(ModernStyle.BORDER_COLOR)
        self.ax.spines["left"].set_color(ModernStyle.BORDER_COLOR)

        self.canvas = FigureCanvasTkAgg(self.fig, plot_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_results_display(self, parent):
        """Set up structured results display with tabs."""
        # Create notebook for tabs
        self.results_notebook = ttk.Notebook(parent)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Summary & Metrics
        summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(summary_frame, text="Summary & Metrics")
        self.setup_summary_tab(summary_frame)

        # Tab 2: Cost Breakdown
        cost_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(cost_frame, text="Cost Breakdown")
        self.setup_cost_tab(cost_frame)

        # Tab 3: Portfolio Greeks
        greeks_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(greeks_frame, text="Portfolio Greeks")
        self.setup_greeks_tab(greeks_frame)

        # Tab 4: Detailed Output (for compatibility)
        detail_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(detail_frame, text="Detailed Output")
        self.setup_detail_tab(detail_frame)

    def setup_summary_tab(self, parent):
        """Set up summary and metrics tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=ModernStyle.BG_PRIMARY)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=ModernStyle.BG_PRIMARY)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Approximation Quality - Modern card (compact)
        quality_frame = tk.LabelFrame(
            scrollable_frame,
            text="Approximation Quality",
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=10,
        )
        quality_frame.pack(fill=tk.X, padx=8, pady=6)

        self.mse_label = tk.Label(
            quality_frame,
            text="MSE: -",
            font=("Segoe UI", 9),
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
        )
        self.mse_label.pack(anchor=tk.W, pady=2)

        self.rmse_label = tk.Label(
            quality_frame,
            text="RMSE: -",
            font=("Segoe UI", 9),
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
        )
        self.rmse_label.pack(anchor=tk.W, pady=2)

        # Total Cost - Modern card (compact)
        cost_frame = tk.LabelFrame(
            scrollable_frame,
            text="Total Portfolio Cost",
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=10,
        )
        cost_frame.pack(fill=tk.X, padx=8, pady=6)

        self.total_cost_label = tk.Label(
            cost_frame,
            text="Total Cost: $0.00",
            font=("Segoe UI", 14, "bold"),
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.ACCENT_PRIMARY,
        )
        self.total_cost_label.pack(pady=5)

        # Portfolio Composition - Modern card (compact)
        comp_frame = tk.LabelFrame(
            scrollable_frame,
            text="Portfolio Composition",
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=10,
        )
        comp_frame.pack(fill=tk.X, padx=8, pady=6)

        self.comp_label = tk.Label(
            comp_frame,
            text="No calculation performed yet.",
            font=("Segoe UI", 8),
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
            justify=tk.LEFT,
        )
        self.comp_label.pack(anchor=tk.W, pady=2)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def setup_cost_tab(self, parent):
        """Set up cost breakdown tab with TreeView (compact)."""
        # Create TreeView for options
        tree_frame = tk.Frame(parent, bg=ModernStyle.BG_PRIMARY)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Call options tree
        calls_label = tk.Label(
            tree_frame,
            text="Call Options",
            font=("Segoe UI", 10, "bold"),
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.ACCENT_SUCCESS,
        )
        calls_label.pack(anchor=tk.W, pady=(0, 4))

        calls_tree_frame = tk.Frame(tree_frame, bg=ModernStyle.BG_PRIMARY)
        calls_tree_frame.pack(fill=tk.BOTH, expand=True)

        self.calls_tree = ttk.Treeview(
            calls_tree_frame,
            columns=("Strike", "Weight", "Premium/Unit", "Total Cost"),
            show="headings",
            height=6,
        )
        self.calls_tree.heading("Strike", text="Strike")
        self.calls_tree.heading("Weight", text="Weight")
        self.calls_tree.heading("Premium/Unit", text="Premium/Unit")
        self.calls_tree.heading("Total Cost", text="Total Cost")

        for col in ("Strike", "Weight", "Premium/Unit", "Total Cost"):
            self.calls_tree.column(col, width=120, anchor=tk.CENTER)

        calls_scrollbar = ttk.Scrollbar(
            calls_tree_frame, orient=tk.VERTICAL, command=self.calls_tree.yview
        )
        self.calls_tree.configure(yscrollcommand=calls_scrollbar.set)

        self.calls_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        calls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Put options tree
        puts_label = tk.Label(
            tree_frame,
            text="Put Options",
            font=("Segoe UI", 10, "bold"),
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.ACCENT_DANGER,
        )
        puts_label.pack(anchor=tk.W, pady=(8, 4))

        puts_tree_frame = tk.Frame(tree_frame, bg=ModernStyle.BG_PRIMARY)
        puts_tree_frame.pack(fill=tk.BOTH, expand=True)

        self.puts_tree = ttk.Treeview(
            puts_tree_frame,
            columns=("Strike", "Weight", "Premium/Unit", "Total Cost"),
            show="headings",
            height=6,
        )
        self.puts_tree.heading("Strike", text="Strike")
        self.puts_tree.heading("Weight", text="Weight")
        self.puts_tree.heading("Premium/Unit", text="Premium/Unit")
        self.puts_tree.heading("Total Cost", text="Total Cost")

        for col in ("Strike", "Weight", "Premium/Unit", "Total Cost"):
            self.puts_tree.column(col, width=120, anchor=tk.CENTER)

        puts_scrollbar = ttk.Scrollbar(
            puts_tree_frame, orient=tk.VERTICAL, command=self.puts_tree.yview
        )
        self.puts_tree.configure(yscrollcommand=puts_scrollbar.set)

        self.puts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        puts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_greeks_tab(self, parent):
        """Set up portfolio Greeks tab (compact)."""
        greeks_container = tk.Frame(parent, bg=ModernStyle.BG_PRIMARY)
        greeks_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Create styled boxes for each Greek
        greeks_data = [
            ("Delta", "Price Sensitivity", "$ per $1 stock move"),
            ("Gamma", "Delta Sensitivity", "Delta change per $1 stock move"),
            ("Vega", "Volatility Sensitivity", "$ per 1% vol change"),
            ("Theta", "Time Decay", "$ per day"),
        ]

        self.greek_labels = {}
        for i, (greek_name, description, units) in enumerate(greeks_data):
            greek_frame = tk.LabelFrame(
                greeks_container,
                text=greek_name,
                bg=ModernStyle.BG_CARD,
                fg=ModernStyle.TEXT_PRIMARY,
                font=("Segoe UI", 9, "bold"),
                padx=10,
                pady=10,
                relief="flat",
                borderwidth=1,
            )
            greek_frame.pack(fill=tk.X, pady=4)

            desc_label = tk.Label(
                greek_frame,
                text=description,
                font=("Segoe UI", 8),
                bg=ModernStyle.BG_CARD,
                fg=ModernStyle.TEXT_SECONDARY,
            )
            desc_label.pack(anchor=tk.W)

            value_label = tk.Label(
                greek_frame,
                text="-",
                font=("Segoe UI", 14, "bold"),
                bg=ModernStyle.BG_CARD,
                fg=ModernStyle.ACCENT_PRIMARY,
            )
            value_label.pack(pady=3)
            self.greek_labels[greek_name.lower()] = value_label

            units_label = tk.Label(
                greek_frame,
                text=units,
                font=("Segoe UI", 7),
                bg=ModernStyle.BG_CARD,
                fg=ModernStyle.TEXT_SECONDARY,
            )
            units_label.pack(anchor=tk.W)

    def setup_detail_tab(self, parent):
        """Set up detailed text output tab (compact)."""
        detail_label = tk.Label(
            parent,
            text="Detailed Output",
            font=("Segoe UI", 9, "bold"),
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
        )
        detail_label.pack(anchor=tk.W, padx=8, pady=4)

        self.output_text = scrolledtext.ScrolledText(
            parent,
            height=18,
            width=80,
            wrap=tk.WORD,
            bg=ModernStyle.BG_CARD,
            fg=ModernStyle.TEXT_PRIMARY,
            font=("Consolas", 8),
            relief="flat",
            borderwidth=1,
            insertbackground=ModernStyle.ACCENT_PRIMARY,
            selectbackground=ModernStyle.ACCENT_PRIMARY,
            selectforeground="#ffffff",
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def complex_payoff_function(self, x):
        """Complex payoff example."""
        strike1 = 90.0
        strike2 = 110.0
        bull_spread = np.maximum(x - strike1, 0) - np.maximum(x - strike2, 0)
        vol_component = 5.0 * np.exp(-((x - 100.0) ** 2) / (2 * 10.0**2))
        transition = 2.0 * sigmoid(x, center=100.0, scale=0.1)
        return bull_spread + vol_component + transition

    def on_function_change(self):
        """Update price range based on selected function."""
        try:
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
        except Exception as e:
            self.show_error("Error updating function", str(e))

    def validate_inputs(self) -> tuple:
        """Validate all user inputs and return parameters."""
        try:
            # Validate function
            func_key = self.function_var.get()
            if func_key not in self.function_dict:
                raise ValidationError("Please select a function")

            # Validate price range
            try:
                min_price = float(self.min_price_var.get())
                max_price = float(self.max_price_var.get())
            except ValueError:
                raise ValidationError("Price range must be valid numbers")

            if min_price >= max_price:
                raise ValidationError("Min price must be less than max price")

            if min_price < 0:
                raise ValidationError("Min price cannot be negative")

            # Validate number of options
            try:
                n_options = int(self.n_options_var.get())
            except ValueError:
                raise ValidationError("Number of options must be an integer")

            if n_options < 1 or n_options > 100:
                raise ValidationError("Number of options must be between 1 and 100")

            # Validate pricing parameters
            try:
                pricing_params = {
                    "S0": float(self.pricing_vars["S0"].get()),
                    "r": float(self.pricing_vars["r"].get()),
                    "T": float(self.pricing_vars["T"].get()),
                    "sigma": float(self.pricing_vars["sigma"].get()),
                }
            except ValueError:
                raise ValidationError("All pricing parameters must be valid numbers")

            if pricing_params["S0"] <= 0:
                raise ValidationError("Stock price must be positive")
            if pricing_params["r"] < 0 or pricing_params["r"] > 1:
                raise ValidationError("Risk-free rate should be between 0 and 1")
            if pricing_params["T"] <= 0:
                raise ValidationError("Time to expiration must be positive")
            if pricing_params["sigma"] <= 0 or pricing_params["sigma"] > 2:
                raise ValidationError("Volatility should be between 0 and 2 (0-200%)")

            # Validate basis functions
            if not (
                self.use_calls_var.get()
                or self.use_puts_var.get()
                or self.use_stock_var.get()
            ):
                raise ValidationError("At least one basis function must be selected")

            return func_key, min_price, max_price, n_options, pricing_params

        except ValidationError as e:
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected validation error: {str(e)}")

    def use_custom_function(self):
        """Use custom function from text entry with better error handling."""
        try:
            code = self.custom_func_entry.get("1.0", tk.END).strip()
            if not code or code.startswith("#"):
                self.show_warning("Please enter a valid function expression")
                return

            # Create a function from the code
            def custom_func(x):
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
                try:
                    result = eval(code, namespace)
                    if not isinstance(result, np.ndarray):
                        result = np.array(result)
                    return result
                except Exception as e:
                    raise ValueError(f"Function evaluation error: {str(e)}")

            # Test the function
            test_x = np.array([1.0, 2.0, 3.0])
            test_result = custom_func(test_x)

            if not isinstance(test_result, np.ndarray):
                raise ValueError("Function must return a numpy array")

            if len(test_result) != len(test_x):
                raise ValueError("Function output length must match input length")

            self.function_dict["custom"] = ("Custom Function", custom_func)
            self.function_var.set("custom")
            self.update_status(
                "Custom function loaded successfully", ModernStyle.ACCENT_SUCCESS
            )

        except Exception as e:
            error_msg = f"Invalid function: {str(e)}\n\nMake sure to use numpy functions (np.sin, np.exp, etc.)"
            self.show_error("Custom Function Error", error_msg)
            self.update_status(
                "Error loading custom function", ModernStyle.STATUS_ERROR
            )

    def calculate_approximation(self):
        """Calculate the approximation in a separate thread."""
        if self.is_calculating:
            self.show_warning("Calculation already in progress. Please wait.")
            return

        try:
            # Validate inputs
            func_key, min_price, max_price, n_options, pricing_params = (
                self.validate_inputs()
            )

            # Get function
            _, target_func = self.function_dict[func_key]
            self.current_function = target_func

            # Disable button and show progress
            self.is_calculating = True
            self.calculate_btn.config(state="disabled")
            self.update_status("Calculating...", ModernStyle.STATUS_INFO)
            self.progress_var.set("Initializing approximation...")
            self.root.update()

            # Run calculation in separate thread
            self.calculation_thread = threading.Thread(
                target=self._calculate_approximation_thread,
                args=(target_func, min_price, max_price, n_options, pricing_params),
                daemon=True,
            )
            self.calculation_thread.start()

        except ValidationError as e:
            self.show_error("Validation Error", str(e))
            self.is_calculating = False
            self.calculate_btn.config(state="normal")
        except Exception as e:
            self.show_error(
                "Error", f"Unexpected error: {str(e)}\n\n{traceback.format_exc()}"
            )
            self.is_calculating = False
            self.calculate_btn.config(state="normal")

    def _calculate_approximation_thread(
        self, target_func, min_price, max_price, n_options, pricing_params
    ):
        """Perform calculation in background thread."""
        try:
            # Update progress
            self.root.after(
                0, lambda: self.progress_var.set("Creating approximator...")
            )

            # Create approximator
            approximator = OptionsFunctionApproximator(
                n_options=n_options,
                price_range=(min_price, max_price),
                use_calls=self.use_calls_var.get(),
                use_puts=self.use_puts_var.get(),
                use_stock=self.use_stock_var.get(),
                **pricing_params,
            )

            # Update progress
            self.root.after(
                0, lambda: self.progress_var.set("Computing approximation...")
            )

            # Calculate approximation
            weights, mse = approximator.approximate(
                target_func, n_points=1000, regularization=0.001
            )

            # Calculate additional metrics
            self.root.after(
                0, lambda: self.progress_var.set("Calculating premiums and Greeks...")
            )
            premiums, premium_details = approximator.calculate_premiums()
            greeks = approximator.calculate_portfolio_greeks()

            # Store results
            self.approximator = approximator
            self.approximation_error = {"mse": mse, "rmse": np.sqrt(mse)}
            self.premium_details = premium_details
            self.greeks = greeks

            # Update UI in main thread
            self.root.after(0, self._update_ui_after_calculation)

        except Exception as e:
            error_msg = f"Calculation failed: {str(e)}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self.show_error("Calculation Error", error_msg))
            self.root.after(
                0, lambda: self.update_status("Error", ModernStyle.STATUS_ERROR)
            )
            self.root.after(0, lambda: self.calculate_btn.config(state="normal"))
            self.is_calculating = False

    def _update_ui_after_calculation(self):
        """Update UI after calculation completes."""
        try:
            # Update plot
            if self.current_function:
                min_price = float(self.min_price_var.get())
                max_price = float(self.max_price_var.get())
                self.update_plot(self.current_function, min_price, max_price)

            # Update all display tabs
            self.update_summary_tab()
            self.update_cost_tab()
            self.update_greeks_tab()
            self.update_detail_tab()

            # Reset UI state
            self.progress_var.set("")
            self.update_status("Calculation complete!", ModernStyle.STATUS_SUCCESS)
            self.calculate_btn.config(state="normal")
            self.is_calculating = False

        except Exception as e:
            self.show_error("UI Update Error", f"Error updating display: {str(e)}")
            self.calculate_btn.config(state="normal")
            self.is_calculating = False

    def update_plot(self, target_func, min_price, max_price):
        """Update the plot with approximation."""
        try:
            self.ax.clear()

            stock_prices = np.linspace(min_price, max_price, 1000)
            target_values = target_func(stock_prices)

            if self.approximator is not None:
                approx_values = self.approximator.evaluate(stock_prices)

                # Modern professional chart colors
                self.ax.plot(
                    stock_prices,
                    target_values,
                    color=ModernStyle.CHART_LINE_1,
                    linestyle="-",
                    label="Target Function",
                    linewidth=2.5,
                    alpha=0.9,
                )
                self.ax.plot(
                    stock_prices,
                    approx_values,
                    color=ModernStyle.CHART_LINE_2,
                    linestyle="--",
                    label="Options Approximation",
                    linewidth=2,
                    alpha=0.9,
                )

                # Modern legend styling
                legend = self.ax.legend(loc="best", fontsize=10, framealpha=0.95)
                legend.get_frame().set_facecolor(ModernStyle.BG_CARD)
                legend.get_frame().set_edgecolor(ModernStyle.BORDER_COLOR)
                legend.get_frame().set_linewidth(1)
                for text in legend.get_texts():
                    text.set_color(ModernStyle.TEXT_PRIMARY)

                self.ax.set_xlabel(
                    "Stock Price", fontsize=11, color=ModernStyle.TEXT_PRIMARY
                )
                self.ax.set_ylabel(
                    "Function Value", fontsize=11, color=ModernStyle.TEXT_PRIMARY
                )
                self.ax.set_title(
                    "Target Function vs. Options Approximation",
                    fontsize=12,
                    fontweight="bold",
                    color=ModernStyle.TEXT_PRIMARY,
                    pad=12,
                )
                self.ax.tick_params(colors=ModernStyle.CHART_AXIS, labelsize=9)
                self.ax.grid(
                    True,
                    alpha=0.2,
                    linestyle="-",
                    color=ModernStyle.CHART_GRID,
                    linewidth=0.5,
                )
                self.ax.set_facecolor(ModernStyle.BG_CARD)

                # Modern spine styling
                for spine in self.ax.spines.values():
                    spine.set_color(ModernStyle.BORDER_COLOR)
                    spine.set_linewidth(1)

                # Modern error info box
                if self.approximation_error:
                    rmse = self.approximation_error["rmse"]
                    self.ax.text(
                        0.02,
                        0.98,
                        f"RMSE: {rmse:.6f}",
                        transform=self.ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor=ModernStyle.BG_CARD,
                            edgecolor=ModernStyle.BORDER_COLOR,
                            alpha=0.95,
                            linewidth=1,
                        ),
                        fontsize=9,
                        color=ModernStyle.TEXT_PRIMARY,
                        weight="normal",
                    )
            else:
                self.ax.text(
                    0.5,
                    0.5,
                    "No data to display",
                    transform=self.ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=ModernStyle.TEXT_SECONDARY,
                )

            self.canvas.draw()

        except Exception as e:
            self.show_error("Plot Error", f"Error updating plot: {str(e)}")

    def update_summary_tab(self):
        """Update summary tab with metrics."""
        try:
            if self.approximation_error:
                self.mse_label.config(
                    text=f"MSE: {self.approximation_error['mse']:.6f}"
                )
                self.rmse_label.config(
                    text=f"RMSE: {self.approximation_error['rmse']:.6f}"
                )

            if self.approximator:
                total_cost = self.approximator.get_total_cost()
                self.total_cost_label.config(text=f"Total Cost: ${total_cost:,.2f}")

                if self.premium_details:
                    num_calls = len(self.premium_details.get("calls", []))
                    num_puts = len(self.premium_details.get("puts", []))
                    stock_cost = self.premium_details.get("stock_cost", 0.0)

                    comp_text = f"• Call Options: {num_calls}\n"
                    comp_text += f"• Put Options: {num_puts}\n"
                    comp_text += f"• Stock Position Cost: ${stock_cost:,.2f}\n"
                    comp_text += f"• Total Options: {num_calls + num_puts}"

                    self.comp_label.config(text=comp_text)
        except Exception as e:
            self.show_error("Summary Update Error", str(e))

    def update_cost_tab(self):
        """Update cost breakdown tab."""
        try:
            # Clear existing items
            for item in self.calls_tree.get_children():
                self.calls_tree.delete(item)
            for item in self.puts_tree.get_children():
                self.puts_tree.delete(item)

            if self.premium_details:
                # Add call options
                for call in sorted(
                    self.premium_details.get("calls", []), key=lambda x: x["strike"]
                ):
                    self.calls_tree.insert(
                        "",
                        tk.END,
                        values=(
                            f"${call['strike']:.2f}",
                            f"{call['weight']:.4f}",
                            f"${call['premium_per_unit']:.2f}",
                            f"${call['total_cost']:.2f}",
                        ),
                    )

                # Add put options
                for put in sorted(
                    self.premium_details.get("puts", []), key=lambda x: x["strike"]
                ):
                    self.puts_tree.insert(
                        "",
                        tk.END,
                        values=(
                            f"${put['strike']:.2f}",
                            f"{put['weight']:.4f}",
                            f"${put['premium_per_unit']:.2f}",
                            f"${put['total_cost']:.2f}",
                        ),
                    )
        except Exception as e:
            self.show_error("Cost Tab Update Error", str(e))

    def update_greeks_tab(self):
        """Update portfolio Greeks tab."""
        try:
            if self.greeks:
                self.greek_labels["delta"].config(text=f"{self.greeks['delta']:.4f}")
                self.greek_labels["gamma"].config(text=f"{self.greeks['gamma']:.4f}")
                self.greek_labels["vega"].config(text=f"{self.greeks['vega']:.2f}")
                self.greek_labels["theta"].config(text=f"{self.greeks['theta']:.4f}")
        except Exception as e:
            self.show_error("Greeks Update Error", str(e))

    def update_detail_tab(self):
        """Update detailed text output tab."""
        try:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)

            if self.approximator is None:
                return

            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            try:
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

                self.approximator.print_cost_breakdown()
                output = buffer.getvalue()
                self.output_text.insert(tk.END, output)

            finally:
                sys.stdout = old_stdout

            self.output_text.config(state=tk.DISABLED)

        except Exception as e:
            self.show_error("Detail Tab Update Error", str(e))

    def update_status(self, message: str, color: str = ModernStyle.STATUS_SUCCESS):
        """Update status label."""
        self.status_var.set(message)
        self.status_label.config(fg=color)

    def show_error(self, title: str, message: str):
        """Show error message box."""
        messagebox.showerror(title, message)
        self.update_status("Error", ModernStyle.STATUS_ERROR)

    def show_warning(self, message: str):
        """Show warning message box."""
        messagebox.showwarning("Warning", message)

    def on_closing(self):
        """Handle window closing."""
        if self.is_calculating:
            if messagebox.askokcancel("Quit", "Calculation in progress. Quit anyway?"):
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Main entry point."""
    try:
        root = tk.Tk()
        app = OptionsApproximatorGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror(
            "Fatal Error",
            f"Application failed to start: {str(e)}\n\n{traceback.format_exc()}",
        )


if __name__ == "__main__":
    main()
