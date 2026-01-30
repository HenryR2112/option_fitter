"""
Professional GUI for Options-Based Function Approximation

A modern, polished tkinter-based interface inspired by professional trading terminals.
Features include interactive charts, real-time validation, export capabilities,
and a clean, dark theme optimized for financial applications.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import traceback
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import csv
import io
import sys

from options_func_maker import (
    OptionsFunctionApproximator,
    sigmoid,
    gaussian_pdf,
    volatility_smile,
)


class Theme:
    """Professional dark theme - inspired by modern trading terminals."""

    # Core backgrounds
    BG_DARK = "#0d1117"       # Darkest background
    BG_PRIMARY = "#161b22"    # Main panels
    BG_SECONDARY = "#21262d"  # Cards and elevated elements
    BG_TERTIARY = "#30363d"   # Input fields, hover states
    BG_HOVER = "#2d333b"      # Subtle hover highlight

    # Text hierarchy
    TEXT_PRIMARY = "#f0f6fc"   # Primary text
    TEXT_SECONDARY = "#8b949e" # Secondary text
    TEXT_MUTED = "#6e7681"     # Muted/disabled text
    TEXT_LINK = "#58a6ff"      # Links and interactive text

    # Accent colors
    ACCENT_BLUE = "#58a6ff"    # Primary actions
    ACCENT_GREEN = "#3fb950"   # Success, positive values
    ACCENT_RED = "#f85149"     # Errors, negative values
    ACCENT_YELLOW = "#d29922"  # Warnings
    ACCENT_PURPLE = "#a371f7"  # Special highlights
    ACCENT_CYAN = "#39c5cf"    # Info

    # Chart colors
    CHART_PRIMARY = "#58a6ff"   # Main series
    CHART_SECONDARY = "#f85149" # Comparison series
    CHART_GRID = "#30363d"      # Grid lines
    CHART_BG = "#0d1117"        # Chart background

    # Borders
    BORDER_DEFAULT = "#30363d"
    BORDER_FOCUS = "#58a6ff"
    BORDER_MUTED = "#21262d"

    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_MONO = "Consolas"

    @classmethod
    def configure_matplotlib(cls):
        """Configure matplotlib for dark theme."""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': cls.CHART_BG,
            'axes.facecolor': cls.BG_PRIMARY,
            'axes.edgecolor': cls.BORDER_DEFAULT,
            'axes.labelcolor': cls.TEXT_PRIMARY,
            'axes.titlecolor': cls.TEXT_PRIMARY,
            'xtick.color': cls.TEXT_SECONDARY,
            'ytick.color': cls.TEXT_SECONDARY,
            'text.color': cls.TEXT_PRIMARY,
            'grid.color': cls.CHART_GRID,
            'grid.alpha': 0.3,
            'legend.facecolor': cls.BG_SECONDARY,
            'legend.edgecolor': cls.BORDER_DEFAULT,
            'font.family': 'sans-serif',
            'font.sans-serif': [cls.FONT_FAMILY, 'Arial', 'Helvetica'],
        })


class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


class StyledEntry(tk.Entry):
    """Custom styled entry widget with validation feedback."""

    def __init__(self, parent, **kwargs):
        self.var = kwargs.pop('textvariable', tk.StringVar())
        super().__init__(
            parent,
            textvariable=self.var,
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.ACCENT_BLUE,
            relief='flat',
            highlightthickness=1,
            highlightbackground=Theme.BORDER_DEFAULT,
            highlightcolor=Theme.BORDER_FOCUS,
            font=(Theme.FONT_MONO, 10),
            **kwargs
        )
        self.bind('<FocusIn>', self._on_focus_in)
        self.bind('<FocusOut>', self._on_focus_out)

    def _on_focus_in(self, event):
        self.configure(highlightbackground=Theme.BORDER_FOCUS)

    def _on_focus_out(self, event):
        self.configure(highlightbackground=Theme.BORDER_DEFAULT)

    def set_error(self, is_error: bool):
        """Visual feedback for validation errors."""
        if is_error:
            self.configure(highlightbackground=Theme.ACCENT_RED, highlightcolor=Theme.ACCENT_RED)
        else:
            self.configure(highlightbackground=Theme.BORDER_DEFAULT, highlightcolor=Theme.BORDER_FOCUS)


class Card(tk.Frame):
    """A styled card container with optional title."""

    def __init__(self, parent, title: str = None, **kwargs):
        super().__init__(parent, bg=Theme.BG_SECONDARY, **kwargs)

        self.configure(
            highlightthickness=1,
            highlightbackground=Theme.BORDER_MUTED,
            highlightcolor=Theme.BORDER_MUTED,
        )

        if title:
            title_frame = tk.Frame(self, bg=Theme.BG_SECONDARY)
            title_frame.pack(fill='x', padx=12, pady=(12, 8))

            tk.Label(
                title_frame,
                text=title,
                font=(Theme.FONT_FAMILY, 11, 'bold'),
                fg=Theme.TEXT_PRIMARY,
                bg=Theme.BG_SECONDARY
            ).pack(side='left')

        self.content = tk.Frame(self, bg=Theme.BG_SECONDARY)
        self.content.pack(fill='both', expand=True, padx=12, pady=(0, 12))


class MetricDisplay(tk.Frame):
    """A styled metric display widget."""

    def __init__(self, parent, label: str, initial_value: str = "-", **kwargs):
        super().__init__(parent, bg=Theme.BG_SECONDARY, **kwargs)

        tk.Label(
            self,
            text=label,
            font=(Theme.FONT_FAMILY, 9),
            fg=Theme.TEXT_SECONDARY,
            bg=Theme.BG_SECONDARY
        ).pack(anchor='w')

        self.value_label = tk.Label(
            self,
            text=initial_value,
            font=(Theme.FONT_FAMILY, 16, 'bold'),
            fg=Theme.ACCENT_BLUE,
            bg=Theme.BG_SECONDARY
        )
        self.value_label.pack(anchor='w', pady=(2, 0))

    def set_value(self, value: str, color: str = None):
        self.value_label.configure(text=value)
        if color:
            self.value_label.configure(fg=color)


class OptionsApproximatorGUI:
    """Main application class for the Options Function Approximator."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Options Function Approximator")
        self.root.configure(bg=Theme.BG_DARK)

        # Configure for fullscreen with proper sizing
        self.root.state('zoomed')
        self.root.minsize(1200, 800)

        # Application state
        self.approximator: Optional[OptionsFunctionApproximator] = None
        self.current_function: Optional[Callable] = None
        self.approximation_error: Optional[Dict[str, float]] = None
        self.premium_details: Optional[Dict[str, Any]] = None
        self.greeks: Optional[Dict[str, float]] = None
        self.is_calculating = False
        self.calculation_thread: Optional[threading.Thread] = None

        # Initialize
        Theme.configure_matplotlib()
        self._setup_styles()
        self._build_ui()

        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.bind('<Control-e>', lambda e: self._export_results())
        self.root.bind('<Control-r>', lambda e: self._calculate_approximation())

    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')

        # Notebook tabs
        style.configure('TNotebook', background=Theme.BG_DARK, borderwidth=0)
        style.configure('TNotebook.Tab',
            background=Theme.BG_SECONDARY,
            foreground=Theme.TEXT_SECONDARY,
            padding=[16, 8],
            font=(Theme.FONT_FAMILY, 9)
        )
        style.map('TNotebook.Tab',
            background=[('selected', Theme.BG_PRIMARY)],
            foreground=[('selected', Theme.ACCENT_BLUE)]
        )

        # Treeview
        style.configure('Treeview',
            background=Theme.BG_PRIMARY,
            foreground=Theme.TEXT_PRIMARY,
            fieldbackground=Theme.BG_PRIMARY,
            borderwidth=0,
            font=(Theme.FONT_MONO, 9)
        )
        style.configure('Treeview.Heading',
            background=Theme.BG_SECONDARY,
            foreground=Theme.TEXT_PRIMARY,
            font=(Theme.FONT_FAMILY, 9, 'bold')
        )
        style.map('Treeview',
            background=[('selected', Theme.BG_HOVER)],
            foreground=[('selected', Theme.TEXT_PRIMARY)]
        )

        # Scrollbar
        style.configure('TScrollbar',
            background=Theme.BG_SECONDARY,
            troughcolor=Theme.BG_PRIMARY,
            borderwidth=0,
            arrowcolor=Theme.TEXT_SECONDARY
        )

    def _build_ui(self):
        """Build the main user interface."""
        # Main container
        main_container = tk.Frame(self.root, bg=Theme.BG_DARK)
        main_container.pack(fill='both', expand=True)

        # Header/Toolbar
        self._build_header(main_container)

        # Content area with sidebar and main panel
        content = tk.Frame(main_container, bg=Theme.BG_DARK)
        content.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        # Left sidebar (controls)
        sidebar = tk.Frame(content, bg=Theme.BG_DARK, width=340)
        sidebar.pack(side='left', fill='y', padx=(0, 8))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Main panel (chart and results)
        main_panel = tk.Frame(content, bg=Theme.BG_DARK)
        main_panel.pack(side='left', fill='both', expand=True)
        self._build_main_panel(main_panel)

    def _build_header(self, parent):
        """Build the application header with toolbar."""
        header = tk.Frame(parent, bg=Theme.BG_PRIMARY, height=56)
        header.pack(fill='x', padx=8, pady=8)
        header.pack_propagate(False)

        # Left side - Title and status
        left_frame = tk.Frame(header, bg=Theme.BG_PRIMARY)
        left_frame.pack(side='left', fill='y', padx=16)

        title_label = tk.Label(
            left_frame,
            text="Options Function Approximator",
            font=(Theme.FONT_FAMILY, 14, 'bold'),
            fg=Theme.TEXT_PRIMARY,
            bg=Theme.BG_PRIMARY
        )
        title_label.pack(side='left', pady=12)

        # Separator
        sep = tk.Frame(left_frame, bg=Theme.BORDER_DEFAULT, width=1)
        sep.pack(side='left', fill='y', padx=16, pady=12)

        # Status indicator
        self.status_dot = tk.Canvas(left_frame, width=10, height=10, bg=Theme.BG_PRIMARY, highlightthickness=0)
        self.status_dot.pack(side='left', pady=12)
        self.status_dot.create_oval(2, 2, 8, 8, fill=Theme.ACCENT_GREEN, outline='')

        self.status_label = tk.Label(
            left_frame,
            text="Ready",
            font=(Theme.FONT_FAMILY, 10),
            fg=Theme.TEXT_SECONDARY,
            bg=Theme.BG_PRIMARY
        )
        self.status_label.pack(side='left', padx=(6, 0), pady=12)

        # Right side - Action buttons
        right_frame = tk.Frame(header, bg=Theme.BG_PRIMARY)
        right_frame.pack(side='right', fill='y', padx=16)

        # Export button
        export_btn = tk.Button(
            right_frame,
            text="Export",
            font=(Theme.FONT_FAMILY, 9),
            fg=Theme.TEXT_PRIMARY,
            bg=Theme.BG_TERTIARY,
            activebackground=Theme.BG_TERTIARY,
            activeforeground=Theme.TEXT_PRIMARY,
            highlightthickness=0,
            highlightbackground=Theme.BG_TERTIARY,
            bd=0,
            padx=16,
            pady=6,
            command=self._export_results
        )
        export_btn.pack(side='left', pady=12, padx=(0, 8))

        # Calculate button (primary action)
        self.calculate_btn = tk.Button(
            right_frame,
            text="Calculate",
            font=(Theme.FONT_FAMILY, 9, 'bold'),
            fg='#ffffff',
            bg=Theme.ACCENT_BLUE,
            activebackground=Theme.ACCENT_BLUE,
            activeforeground='#ffffff',
            highlightthickness=0,
            highlightbackground=Theme.ACCENT_BLUE,
            bd=0,
            padx=20,
            pady=6,
            command=self._calculate_approximation
        )
        self.calculate_btn.pack(side='left', pady=12)

    def _build_sidebar(self, parent):
        """Build the left sidebar with controls."""
        # Scrollable container
        canvas = tk.Canvas(parent, bg=Theme.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=Theme.BG_DARK)

        scrollable.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable, anchor='nw', width=332)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel)

        # Function Selection Card
        func_card = Card(scrollable, title="Target Function")
        func_card.pack(fill='x', pady=(0, 8))

        self.function_var = tk.StringVar(value='sin')
        self.functions = {
            'sin': ('Sine Wave', 'sin(x)', lambda x: np.sin(x)),
            'cos': ('Cosine Wave', 'cos(x)', lambda x: np.cos(x)),
            'sigmoid': ('Sigmoid', '1/(1+e^(-x))', lambda x: sigmoid(x, center=5.0, scale=2.0)),
            'gaussian': ('Gaussian PDF', 'Normal distribution', lambda x: gaussian_pdf(x, mu=5.0, sigma=1.0)),
            'polynomial': ('Polynomial', 'x³ - 2x² + x', lambda x: x**3 - 2*x**2 + x),
            'bull_spread': ('Bull Spread', 'Call spread payoff', lambda x: np.maximum(x - 90, 0) - np.maximum(x - 110, 0)),
            'butterfly': ('Butterfly', 'Butterfly spread', lambda x: np.maximum(x - 95, 0) - 2*np.maximum(x - 100, 0) + np.maximum(x - 105, 0)),
        }

        for key, (name, _, _) in self.functions.items():
            frame = tk.Frame(func_card.content, bg=Theme.BG_SECONDARY)
            frame.pack(fill='x', pady=2)

            rb = tk.Radiobutton(
                frame,
                text=name,
                variable=self.function_var,
                value=key,
                bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_PRIMARY,
                selectcolor=Theme.BG_PRIMARY,
                activebackground=Theme.BG_SECONDARY,
                activeforeground=Theme.TEXT_PRIMARY,
                highlightthickness=0,
                font=(Theme.FONT_FAMILY, 10),
                anchor='w',
                command=self._on_function_change
            )
            rb.pack(side='left', fill='x', expand=True)

        # Custom function
        tk.Label(
            func_card.content,
            text="Custom Function (Python expr)",
            font=(Theme.FONT_FAMILY, 9),
            fg=Theme.TEXT_SECONDARY,
            bg=Theme.BG_SECONDARY
        ).pack(anchor='w', pady=(12, 4))

        self.custom_entry = tk.Text(
            func_card.content,
            height=2,
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.ACCENT_BLUE,
            relief='flat',
            font=(Theme.FONT_MONO, 9),
            padx=8,
            pady=6
        )
        self.custom_entry.pack(fill='x')
        self.custom_entry.insert('1.0', 'np.sin(x) + 0.5*x')

        custom_btn = tk.Button(
            func_card.content,
            text="Use Custom",
            font=(Theme.FONT_FAMILY, 9),
            fg=Theme.TEXT_PRIMARY,
            bg=Theme.BG_TERTIARY,
            activebackground=Theme.BG_TERTIARY,
            activeforeground=Theme.TEXT_PRIMARY,
            highlightthickness=0,
            highlightbackground=Theme.BG_TERTIARY,
            bd=0,
            padx=12,
            pady=4,
            command=self._use_custom_function
        )
        custom_btn.pack(anchor='e', pady=(8, 0))

        # Price Range Card
        range_card = Card(scrollable, title="Price Range")
        range_card.pack(fill='x', pady=(0, 8))

        self.range_vars = {}
        range_params = [
            ('min_price', 'Minimum Price', '0'),
            ('max_price', 'Maximum Price', '10'),
            ('n_options', 'Number of Options', '15'),
        ]

        for key, label, default in range_params:
            frame = tk.Frame(range_card.content, bg=Theme.BG_SECONDARY)
            frame.pack(fill='x', pady=4)

            tk.Label(
                frame, text=label,
                font=(Theme.FONT_FAMILY, 9),
                fg=Theme.TEXT_SECONDARY,
                bg=Theme.BG_SECONDARY,
                width=18,
                anchor='w'
            ).pack(side='left')

            var = tk.StringVar(value=default)
            self.range_vars[key] = var

            entry = StyledEntry(frame, textvariable=var, width=12)
            entry.pack(side='right')

        # Pricing Parameters Card
        pricing_card = Card(scrollable, title="Pricing Parameters")
        pricing_card.pack(fill='x', pady=(0, 8))

        self.pricing_vars = {}
        pricing_params = [
            ('S0', 'Stock Price (S₀)', '100.0'),
            ('r', 'Risk-free Rate (r)', '0.05'),
            ('T', 'Time to Expiry (T)', '0.25'),
            ('sigma', 'Volatility (σ)', '0.2'),
        ]

        for key, label, default in pricing_params:
            frame = tk.Frame(pricing_card.content, bg=Theme.BG_SECONDARY)
            frame.pack(fill='x', pady=4)

            tk.Label(
                frame, text=label,
                font=(Theme.FONT_FAMILY, 9),
                fg=Theme.TEXT_SECONDARY,
                bg=Theme.BG_SECONDARY,
                width=18,
                anchor='w'
            ).pack(side='left')

            var = tk.StringVar(value=default)
            self.pricing_vars[key] = var

            entry = StyledEntry(frame, textvariable=var, width=12)
            entry.pack(side='right')

        # Basis Functions Card
        basis_card = Card(scrollable, title="Basis Functions")
        basis_card.pack(fill='x', pady=(0, 8))

        self.use_calls_var = tk.BooleanVar(value=True)
        self.use_puts_var = tk.BooleanVar(value=True)
        self.use_stock_var = tk.BooleanVar(value=True)

        for var, text in [
            (self.use_calls_var, 'Call Options'),
            (self.use_puts_var, 'Put Options'),
            (self.use_stock_var, 'Stock Position'),
        ]:
            cb = tk.Checkbutton(
                basis_card.content,
                text=text,
                variable=var,
                font=(Theme.FONT_FAMILY, 10),
                fg=Theme.TEXT_PRIMARY,
                bg=Theme.BG_SECONDARY,
                selectcolor=Theme.BG_TERTIARY,
                activebackground=Theme.BG_SECONDARY,
                activeforeground=Theme.ACCENT_BLUE,
                highlightthickness=0
            )
            cb.pack(anchor='w', pady=2)

    def _build_main_panel(self, parent):
        """Build the main content panel with chart and results."""
        # Chart area (top)
        chart_frame = tk.Frame(parent, bg=Theme.BG_PRIMARY)
        chart_frame.pack(fill='both', expand=True, pady=(0, 8))

        # Chart header
        chart_header = tk.Frame(chart_frame, bg=Theme.BG_PRIMARY)
        chart_header.pack(fill='x', padx=16, pady=(12, 0))

        tk.Label(
            chart_header,
            text="Function Approximation",
            font=(Theme.FONT_FAMILY, 12, 'bold'),
            fg=Theme.TEXT_PRIMARY,
            bg=Theme.BG_PRIMARY
        ).pack(side='left')

        # Chart
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor=Theme.CHART_BG)
        self.ax = self.fig.add_subplot(111, facecolor=Theme.BG_PRIMARY)
        self._style_axes(self.ax)

        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=8)

        # Matplotlib toolbar
        toolbar_frame = tk.Frame(chart_frame, bg=Theme.BG_PRIMARY)
        toolbar_frame.pack(fill='x', padx=8)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.configure(bg=Theme.BG_PRIMARY)
        self.toolbar.update()

        # Results area (bottom)
        results_frame = tk.Frame(parent, bg=Theme.BG_DARK, height=280)
        results_frame.pack(fill='x')
        results_frame.pack_propagate(False)

        # Results notebook
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill='both', expand=True)

        self._build_summary_tab()
        self._build_options_tab()
        self._build_greeks_tab()
        self._build_details_tab()

    def _style_axes(self, ax):
        """Apply consistent styling to axes."""
        ax.set_xlabel('Stock Price', fontsize=10, color=Theme.TEXT_PRIMARY)
        ax.set_ylabel('Value', fontsize=10, color=Theme.TEXT_PRIMARY)
        ax.tick_params(colors=Theme.TEXT_SECONDARY, labelsize=9)
        ax.grid(True, alpha=0.2, color=Theme.CHART_GRID)
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER_DEFAULT)

    def _build_summary_tab(self):
        """Build the summary metrics tab."""
        frame = tk.Frame(self.notebook, bg=Theme.BG_PRIMARY)
        self.notebook.add(frame, text='Summary')

        # Metrics row
        metrics_frame = tk.Frame(frame, bg=Theme.BG_PRIMARY)
        metrics_frame.pack(fill='x', padx=16, pady=16)

        # Create metric displays
        self.metrics = {}
        metrics_config = [
            ('mse', 'Mean Squared Error'),
            ('rmse', 'Root MSE'),
            ('total_cost', 'Total Portfolio Cost'),
            ('num_options', 'Total Options'),
        ]

        for i, (key, label) in enumerate(metrics_config):
            metric = MetricDisplay(metrics_frame, label)
            metric.pack(side='left', padx=(0, 40))
            self.metrics[key] = metric

    def _build_options_tab(self):
        """Build the options breakdown tab."""
        frame = tk.Frame(self.notebook, bg=Theme.BG_PRIMARY)
        self.notebook.add(frame, text='Options Breakdown')

        # Split into calls and puts
        paned = tk.PanedWindow(frame, orient='horizontal', bg=Theme.BG_PRIMARY, sashwidth=4)
        paned.pack(fill='both', expand=True, padx=8, pady=8)

        # Calls section
        calls_frame = tk.Frame(paned, bg=Theme.BG_PRIMARY)
        paned.add(calls_frame)

        tk.Label(
            calls_frame,
            text="Call Options",
            font=(Theme.FONT_FAMILY, 10, 'bold'),
            fg=Theme.ACCENT_GREEN,
            bg=Theme.BG_PRIMARY
        ).pack(anchor='w', padx=8, pady=(8, 4))

        self.calls_tree = self._create_options_tree(calls_frame)

        # Puts section
        puts_frame = tk.Frame(paned, bg=Theme.BG_PRIMARY)
        paned.add(puts_frame)

        tk.Label(
            puts_frame,
            text="Put Options",
            font=(Theme.FONT_FAMILY, 10, 'bold'),
            fg=Theme.ACCENT_RED,
            bg=Theme.BG_PRIMARY
        ).pack(anchor='w', padx=8, pady=(8, 4))

        self.puts_tree = self._create_options_tree(puts_frame)

    def _create_options_tree(self, parent) -> ttk.Treeview:
        """Create a styled treeview for options display."""
        tree_frame = tk.Frame(parent, bg=Theme.BG_PRIMARY)
        tree_frame.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        columns = ('Strike', 'Weight', 'Premium', 'Cost')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=6)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=90, anchor='center')

        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        return tree

    def _build_greeks_tab(self):
        """Build the portfolio Greeks tab."""
        frame = tk.Frame(self.notebook, bg=Theme.BG_PRIMARY)
        self.notebook.add(frame, text='Portfolio Greeks')

        greeks_frame = tk.Frame(frame, bg=Theme.BG_PRIMARY)
        greeks_frame.pack(fill='x', padx=16, pady=16)

        self.greek_displays = {}
        greeks_config = [
            ('delta', 'Delta', 'Price sensitivity'),
            ('gamma', 'Gamma', 'Delta sensitivity'),
            ('vega', 'Vega', 'Volatility sensitivity'),
            ('theta', 'Theta', 'Time decay'),
        ]

        for key, name, desc in greeks_config:
            card = tk.Frame(greeks_frame, bg=Theme.BG_SECONDARY, padx=16, pady=12)
            card.pack(side='left', padx=(0, 12))

            tk.Label(
                card, text=name,
                font=(Theme.FONT_FAMILY, 11, 'bold'),
                fg=Theme.TEXT_PRIMARY,
                bg=Theme.BG_SECONDARY
            ).pack(anchor='w')

            tk.Label(
                card, text=desc,
                font=(Theme.FONT_FAMILY, 8),
                fg=Theme.TEXT_MUTED,
                bg=Theme.BG_SECONDARY
            ).pack(anchor='w')

            value_label = tk.Label(
                card, text='-',
                font=(Theme.FONT_MONO, 18, 'bold'),
                fg=Theme.ACCENT_CYAN,
                bg=Theme.BG_SECONDARY
            )
            value_label.pack(anchor='w', pady=(8, 0))

            self.greek_displays[key] = value_label

    def _build_details_tab(self):
        """Build the detailed output tab."""
        frame = tk.Frame(self.notebook, bg=Theme.BG_PRIMARY)
        self.notebook.add(frame, text='Details')

        self.details_text = scrolledtext.ScrolledText(
            frame,
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.ACCENT_BLUE,
            font=(Theme.FONT_MONO, 9),
            relief='flat',
            padx=12,
            pady=12
        )
        self.details_text.pack(fill='both', expand=True, padx=8, pady=8)

    def _on_function_change(self):
        """Handle function selection change."""
        func_key = self.function_var.get()

        # Update default ranges based on function
        ranges = {
            'sin': ('0', str(round(2 * np.pi, 4))),
            'cos': ('0', str(round(2 * np.pi, 4))),
            'sigmoid': ('0', '10'),
            'gaussian': ('0', '10'),
            'polynomial': ('0', '10'),
            'bull_spread': ('80', '120'),
            'butterfly': ('85', '115'),
        }

        if func_key in ranges:
            min_val, max_val = ranges[func_key]
            self.range_vars['min_price'].set(min_val)
            self.range_vars['max_price'].set(max_val)

    def _use_custom_function(self):
        """Parse and use custom function."""
        code = self.custom_entry.get('1.0', 'end').strip()

        try:
            def custom_func(x):
                namespace = {
                    'np': np, 'x': x,
                    'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
                    'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
                    'maximum': np.maximum, 'minimum': np.minimum,
                    '__builtins__': {}
                }
                return eval(code, namespace)

            # Test
            test_x = np.array([1.0, 2.0, 3.0])
            result = custom_func(test_x)
            if len(result) != len(test_x):
                raise ValueError("Function must return array of same length")

            self.functions['custom'] = ('Custom', code, custom_func)
            self.function_var.set('custom')
            self._update_status('Custom function loaded', Theme.ACCENT_GREEN)

        except Exception as e:
            messagebox.showerror('Invalid Function', f'Error: {str(e)}')
            self._update_status('Error loading function', Theme.ACCENT_RED)

    def _validate_inputs(self) -> tuple:
        """Validate all inputs and return parameters."""
        try:
            min_price = float(self.range_vars['min_price'].get())
            max_price = float(self.range_vars['max_price'].get())
            n_options = int(self.range_vars['n_options'].get())

            if min_price >= max_price:
                raise ValidationError('Min price must be less than max price')
            if n_options < 1 or n_options > 100:
                raise ValidationError('Number of options must be 1-100')

            pricing = {
                'S0': float(self.pricing_vars['S0'].get()),
                'r': float(self.pricing_vars['r'].get()),
                'T': float(self.pricing_vars['T'].get()),
                'sigma': float(self.pricing_vars['sigma'].get()),
            }

            if pricing['S0'] <= 0:
                raise ValidationError('Stock price must be positive')
            if pricing['sigma'] <= 0:
                raise ValidationError('Volatility must be positive')
            if pricing['T'] <= 0:
                raise ValidationError('Time to expiry must be positive')

            if not (self.use_calls_var.get() or self.use_puts_var.get() or self.use_stock_var.get()):
                raise ValidationError('Select at least one basis function')

            return min_price, max_price, n_options, pricing

        except ValueError as e:
            raise ValidationError(f'Invalid numeric input: {str(e)}')

    def _calculate_approximation(self):
        """Start the approximation calculation."""
        if self.is_calculating:
            return

        try:
            min_price, max_price, n_options, pricing = self._validate_inputs()

            func_key = self.function_var.get()
            if func_key not in self.functions:
                raise ValidationError('Please select a function')

            _, _, target_func = self.functions[func_key]
            self.current_function = target_func

            self.is_calculating = True
            self.calculate_btn.configure(state='disabled', text='Calculating...')
            self._update_status('Calculating...', Theme.ACCENT_YELLOW)

            thread = threading.Thread(
                target=self._run_calculation,
                args=(target_func, min_price, max_price, n_options, pricing),
                daemon=True
            )
            thread.start()

        except ValidationError as e:
            messagebox.showerror('Validation Error', str(e))
        except Exception as e:
            messagebox.showerror('Error', f'Unexpected error: {str(e)}')

    def _run_calculation(self, target_func, min_price, max_price, n_options, pricing):
        """Run calculation in background thread."""
        try:
            approximator = OptionsFunctionApproximator(
                n_options=n_options,
                price_range=(min_price, max_price),
                use_calls=self.use_calls_var.get(),
                use_puts=self.use_puts_var.get(),
                use_stock=self.use_stock_var.get(),
                **pricing
            )

            weights, mse = approximator.approximate(target_func, n_points=1000, regularization=0.001)
            premiums, premium_details = approximator.calculate_premiums()
            greeks = approximator.calculate_portfolio_greeks()

            self.approximator = approximator
            self.approximation_error = {'mse': mse, 'rmse': np.sqrt(mse)}
            self.premium_details = premium_details
            self.greeks = greeks

            self.root.after(0, lambda: self._update_ui(min_price, max_price))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror('Calculation Error', str(e)))
            self.root.after(0, lambda: self._update_status('Error', Theme.ACCENT_RED))
        finally:
            self.root.after(0, self._reset_calculate_button)

    def _reset_calculate_button(self):
        """Reset calculate button state."""
        self.is_calculating = False
        self.calculate_btn.configure(state='normal', text='Calculate')

    def _update_ui(self, min_price, max_price):
        """Update all UI elements after calculation."""
        self._update_plot(min_price, max_price)
        self._update_metrics()
        self._update_options_trees()
        self._update_greeks()
        self._update_details()
        self._update_status('Calculation complete', Theme.ACCENT_GREEN)

    def _update_plot(self, min_price, max_price):
        """Update the chart."""
        self.ax.clear()

        x = np.linspace(min_price, max_price, 1000)
        target_y = self.current_function(x)
        approx_y = self.approximator.evaluate(x)

        self.ax.plot(x, target_y, color=Theme.CHART_PRIMARY, linewidth=2.5, label='Target', alpha=0.9)
        self.ax.plot(x, approx_y, color=Theme.CHART_SECONDARY, linewidth=2, linestyle='--', label='Approximation', alpha=0.9)

        legend = self.ax.legend(loc='best', fontsize=9)
        legend.get_frame().set_facecolor(Theme.BG_SECONDARY)
        legend.get_frame().set_edgecolor(Theme.BORDER_DEFAULT)
        for text in legend.get_texts():
            text.set_color(Theme.TEXT_PRIMARY)

        self._style_axes(self.ax)
        self.ax.set_title('Target vs Approximation', fontsize=11, color=Theme.TEXT_PRIMARY, pad=10)

        # Add RMSE annotation
        rmse = self.approximation_error['rmse']
        self.ax.text(0.02, 0.98, f'RMSE: {rmse:.6f}',
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            color=Theme.TEXT_PRIMARY,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=Theme.BG_SECONDARY, edgecolor=Theme.BORDER_DEFAULT)
        )

        self.fig.tight_layout()
        self.canvas.draw()

    def _update_metrics(self):
        """Update summary metrics."""
        if self.approximation_error:
            self.metrics['mse'].set_value(f'{self.approximation_error["mse"]:.6f}')
            self.metrics['rmse'].set_value(f'{self.approximation_error["rmse"]:.6f}')

        if self.approximator:
            total_cost = self.approximator.get_total_cost()
            self.metrics['total_cost'].set_value(f'${total_cost:,.2f}')

        if self.premium_details:
            n_calls = len(self.premium_details.get('calls', []))
            n_puts = len(self.premium_details.get('puts', []))
            self.metrics['num_options'].set_value(str(n_calls + n_puts))

    def _update_options_trees(self):
        """Update options breakdown tables."""
        # Clear
        for item in self.calls_tree.get_children():
            self.calls_tree.delete(item)
        for item in self.puts_tree.get_children():
            self.puts_tree.delete(item)

        if not self.premium_details:
            return

        for call in sorted(self.premium_details.get('calls', []), key=lambda x: x['strike']):
            self.calls_tree.insert('', 'end', values=(
                f'${call["strike"]:.2f}',
                f'{call["weight"]:.4f}',
                f'${call["premium_per_unit"]:.2f}',
                f'${call["total_cost"]:.2f}'
            ))

        for put in sorted(self.premium_details.get('puts', []), key=lambda x: x['strike']):
            self.puts_tree.insert('', 'end', values=(
                f'${put["strike"]:.2f}',
                f'{put["weight"]:.4f}',
                f'${put["premium_per_unit"]:.2f}',
                f'${put["total_cost"]:.2f}'
            ))

    def _update_greeks(self):
        """Update Greeks display."""
        if not self.greeks:
            return

        for key, label in self.greek_displays.items():
            value = self.greeks.get(key, 0)
            label.configure(text=f'{value:.4f}')

    def _update_details(self):
        """Update detailed output."""
        self.details_text.configure(state='normal')
        self.details_text.delete('1.0', 'end')

        if not self.approximator:
            return

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            output = f"Options Function Approximation Results\n"
            output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            output += "=" * 60 + "\n\n"

            if self.approximation_error:
                output += "APPROXIMATION QUALITY\n"
                output += "-" * 40 + "\n"
                output += f"MSE:  {self.approximation_error['mse']:.6f}\n"
                output += f"RMSE: {self.approximation_error['rmse']:.6f}\n\n"

            self.approximator.print_cost_breakdown()
            output += buffer.getvalue()

            self.details_text.insert('end', output)

        finally:
            sys.stdout = old_stdout

        self.details_text.configure(state='disabled')

    def _update_status(self, message: str, color: str = Theme.TEXT_SECONDARY):
        """Update status indicator."""
        self.status_label.configure(text=message)

        # Update status dot color
        self.status_dot.delete('all')
        self.status_dot.create_oval(2, 2, 8, 8, fill=color, outline='')

    def _export_results(self):
        """Export results to file."""
        if not self.approximator:
            messagebox.showinfo('Export', 'No results to export. Run a calculation first.')
            return

        filetypes = [
            ('CSV files', '*.csv'),
            ('Text files', '*.txt'),
            ('All files', '*.*')
        ]

        filepath = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=filetypes,
            title='Export Results'
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(['Options Function Approximation Results'])
                writer.writerow([f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
                writer.writerow([])

                # Metrics
                writer.writerow(['Metrics'])
                writer.writerow(['MSE', self.approximation_error['mse']])
                writer.writerow(['RMSE', self.approximation_error['rmse']])
                writer.writerow(['Total Cost', self.approximator.get_total_cost()])
                writer.writerow([])

                # Greeks
                writer.writerow(['Portfolio Greeks'])
                for key, value in self.greeks.items():
                    writer.writerow([key.capitalize(), value])
                writer.writerow([])

                # Options
                if self.premium_details:
                    writer.writerow(['Call Options'])
                    writer.writerow(['Strike', 'Weight', 'Premium/Unit', 'Total Cost'])
                    for call in self.premium_details.get('calls', []):
                        writer.writerow([call['strike'], call['weight'], call['premium_per_unit'], call['total_cost']])
                    writer.writerow([])

                    writer.writerow(['Put Options'])
                    writer.writerow(['Strike', 'Weight', 'Premium/Unit', 'Total Cost'])
                    for put in self.premium_details.get('puts', []):
                        writer.writerow([put['strike'], put['weight'], put['premium_per_unit'], put['total_cost']])

            self._update_status(f'Exported to {filepath}', Theme.ACCENT_GREEN)

        except Exception as e:
            messagebox.showerror('Export Error', f'Failed to export: {str(e)}')

    def _on_closing(self):
        """Handle window close."""
        if self.is_calculating:
            if not messagebox.askokcancel('Quit', 'Calculation in progress. Quit anyway?'):
                return
        self.root.destroy()


def main():
    """Application entry point."""
    root = tk.Tk()
    app = OptionsApproximatorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
