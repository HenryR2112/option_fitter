# Contributing to Options Function Approximator

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

This project adheres to a standard of respectful, professional interaction. We welcome contributors from all backgrounds and experience levels.

### Our Standards

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project
- Show empathy towards other contributors

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Git
- Familiarity with NumPy and Matplotlib
- (Optional) Lean 4 for proof contributions
- (Optional) LaTeX for documentation contributions

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/option_fitter.git
   cd option_fitter
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/HenryR2112/option_fitter.git
   ```

## How to Contribute

### Reporting Bugs

Use GitHub Issues to report bugs. Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS
- Relevant code snippets or error messages

**Template:**
```markdown
**Bug Description:**
[Clear description of the bug]

**To Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Environment:**
- OS: [e.g., macOS 13.0]
- Python: [e.g., 3.10.5]
- NumPy: [e.g., 2.0.1]

**Additional Context:**
[Any other relevant information]
```

### Suggesting Enhancements

Use GitHub Issues for feature requests. Include:
- Clear description of the enhancement
- Use cases and benefits
- Potential implementation approach
- Any relevant examples or references

### Pull Requests

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Test thoroughly
4. Update documentation
5. Commit with clear messages
6. Push to your fork
7. Submit a pull request

## Development Setup

### Install Development Dependencies

```bash
# Clone and enter directory
git clone https://github.com/YOUR-USERNAME/option_fitter.git
cd option_fitter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install pytest black flake8 mypy
```

### Verify Installation

```bash
# Test core library
python -c "from options_func_maker import OptionsFunctionApproximator; print('✓ Core library OK')"

# Test GUI (should launch window)
python options_gui.py

# Run experiments
python experiments/run_experiments.py
```

### Lean 4 Setup (for proof contributions)

```bash
# Install Lean 4
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build proofs
cd proof
lake build
```

## Coding Standards

### Python Style

Follow PEP 8 with these specifics:

**Code Style:**
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names
- Avoid single-letter variables (except loop counters like `i`, `j`)

**Naming Conventions:**
- Functions/methods: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

**Type Hints:**
```python
def approximate(
    self,
    func: Callable[[np.ndarray], np.ndarray],
    n_points: int = 1000,
    regularization: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """
    Approximate a function using option basis.

    Args:
        func: Target function to approximate
        n_points: Number of sample points
        regularization: L2 regularization parameter

    Returns:
        Tuple of (weights, mean_squared_error)
    """
    # Implementation here
    pass
```

**Docstrings:**
Use Google style:
```python
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call option price.

    Uses the Black-Scholes formula for European call options.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)

    Returns:
        Call option premium

    Example:
        >>> price = black_scholes_call(100, 105, 0.25, 0.05, 0.2)
        >>> print(f"Call price: ${price:.2f}")
    """
    # Implementation
```

### Code Formatting

We recommend using Black for automatic formatting:

```bash
# Install Black
pip install black

# Format all Python files
black .

# Check without making changes
black --check .
```

### Linting

Use flake8 for style checking:

```bash
# Install flake8
pip install flake8

# Run linter
flake8 --max-line-length=88 --extend-ignore=E203 .
```

### Type Checking (Optional but Recommended)

```bash
# Install mypy
pip install mypy

# Run type checker
mypy options_func_maker.py
```

## Testing

### Manual Testing

Before submitting, verify:
1. Core library functions work correctly
2. GUI launches and calculates approximations
3. Experiments run without errors
4. No import errors or dependency issues

### Example Test Script

Create `test_basic.py`:
```python
import numpy as np
from options_func_maker import OptionsFunctionApproximator

def test_sin_approximation():
    """Test basic sin(x) approximation."""
    approx = OptionsFunctionApproximator(
        n_options=10,
        price_range=(0, 2*np.pi),
        use_calls=True,
        use_puts=True
    )

    weights, mse = approx.approximate(np.sin, n_points=500)

    assert mse < 0.01, f"MSE too high: {mse}"
    print(f"✓ Sin approximation test passed (MSE={mse:.6f})")

if __name__ == "__main__":
    test_sin_approximation()
```

Run with:
```bash
python test_basic.py
```

### Testing Guidelines

- Add tests for new functionality
- Ensure existing functionality still works
- Test edge cases (empty inputs, extreme values)
- Verify numerical stability

## Documentation

### Updating Documentation

When adding features or changing behavior:
1. Update relevant README files
2. Update docstrings
3. Add examples if appropriate
4. Update CHANGES.md

### Documentation Standards

- Use clear, concise language
- Provide examples for complex features
- Include code snippets with syntax highlighting
- Link between related documentation sections

### LaTeX Documentation

For academic contributions (`tex/draft.tex`):
- Follow existing formatting
- Add references to bibliography
- Ensure equations compile correctly
- Run `pdflatex draft.tex` to verify

### Lean Proofs Documentation

For formal verification (`proof/`):
- Document new theorems clearly
- Explain proof strategies
- Update proof/README.md with status
- Ensure `lake build` succeeds

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

**Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat: Add support for custom strike placement

- Allow users to specify custom strike arrays
- Update GUI with strike placement options
- Add validation for strike ordering

Closes #42
```

```
fix: Prevent singular matrix error in edge cases

Increase default regularization from 1e-8 to 1e-6 to avoid
numerical instability with many basis functions.

Fixes #38
```

### Pull Request Process

1. **Update Documentation**: Ensure all docs reflect your changes
2. **Update CHANGES.md**: Add entry under "Unreleased"
3. **Self-Review**: Check your own changes before submitting
4. **Write Clear PR Description**:
   ```markdown
   ## Description
   [Clear description of changes]

   ## Motivation
   [Why this change is needed]

   ## Changes Made
   - Change 1
   - Change 2

   ## Testing
   [How you tested these changes]

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] CHANGES.md updated
   - [ ] Tested manually
   - [ ] No breaking changes (or clearly documented)
   ```

5. **Respond to Reviews**: Address feedback promptly and professionally

### What Happens Next

- Maintainers will review your PR
- You may be asked to make changes
- Once approved, your PR will be merged
- Your contribution will be acknowledged in CHANGES.md

## Areas for Contribution

### High-Priority Areas

1. **Additional Basis Functions**
   - Wavelets
   - Fourier basis
   - Chebyshev polynomials
   - Custom user-defined bases

2. **Performance Optimizations**
   - Sparse matrix support
   - Vectorization improvements
   - Caching mechanisms
   - Parallel computation

3. **Testing Suite**
   - Unit tests for core functions
   - Integration tests for GUI
   - Regression tests
   - Performance benchmarks

4. **GUI Enhancements**
   - Real-time parameter adjustment
   - 3D visualization for multi-asset
   - Save/load configuration
   - Batch processing interface

5. **Numerical Experiments**
   - Comparative studies (different bases)
   - Convergence rate analysis
   - Cost-accuracy trade-off studies
   - Robustness testing

6. **Lean Proofs**
   - Complete proofs (replace `sorry`)
   - Additional theorems
   - Optimization of proof strategies
   - Documentation of proof techniques

7. **Documentation**
   - Tutorial notebooks
   - Video tutorials
   - Use case examples
   - FAQ section

### Medium-Priority Areas

- Support for American options (early exercise)
- Multi-asset extensions
- Time-dependent replication (dynamic hedging)
- Integration with financial data APIs
- Exotic option support (barriers, digitals)
- Risk management tools (VaR, CVaR)

### Good First Issues

Look for issues labeled `good first issue` on GitHub:
- Documentation improvements
- Adding examples
- Bug fixes with clear solutions
- Test coverage improvements

## Getting Help

- **Questions**: Use GitHub Discussions
- **Bugs**: Open an issue
- **Chat**: (If you have a community chat, add link here)
- **Email**: (Optional: maintainer email)

## Recognition

Contributors are recognized in:
- CHANGES.md under version releases
- GitHub contributors page
- (Optional: AUTHORS or CONTRIBUTORS file)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Options Function Approximator project! Your efforts help advance the connection between finance, approximation theory, and machine learning. 🚀
