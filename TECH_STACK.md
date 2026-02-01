# Tech Stack - pgmpy

This document outlines the core technologies and tools used in the development and maintenance of `pgmpy`.

## Languages & Core Libraries
- **Python (>= 3.10)**: The primary programming language.
- **NetworkX**: Used for representing and manipulating graph structures.
- **NumPy**: The foundation for numerical computations and matrix operations.
- **SciPy**: For advanced mathematical functions and optimization.
- **Pandas**: For efficient handling of tabular data.

## Machine Learning & Causal Inference
- **Scikit-learn**: Used in some estimators and utility functions.
- **Statsmodels**: For statistical tests in structure learning (e.g., CI tests).
- **Pytorch / Pyro (Optional)**: Used for functional Bayesian Networks and gradient-based learning.
- **Daft**: For PGM visualization.

## Development & Quality Assurance
- **Pytest**: The standard testing framework.
- **Ruff**: Fast linting and formatting (replaces Flake8, Isort, Black).
- **Black**: Replaced by Ruff but historically used for formatting.
- **Pre-commit**: Manages git hooks for linting/formatting before commits.
- **Coverage.py**: Tracks test coverage.

## Documentation
- **Sphinx**: Documentation generator.
- **Numpydoc**: Docstring convention.
- **Nbsphinx**: For building documentation from Jupyter notebooks.
- **ReadTheDocs**: Hosting platform for documentation.

## CI/CD
- **GitHub Actions**: For automated testing, linting, and release workflows.
- **ASV (Airspeed Velocity)**: For performance benchmarking.
