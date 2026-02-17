# Agent Instructions - pgmpy

This file provides guidance for all AI agents when working with code in this repository.

## Project Overview

pgmpy is a Python library for Causal AI using DAGs, Bayesian Networks, and related models. It provides tools for causal
discovery, parameter estimation, inference, causal identification, and causal effect estimation.


## Development Setup

```bash
pip install -e .[tests]        # bash
pip install -e ".[tests]"      # zsh
```

For PyTorch/GPU support: `pip install -e .[torch,tests]`

## Common Commands

pgmpy uses `pytest` for testing and `pre-commit` for code quality. Here are some common commands:

```bash
# Run all tests
pytest -v pgmpy

# Run a specific test file
pytest -v pgmpy/tests/test_models/test_DiscreteBayesianNetwork.py

# Run a specific test class or method
pytest -v pgmpy/tests/test_models/test_DiscreteBayesianNetwork.py::TestBayesianModelCreation::test_add_cpds

# Linting (pre-commit runs black, flake8, isort, blackdoc)
pre-commit run --all-files
```

## Instructions for Tasks
1. Always make a plan before coding.
2. For any change write tests first (Test-Driven Development). Avoid adding too many tests and try to combine tests when
   possible.
3. Follow existing patterns in the codebase. Always check for similar implementations before creating new ones.
4. Use type hints and docstrings for all public methods.
5. Run `pre-commit` hooks and `pytest` after changes to ensure code quality and correctness.
6. Avoid redundant checks in the code. For example, if a variable is always expected to be a list, do not add checks to
   verify that it is a list. Try to avoid adding try except blocks unless absolutely necessary. If you need to add error
   handling, make sure to be explicit about the expected exceptions and handle them appropriately.
7. Check if the method that you are using has a deprecation warning. If it does, try to use the recommended alternative
   instead of the deprecated method.
8. When making changes to codebase, aim for modularity and reusability.
9. When adding new functionality, consider how it can be integrated with existing components and whether it can be
   implemented in a way that allows for future extensions or modifications without requiring significant changes to the
   existing codebase.

## Code Style

- **Formatter**: black
- **Import sorting**: isort
- **Linter**: flake8 + ruff (line length: 120)
- **Docstring formatting**: blackdoc
- Pre-commit hooks auto-run on commit after `pre-commit install`

## Standard Workflow
```bash
# 1. Create feature or a bugfix branch
git checkout -b feature/your-feature dev

# 2. Make changes with tests
# ... edit files ...

# 3. Verify
pre-commit run --all-files
pytest pgmpy
```

## Architecture

### Base Graph Classes (`pgmpy/base/`)
All models build on NetworkX graph classes. Key bases: `DAG`, `PDAG`, `MAG`, `ADMG`, `UndirectedGraph`. These support node role annotations (exposures, outcomes, confounders, mediators, etc.).

### Models (`pgmpy/models/`)

Representation of models:
- **DiscreteBayesianNetwork** — standard BN with `TabularCPD` factors
- **LinearGaussianBayesianNetwork** — continuous Gaussian with `LinearGaussianCPD`
- **FunctionalBayesianNetwork** — arbitrary distributions via `FunctionalCPD` (requires torch/pyro)
- **DynamicBayesianNetwork** — time-series BNs
- **DiscreteMarkovNetwork** — undirected graphical models
- **JunctionTree**, **FactorGraph**, **ClusterGraph** — inference data structures
- **SEM** — structural equation models
- **NaiveBayes** - Special case of BN.

### Factors (`pgmpy/factors/`)
Probability representations attached to models:
- **discrete/** — `DiscreteFactor`, `TabularCPD`, `NoisyORCPD`, `JointProbabilityDistribution`
- **continuous/** — `LinearGaussianCPD`
- **hybrid/** — `FunctionalCPD` (Pyro distribution-based)

### Estimators (`pgmpy/estimators/`)
Estimators for learning model parameters from data:
- **Parameter learning**: `MaximumLikelihoodEstimator`, `BayesianEstimator`, `ExpectationMaximization`
- **Structure learning**: `PC`, `HillClimbSearch`, `ExhaustiveSearch`, `GES`, `MmhcEstimator`, `TreeSearch`
- **Scoring**: K2, BDeu, BDs, BIC, AIC (and Gaussian, and conditional gaussian variants)

### Inference (`pgmpy/inference/`)
Computing posterior distributions given model and evidence:
- **Exact**: `VariableElimination`, `BeliefPropagation`
- **Approximate**: `ApproxInference`, `GibbsSampling`
- **Causal**: `CausalInference` (do-calculus)
- **Dynamic**: `DBNInference`

### Causal Discovery (`pgmpy/causal_discovery/`)
Learning causal structure from data:
- **Constraint-based**: `PC`, `FCI`
- **Score-based**: `GES`, `MMHC`

### Causal Identification (`pgmpy/causal_identification/`)
Identification of causal effects from a given causal graph:
- **Generalized adjustment sets**: `Adjustment`
- **Frontdoor criterion**: `Frontdoor`

### Causal Effect Estimation (`pgmpy/causal_estimation/`)
Estimating causal effects from data given a causal graph:
- **Adjustment regression**: `NaiveAdjustmentRegressor`
- **Doubly robust**: `DoublyMLRegressor`
- **IV estimation**: `NaiveIVRegressor`

### Example Datasets (`pgmpy/datasets/`)
Set of example datasets. `list_datasets` method provides all available datasets.

### Example models (`pgmpy/example_models/`)
Set of example models. `list_models` method provides all available models.

### Metrics (`pgmpy/metrics/`)
Evaluation metrics for causal discovery and models against given dataset.
- **Model evaluation**: `CorrelationScore`, `FisherC`, `ImpliedCIs`, `SHD`, `StructureScore`.

### Backend System (`pgmpy/global_vars.py`)
Global `config` object switches between numpy and torch backends:
```python
from pgmpy.global_vars import config

config.set_backend("torch", device="cuda")
```
Also provides functionality to specify configuration options on a global level.

### I/O (`pgmpy/readwrite/`)
Use `load` and `save` functions in `DiscreteBayesianNetwork` and `LinearGaussianBayesianNetwork` for model
serialization. Many formats are supported.

### Tests (`pgmpy/tests/`)
Mirror the main package structure. Every subpackage and module has a corresponding `test_` directory, and `test_` file.
