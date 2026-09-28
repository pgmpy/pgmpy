# Agent Instructions - pgmpy

This file provides guidance for all AI agents when working with code in this repository.

## Project Overview

pgmpy is a Python library for Causal AI using DAGs, Bayesian Networks, and related models. It provides tools for causal
discovery, parameter estimation, inference, causal identification, and causal effect estimation.

## Development Setup

Requires Python >=3.10.

```bash
pip install -e .[tests]        # bash
pip install -e ".[tests]"      # zsh
pre-commit install             # set up git hooks
```

For PyTorch/GPU support: `pip install -e .[torch,tests]`. Other extras: `[optional]` (litellm for LLM-based features,
xgboost, plotting), `[docs]`, and `[all]`.

## Common Commands

pgmpy uses `pytest` for testing and `pre-commit` for code quality. Here are some common commands:

```bash
# Run all tests (add -n auto to run in parallel via pytest-xdist)
pytest -v pgmpy

# Run a specific test file
pytest -v pgmpy/tests/test_models/test_DiscreteBayesianNetwork.py

# Run a specific test class or method
pytest -v pgmpy/tests/test_models/test_DiscreteBayesianNetwork.py::TestBayesianModelCreation::test_add_cpds

# Linting/formatting (pre-commit runs ruff check --fix and ruff format; config in pyproject.toml)
pre-commit run --all-files

# Run docstring examples as doctests (also run in CI)
pytest --doctest-modules --ignore=pgmpy/tests pgmpy/
```

## Instructions for Tasks
1. Always make a plan before coding.
2. For behavior changes and bug fixes, prefer writing tests first (Test-Driven Development) when practical. For pure
   documentation or non-behavioral refactors, add tests only if behavior is affected. Avoid adding too many tests and
   try to combine tests when possible.
3. Follow existing patterns in the codebase. Always check for similar implementations before creating new ones. When
   adding a new causal discovery algorithm, CI test, structure score, metric, dataset, or example model, start from
   the matching template in `devtools/extension_templates/`. When changing a base-class API, update the affected
   templates as well.
4. When implementing or suggesting changes, also look into how similar packages — both Python (e.g., causal-learn,
   DoWhy, scikit-learn for API conventions) and R (e.g., bnlearn, pcalg, dagitty) — approach the same problem, and
   combine that information into the suggestion or design. Where possible, use these packages as reference
   implementations to verify the correctness of pgmpy's implementation.
5. Use type hints and docstrings for new or modified public methods.
6. Run the smallest relevant `pytest` target after changes to ensure correctness. Broaden to larger suites as needed.
   Run `pre-commit` when it is available in the environment.
7. Avoid redundant checks in the code. For example, if a variable is always expected to be a list, do not add checks to
   verify that it is a list. Try to avoid adding `try`/`except` blocks unless absolutely necessary. If you need to add error
   handling, make sure to be explicit about the expected exceptions and handle them appropriately.
8. Check if the method that you are using has a deprecation warning. If it does, try to use the recommended alternative
   instead of the deprecated method.
9. Preserve backwards compatibility unless the user explicitly requests or approves a breaking change. For migrations
   and refactors, prefer adding new APIs alongside compatibility shims before removing old paths.
10. If a required command or dependency is unavailable in the environment, state that explicitly and use the best
    available validation instead of silently skipping verification.
11. For code that supports multiple backends or optional dependencies, preserve existing `numpy` and `torch` behavior
    where applicable, and guard optional-dependency tests appropriately.
12. Never run `git commit` or `git push`. Leave all changes in the working tree — the user always reviews and commits
    manually. Suggesting logical commit units or a commit message is fine.

## Code Style

- **Formatter + linter**: ruff (`ruff format`, `ruff check --fix`); import sorting via ruff's isort rules
- **Config**: `[tool.ruff]` in `pyproject.toml` (line length: 120)
- Pre-commit hooks auto-run on commit after `pre-commit install`
- Prefer linear methods that read top to bottom over splitting logic into many small functions, so the whole algorithm
  can be followed in one place. Only factor out a helper when it is called from multiple locations.
- Do not add comments that explain your own reasoning for a change (that belongs in the commit/PR message). Comment
  only what the code cannot express itself.

## Standard Workflow
```bash
# 1. Create a feature or bugfix branch (human workflow; optional for agents
# already working in an assigned branch/worktree)
git checkout -b feature/your-feature dev

# 2. Make changes with tests
# ... edit files ...

# 3. Verify
pre-commit run --all-files
pytest -n auto pgmpy

# 4. Leave the changes uncommitted — the user reviews and commits manually
```

## Architecture

### Base Graph Classes (`pgmpy/base/`)
Key bases: `DAG`, `PDAG`, `MAG`, `ADMG`, `UndirectedGraph`, and `SimpleCausalModel` (a `DAG` built automatically from
node roles). These support node role annotations (exposures, outcomes, confounders, mediators, instruments, etc.).

`PDAG`, `MAG`, and `ADMG` extend `_CoreGraph` (`pgmpy/base/_base.py`), an `nx.MultiGraph` with typed edges
(`edge_type`) and algorithm/roles/plotting mixins; `DAG` still extends `nx.DiGraph`. The two families differ in
`has_edge`/`edges` semantics, so audit callers when moving code between them.

### Models (`pgmpy/models/`)
Representation of models:
- **DiscreteBayesianNetwork** — standard BN with `TabularCPD` factors
- **LinearGaussianBayesianNetwork** — continuous Gaussian with `LinearGaussianCPD`
- **FunctionalBayesianNetwork** — arbitrary distributions via `FunctionalCPD` (requires torch/pyro)
- **DynamicBayesianNetwork** — time-series BNs
- **DiscreteMarkovNetwork** — undirected graphical models
- **MarkovChain** — discrete Markov chain with sampling support
- **JunctionTree**, **FactorGraph**, **ClusterGraph** — inference data structures
- **SEM** — structural equation models
- **NaiveBayes** — special case of a BN

`BayesianNetwork` and `MarkovNetwork` are deprecated aliases of `DiscreteBayesianNetwork` and
`DiscreteMarkovNetwork`.

### Factors (`pgmpy/factors/`)
Probability representations attached to models:
- **discrete/** — `DiscreteFactor`, `TabularCPD`, `NoisyORCPD`, `JointProbabilityDistribution`
- **continuous/** — `LinearGaussianCPD`
- **hybrid/** — `FunctionalCPD` (Pyro distribution-based)

### Estimators (`pgmpy/estimators/`)
Legacy compatibility layer: `MaximumLikelihoodEstimator`, `BayesianEstimator`, `ExpectationMaximization`, `PC`, `GES`,
`HillClimbSearch`, `ExhaustiveSearch`, `TreeSearch`, `MmhcEstimator`, structure scores, etc. Do not add new
functionality here — new work goes in the canonical packages: structure learning in `pgmpy/causal_discovery/`,
parameter learning in `pgmpy/parameter_estimator/`, scores in `pgmpy/structure_score/`, CI tests in `pgmpy/ci_tests/`.

`pgmpy.estimators.ExpertKnowledge` and `pgmpy.causal_discovery.ExpertKnowledge` are two incompatible classes: the
`estimators` one is frozen for backwards compatibility; new code should use the `causal_discovery` one.

### Parameter Estimation (`pgmpy/parameter_estimator/`)
Canonical implementation location for parameter learning from data:
- **Discrete**: `DiscreteMLE`, `DiscreteBayesianEstimator`, `DiscreteEM`
- **Continuous**: `LinearGaussianMLE`

### Structure Scores (`pgmpy/structure_score/`)
Canonical implementation location for structure scores: `K2`, `BDeu`, `BDs`, `LogLikelihood`, `AIC`, `BIC` (plus
`*Gauss` and `*CondGauss` variants), with `BaseStructureScore` and `get_scoring_method`. Prefer this over importing
structure scores from `pgmpy.estimators`.

### Conditional Independence Tests (`pgmpy/ci_tests/`)
Canonical implementation location for CI tests used by constraint-based discovery and model testing: `ChiSquare`,
`GSq`, `PowerDivergence`, `FisherZ`, `Pearsonr`, `PillaiTrace`, `GCM`, `IndependenceMatch`, and others. Tests subclass
`BaseCITest` and return `_CITestResult` objects (`pgmpy/ci_tests/_base.py`).

### Inference (`pgmpy/inference/`)
Computing posterior distributions given model and evidence:
- **Exact**: `VariableElimination`, `BeliefPropagation`
- **Approximate**: `ApproxInference` (sampling-based methods live in `pgmpy/sampling/`)
- **Causal**: `CausalInference` (do-calculus)
- **Dynamic**: `DBNInference`

### Causal Discovery (`pgmpy/causal_discovery/`)
Canonical implementation location for learning causal structure from data:
- **Constraint-based**: `PC`
- **Score-based**: `GES`, `HillClimbSearch`, `TOPIC`
- **Tree-based**: `ChowLiu`, `TAN`
- **Expert/LLM-based**: `ExpertInLoop`, `LLMPairwise` (needs `litellm` from the `[optional]` extra)
- **Prior knowledge**: `ExpertKnowledge` for encoding constraints on the search space

### Causal Identification (`pgmpy/identification/`)
Identification of causal effects from a given causal graph:
- **Generalized adjustment sets**: `Adjustment`
- **Frontdoor criterion**: `Frontdoor`

### Causal Effect Estimation (`pgmpy/prediction/`)
Estimating causal effects from data given a causal graph:
- **Adjustment regression**: `NaiveAdjustmentRegressor`
- **Double ML**: `DoubleMLRegressor`
- **IV estimation**: `NaiveIVRegressor`

### Example Datasets (`pgmpy/datasets/`)
Set of example datasets. The `list_datasets(**filter_tags)` function lists all available datasets.

### Example Models (`pgmpy/example_models/`)
Set of example models. The `list_models(**filter_tags)` function lists all available models.

### Metrics (`pgmpy/metrics/`)
Evaluation metrics for causal discovery and models against given dataset.
- **Model evaluation**: `CorrelationScore`, `FisherC`, `ImpliedCIs`, `StructureScore`
- **Graph comparison**: `SHD`, `AdjacencyConfusionMatrix`, `OrientationConfusionMatrix`
- `get_metrics` looks up metric classes by tag filters.

### Other Subpackages
- **Independencies (`pgmpy/independencies/`)** — `Independencies` for representing sets of conditional independence
  assertions
- **Sampling (`pgmpy/sampling/`)** — `BayesianModelSampling`, `GibbsSampling`
- **Utils (`pgmpy/utils/`)** — shared helpers (numpy/torch compat functions, input checks) and bundled example model
  files

### Extension Templates (`devtools/extension_templates/`)
Skeleton subclasses of the public base classes for new causal discovery algorithms, CI tests, structure scores,
metrics, datasets, and example models. Start new extensions from these, and keep them in sync when base-class APIs
change.

### Backend System (`pgmpy/global_vars.py`)
Global `config` object switches between numpy and torch backends:
```python
from pgmpy.global_vars import config

config.set_backend("torch", device="cuda")
```
The same `config` object also exposes `set_device`, `set_dtype`, and `set_show_progress`.

### I/O (`pgmpy/readwrite/`)
Use the `load` and `save` methods on `DiscreteBayesianNetwork` and `LinearGaussianBayesianNetwork` for model
serialization. Supported formats include BIF, XMLBIF, XBN, NET (HUGIN), XDSL (GeNIe), UAI, and PomdpX.

### Tests (`pgmpy/tests/`)
Mirror the main package structure. Every subpackage and module has a corresponding `test_` directory, and `test_` file.

CI also runs all docstring examples as doctests (`pytest --doctest-modules`) and executes the notebooks in `examples/`
(`.github/workflows/doctests.yml`, `notebooks.yml`) — docstring examples must run as written.
