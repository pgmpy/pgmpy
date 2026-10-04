# Agent Instructions - pgmpy

This file provides guidance for all AI agents when working with code in this repository.

## Project Overview

pgmpy is a Python library for Causal AI using DAGs, Bayesian Networks, and related models. It provides tools for causal
discovery, parameter estimation, inference, causal identification, and causal effect estimation.

## Common Commands

```bash
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

- Prefer linear methods that read top to bottom over splitting logic into many small functions, so the whole algorithm
  can be followed in one place. Only factor out a helper when it is called from multiple locations.
- Do not add comments that explain your own reasoning for a change (that belongs in the commit/PR message). Comment
  only what the code cannot express itself.

## Standard Workflow

Feature and bugfix branches start from `dev`.

## Architecture Notes

- `PDAG`, `MAG`, and `ADMG` extend `_CoreGraph` (`pgmpy/base/_base.py`), an `nx.MultiGraph` with typed edges
  (`edge_type`); `DAG` still extends `nx.DiGraph`. The two families differ in `has_edge`/`edges` semantics, so audit
  callers when moving code between them.
- `BayesianNetwork` and `MarkovNetwork` are deprecated aliases of `DiscreteBayesianNetwork` and
  `DiscreteMarkovNetwork`.
- `pgmpy/estimators/` is a legacy compatibility layer. Do not add new functionality there — new work goes in the
  canonical packages: structure learning in `pgmpy/causal_discovery/`, parameter learning in
  `pgmpy/parameter_estimator/`, scores in `pgmpy/structure_score/`, CI tests in `pgmpy/ci_tests/`.
- `pgmpy.estimators.ExpertKnowledge` and `pgmpy.causal_discovery.ExpertKnowledge` are two incompatible classes: the
  `estimators` one is frozen for backwards compatibility; new code should use the `causal_discovery` one.
- Every skbase tag used by pgmpy objects is registered and documented as a `_BaseTag` subclass in
  `pgmpy/registry/_tags.py`; add new tags there. Multi-valued tags must be lists, not tuples.
- Use the `load` and `save` methods on `DiscreteBayesianNetwork` and `LinearGaussianBayesianNetwork` for model
  serialization.
- CI also runs all docstring examples as doctests (`pytest --doctest-modules`) and executes the notebooks in `examples/`
  (`.github/workflows/doctests.yml`, `notebooks.yml`) — docstring examples must run as written.
