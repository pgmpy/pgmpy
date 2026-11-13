# Agent Instructions - pgmpy

## Project Overview

pgmpy is a Python library for Causal AI using DAGs, Bayesian Networks, and related models. It provides tools for causal
discovery, parameter estimation, inference, causal identification, and causal effect estimation.

## Common Commands

Feature and bugfix branches start from `dev`.

```bash
# Linting/formatting (pre-commit runs ruff check --fix and ruff format; config in pyproject.toml)
pre-commit run --all-files

# Run docstring examples as doctests (also run in CI)
pytest --doctest-modules --ignore=pgmpy/tests pgmpy/
```

## Instructions for Tasks
1. Always make a plan before coding.
2. Write tests first for behavior changes and bug fixes when practical. Refactors and documentation changes need tests
   only if behavior changes. Prefer a few combined tests over many small ones. Try to incorporate new tests within existing ones if possible.
3. Follow existing patterns: check for a similar implementation before creating a new one. New extensions start from
   the templates in `devtools/extension_templates/`, and base-class API changes update them (`add-extension` skill).
4. Look at how other Python and R packages solve the same problem, and verify pgmpy's results against them where
   possible (`cross-check-reference` skill).
5. Use type hints and docstrings for new or modified public methods.
6. Run the smallest relevant `pytest` target after changes, then broaden as needed. Run `pre-commit` when it is
   available.
7. Don't add redundant checks (e.g. checking the type of a value that is always a list) or speculative `try`/`except`
   blocks. When error handling is needed, catch only the specific exceptions you expect.
8. Don't use deprecated APIs; use the replacement the warning names. Preserve backwards compatibility unless the user
   explicitly approves a breaking change: add the new API alongside a deprecated shim before removing the old path
   (`deprecate-api` skill).
9. If a required command or dependency is unavailable, say so explicitly and use the best available validation instead
   of silently skipping verification.
10. For code that supports multiple backends or optional dependencies, preserve existing `numpy` and `torch` behavior
    and guard optional-dependency tests.
11. Never run `git commit` or `git push`. Leave all changes in the working tree. The user always reviews and commits
    manually. Suggesting logical commit units or a commit message is fine.

## Code Style

- Prefer linear methods that read top to bottom over splitting logic into many small functions, so the whole algorithm
  can be followed in one place. Only factor out a helper when it is called from multiple locations.
- Do not add comments that explain your own reasoning for a change (that belongs in the commit/PR message). Comment
  only what the code cannot express itself.
- Use the Step x.x style comments in code such that the Step outlines the pseudo code. You can look up existing methods for examples.
- Method need to be single purpose. Do not write a method that do many things and return many values.

## Architecture Notes

- `PDAG`, `MAG`, and `ADMG` extend `_CoreGraph` (`pgmpy/base/_base.py`), an `nx.MultiGraph` with typed edges
  (`edge_type`); `DAG` still extends `nx.DiGraph`. The two families differ in `has_edge`/`edges` semantics, so audit
  callers when moving code between them.
- Graph classes declare what they support with class attributes (`SUPPORTED_EDGE_TYPES`, `SUPPORTS_LATENTS`) that are
  enforced in the base classes.
- `BayesianNetwork` and `MarkovNetwork` are removed: instantiating them raises `ImportError`. Use
  `DiscreteBayesianNetwork` and `DiscreteMarkovNetwork`.
- `pgmpy/estimators/` is a legacy compatibility layer. Do not add new functionality there — new work goes in the
  canonical packages: structure learning in `pgmpy/causal_discovery/`, parameter learning in
  `pgmpy/parameter_estimator/`, scores in `pgmpy/structure_score/`, CI tests in `pgmpy/ci_tests/`.
- Causal identification lives in `pgmpy/identification/` (the backdoor and frontdoor adjustment-set methods on
  `CausalInference` are deprecated in its favor), and causal effect estimation in `pgmpy/prediction/`.
  `pgmpy/parameterization/` is experimental.
- `pgmpy.estimators.ExpertKnowledge` is a deprecated shim over `pgmpy.causal_discovery.ExpertKnowledge` that keeps the
  legacy argument names and methods; new code should use the `causal_discovery` one.
- Raise user-facing warnings with `pgmpy.utils._warnings._warn_external`. Deprecations use `FutureWarning`
  (`deprecate-api` skill); experimental modules use `ExperimentalWarning`.
- Every skbase tag used by pgmpy objects is registered and documented as a `_BaseTag` subclass in
  `pgmpy/registry/_tags.py`; add new tags there. Multi-valued tags must be lists, not tuples.
- Use the `load` and `save` methods on `DiscreteBayesianNetwork` and `LinearGaussianBayesianNetwork` for model
  serialization.
- Docstring examples must run as written: CI runs them as doctests and also executes the notebooks in `examples/`
  (`.github/workflows/doctests.yml`, `notebooks.yml`).

## Scientific Python SPECs

pgmpy follows these [Scientific Python SPECs](https://scientific-python.org/specs/); flag code that breaks them.

- [SPEC 7](https://scientific-python.org/specs/spec-0007/): take randomness as a keyword-only
  `rng: int | np.random.Generator | None = None`, normalize it once with `np.random.default_rng(rng)`, and pass that
  Generator, not the seed, to callees. Never use numpy's global state (ruff `NPY002`) or pass `None` to a third-party
  `random_state`, which falls back to it. Classes implementing a third-party interface keep its parameter name.

## Skills

Task-specific procedures live in `.agents/skills/<name>/SKILL.md`. Agents that don't load skills automatically should
read the matching file before starting the task.

- `add-extension`: adding a causal discovery algorithm, CI test, structure score, metric, dataset, or example model.
- `cross-check-reference`: verifying results against bnlearn, pcalg, dagitty, causal-learn, and other references.
- `deprecate-api`: renaming, moving, or removing a public API.
- `review-pr`: reviewing a pull request, branch, or diff.
