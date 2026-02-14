# Agent Instructions - pgmpy

Operational guide for autonomous AI agents working on `pgmpy`.

## ⚡ First Steps for Any Task
1. Identify files to modify
2. Identify test files to create/update
3. Plan before implementing
4. Check existing similar implementations
5. Follow naming conventions

## 🎯 Task Decision Tree

```
What are you doing?
│
├── 🐛 Fixing a bug
│   1. Create minimal reproduction test
│   2. Fix the code
│   3. Verify test passes
│   4. Run full test suite
│
├── ✨ Adding a feature
│   1. Identify component type (model/estimator/inference)
│   2. Check existing similar implementations
│   3. Write tests first (TDD)
│   4. Implement following patterns
│   5. Run ruff + pytest
│
├── 📝 Improving docs
│   1. Locate doc in docs/ or docstrings
│   2. Update content
│   3. Verify links work
│
└── 🔧 Refactoring
    1. Ensure tests exist for affected code
    2. Make incremental changes
    3. Run tests after each change
```

## 📁 Component Mapping

| Component | Location | Base Class | Test Location |
|-----------|----------|------------|---------------|
| Bayesian Network | `models/` | `DAG` | `tests/test_models/` |
| Markov Network | `models/` | `UndirectedGraph` | `tests/test_models/` |
| Factor/CPD | `factors/` | `BaseFactor` | `tests/test_factors/` |
| Estimator | `estimators/` | `BaseEstimator` | `tests/test_estimators/` |
| Inference | `inference/` | `Inference` | `tests/test_inference/` |

## 🧩 Full Component Reference

### Models (`pgmpy/models/`)
| Class | Type | Description |
|-------|------|-------------|
| `DiscreteBayesianNetwork` | Bayesian Network | Discrete variables |
| `LinearGaussianBayesianNetwork` | Bayesian Network | Continuous Gaussian |
| `FunctionalBayesianNetwork` | Bayesian Network | Arbitrary functions |
| `NaiveBayes` | Bayesian Network | Classifier |
| `DynamicBayesianNetwork` | Bayesian Network | Time-series |
| `DiscreteMarkovNetwork` | Markov Network | Discrete undirected |
| `FactorGraph` | Markov Network | Factor representation |
| `ClusterGraph` | Markov Network | Clustered representation |
| `JunctionTree` | Inference | Junction tree structure |
| `SEM` | Causal | Structural equations |
| `MarkovChain` | Stochastic | Stochastic process |

### Factors (`pgmpy/factors/`)
| Class | Type |
|-------|------|
| `TabularCPD` | Conditional Probability Distribution (Discrete) |
| `DiscreteFactor` | Generic discrete factor |
| `JointProbabilityDistribution` | Joint Probability Distribution |
| `NoisyORCPD` | Noisy-OR CPD |
| `LinearGaussianCPD` | Continuous CPD |
| `FunctionalCPD` | Arbitrary function CPD |

### Estimators (`pgmpy/estimators/`)
| Class | Type | Method |
|-------|------|--------|
| `MaximumLikelihoodEstimator` | Parameter | Maximum Likelihood |
| `BayesianEstimator` | Parameter | Bayesian |
| `ExpectationMaximization` | Parameter | Expectation-Maximization |
| `PC` | Structure | Constraint-based |
| `HillClimbSearch` | Structure | Score-based |
| `GES` | Structure | Greedy Equivalence Search |
| `ExhaustiveSearch` | Structure | Brute force |
| `TreeSearch` | Structure | Tree structure |
| `MmhcEstimator` | Structure | Hybrid |

### Inference (`pgmpy/inference/`)
| Class | Type |
|-------|------|
| `VariableElimination` | Exact inference |
| `BeliefPropagation` | Exact inference |
| `DBNInference` | Exact inference (Dynamic Bayesian Network) |
| `ApproxInference` | Approximate inference (Sampling) |
| `CausalInference` | Causal inference (do-calculus) |
| `Mplp` | Approximate inference (Message passing) |

## 🔧 Commands Quick Reference

| Task | Command |
|------|---------|
| Install (dev) | `pip install -e .[tests]` |
| Install (all) | `pip install -e .[all,tests]` |
| Test all | `pytest` |
| Test specific | `pytest pgmpy/tests/test_<module>.py` |
| Test verbose | `pytest -v` |
| Coverage | `pytest --cov=pgmpy` |
| Lint | `pre-commit run --all-files` |
| Format | `pre-commit run --all-files` |

## 🚫 Don't Do This

| ❌ Don't | ✅ Do |
|----------|------|
| Use `unittest` | Use `pytest` |
| Skip type hints | Add hints to public methods |
| Loop over DataFrame rows | Use vectorized operations |
| Add core dependencies | Use optional extras |
| Delete existing tests | Add more tests |
| Change public API silently | Discuss in issue/PR first |
| Commit without testing | Run `pre-commit run --all-files && pytest` |

## 🔄 Standard Workflow
```bash
# 1. Create feature branch
git checkout -b feature/your-feature dev

# 2. Make changes with tests
# ... edit files ...

# 3. Verify
pre-commit run --all-files
pytest

```

## 📊 CI/CD
| Workflow | File | Purpose |
|----------|------|---------|
| Tests | `ci.yml` | Py 3.10-3.14, all OS |
| Lint | `lint.yml` | Ruff checks |
| Security | `codeql.yml` | Vulnerability scan |
