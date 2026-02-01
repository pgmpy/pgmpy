# Agent Instructions - pgmpy

Operational guide for autonomous AI agents working on `pgmpy`.

## ⚡ First Steps for Any Task
1. Read `.cursorrules` → coding patterns
2. Read `ARCHITECTURE.md` → project structure
3. Identify files to modify
4. Identify test files to create/update
5. Plan before implementing

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
| `DiscreteBayesianNetwork` | BN | Discrete variables |
| `LinearGaussianBayesianNetwork` | BN | Continuous Gaussian |
| `FunctionalBayesianNetwork` | BN | Arbitrary functions |
| `NaiveBayes` | BN | Classifier |
| `DynamicBayesianNetwork` | BN | Time-series |
| `DiscreteMarkovNetwork` | MN | Discrete undirected |
| `FactorGraph` | MN | Factor representation |
| `ClusterGraph` | MN | Clustered representation |
| `JunctionTree` | Inf | Junction tree structure |
| `SEM` | Causal | Structural equations |
| `MarkovChain` | Stoch | Stochastic process |

### Factors (`pgmpy/factors/`)
| Class | Type |
|-------|------|
| `TabularCPD` | Discrete CPD |
| `DiscreteFactor` | Generic discrete |
| `JointProbabilityDistribution` | JPD |
| `NoisyOR` | Noisy-OR CPD |
| `LinearGaussianCPD` | Continuous |
| `FunctionalCPD` | Arbitrary function |

### Estimators (`pgmpy/estimators/`)
| Class | Type | Method |
|-------|------|--------|
| `MLE` | Param | Maximum Likelihood |
| `BayesianEstimator` | Param | Bayesian |
| `EM` | Param | Expectation-Max |
| `PC` | Struct | Constraint-based |
| `HillClimbSearch` | Struct | Score-based |
| `GES` | Struct | Greedy Equiv. |
| `ExhaustiveSearch` | Struct | Brute force |
| `TreeSearch` | Struct | Tree structure |
| `MmhcEstimator` | Struct | Hybrid |

### Inference (`pgmpy/inference/`)
| Class | Type |
|-------|------|
| `VariableElimination` | Exact |
| `BeliefPropagation` | Exact |
| `DBNInference` | Exact (DBN) |
| `ApproxInference` | Sampling |
| `CausalInference` | Causal/do() |
| `Mplp` | Message passing |

## 🔧 Commands Quick Reference

| Task | Command |
|------|---------|
| Install (dev) | `pip install -e .[tests]` |
| Install (all) | `pip install -e .[all,tests]` |
| Test all | `pytest` |
| Test specific | `pytest pgmpy/tests/test_<module>.py` |
| Test verbose | `pytest -v` |
| Coverage | `pytest --cov=pgmpy` |
| Lint | `ruff check .` |
| Format | `ruff format .` |

## 🚫 Don't Do This

| ❌ Don't | ✅ Do |
|----------|------|
| Use `unittest` | Use `pytest` |
| Skip type hints | Add hints to public methods |
| Loop over DataFrame rows | Use vectorized operations |
| Add core dependencies | Use optional extras |
| Delete existing tests | Add more tests |
| Change public API silently | Discuss in issue/PR first |
| Commit without testing | Run `ruff check . && pytest` |

## 🔄 Standard Workflow
```bash
# 1. Create feature branch
git checkout -b feature/your-feature dev

# 2. Make changes with tests
# ... edit files ...

# 3. Verify
ruff check .
pytest

# 4. Commit
git commit -m "feat: add new feature"

# 5. Push and create PR against dev
git push -u origin feature/your-feature
```

## 📊 CI/CD
| Workflow | File | Purpose |
|----------|------|---------|
| Tests | `ci.yml` | Py 3.10-3.14, all OS |
| Lint | `lint.yml` | Ruff checks |
| Security | `codeql.yml` | Vulnerability scan |
