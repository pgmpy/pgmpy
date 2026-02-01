# pgmpy Copilot Instructions

You are assisting in the development of `pgmpy`, a Python library for Causal and Probabilistic Modeling.

## ⚡ Quick Actions

| Task | Command/Location |
|------|-----------------|
| Run tests | `pytest pgmpy/tests/` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Install dev | `pip install -e .[tests]` |

## 🎯 Knowledge Map

### Models (pgmpy/models/)
| Type | Classes |
|------|---------|
| Bayesian | `DiscreteBayesianNetwork`, `LinearGaussianBayesianNetwork`, `FunctionalBayesianNetwork`, `NaiveBayes`, `DynamicBayesianNetwork` |
| Markov | `DiscreteMarkovNetwork`, `FactorGraph`, `ClusterGraph`, `JunctionTree` |
| Other | `SEM`, `MarkovChain` |

### Estimators (pgmpy/estimators/)
| Type | Classes |
|------|---------|
| Parameter | `MLE`, `BayesianEstimator`, `EM`, `MirrorDescentEstimator` |
| Structure (Score) | `HillClimbSearch`, `ExhaustiveSearch`, `GES` |
| Structure (Constraint) | `PC`, `MmhcEstimator` |

### Inference (pgmpy/inference/)
| Type | Classes |
|------|---------|
| Exact | `VariableElimination`, `BeliefPropagation`, `DBNInference` |
| Approximate | `ApproxInference` |
| Causal | `CausalInference` |

## 📐 Code Standards

### ✅ Always
- Type hints: `def method(self, data: pd.DataFrame) -> Factor:`
- Numpydoc docstrings
- F-strings: `f"Variable {var} not found"`
- Vectorization: `np.sum(data, axis=0)`
- Context managers: `with open(path) as f:`
- Pytest for tests

### ❌ Never
- `unittest` module
- String concatenation with `+`
- Explicit loops over DataFrame rows
- Missing docstrings
- New core dependencies (use optional extras)

## 🔍 Base Classes Reference
```
pgmpy/base/
├── DAG.py              → Directed Acyclic Graphs
├── PDAG.py             → Partially Directed
├── MAG.py              → Maximal Ancestral Graph
├── ADMG.py             → Acyclic Directed Mixed Graph
└── UndirectedGraph.py  → Undirected structures
```

## 🔄 Workflow
1. Branch from `dev`
2. Write tests first (TDD)
3. Implement feature
4. `ruff check . && pytest`
5. PR against `dev`
