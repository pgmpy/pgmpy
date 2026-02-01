# Architecture Documentation - pgmpy

`pgmpy` is a Python library for Probabilistic Graphical Models and Causal Inference.

## ⚡ Quick Navigation

| Need | Location |
|------|----------|
| Add model | `pgmpy/models/` |
| Add factor | `pgmpy/factors/` |
| Add learning algo | `pgmpy/estimators/` |
| Add inference | `pgmpy/inference/` |
| Add I/O | `pgmpy/readwrite/` |
| Add tests | `pgmpy/tests/test_<module>/` |

## 🏗️ Project Structure
```
pgmpy/
├── base/            # Graph foundations
│   ├── DAG.py       # Directed Acyclic Graph
│   ├── PDAG.py      # Partially Directed
│   ├── MAG.py       # Maximal Ancestral
│   ├── ADMG.py      # Acyclic Directed Mixed
│   └── UndirectedGraph.py
│
├── models/          # Graphical models
│   ├── DiscreteBayesianNetwork.py
│   ├── LinearGaussianBayesianNetwork.py
│   ├── FunctionalBayesianNetwork.py
│   ├── DynamicBayesianNetwork.py
│   ├── DiscreteMarkovNetwork.py
│   ├── FactorGraph.py
│   ├── ClusterGraph.py
│   ├── JunctionTree.py
│   ├── SEM.py
│   ├── MarkovChain.py
│   └── NaiveBayes.py
│
├── factors/         # Probability distributions
│   ├── discrete/    # TabularCPD, DiscreteFactor, JPD, NoisyOR
│   ├── continuous/  # LinearGaussianCPD
│   └── hybrid/      # FunctionalCPD
│
├── estimators/      # Learning algorithms
│   ├── MLE.py                # Parameter (ML)
│   ├── BayesianEstimator.py  # Parameter (Bayesian)
│   ├── EM.py                 # Parameter (EM)
│   ├── PC.py                 # Structure (Constraint)
│   ├── HillClimbSearch.py    # Structure (Score)
│   ├── GES.py                # Structure (Score)
│   ├── TreeSearch.py         # Structure (Tree)
│   └── MmhcEstimator.py      # Structure (Hybrid)
│
├── inference/       # Reasoning engines
│   ├── ExactInference.py     # VariableElimination, BP
│   ├── ApproxInference.py    # Sampling methods
│   ├── CausalInference.py    # do-calculus
│   └── dbn_inference.py      # DBN-specific
│
├── readwrite/       # I/O formats
│   ├── BIF.py, XMLBIF.py, UAI.py, PomdpX.py
│
├── sampling/        # Data generation
└── tests/           # Mirrors main structure
```

## 🧠 Core Design Principles

| Principle | Description |
|-----------|-------------|
| **Separation** | Models = structure, Factors = distributions, Inference = logic |
| **Graph-Centric** | All models built on NetworkX graphs |
| **Data-Driven** | Estimators decoupled from models |
| **Extensible** | Easy to add new components |

## 📊 Component Relationships

```
Data (DataFrame)
     │
     ▼
┌─────────────┐     ┌─────────────┐
│  Estimators │────▶│   Models    │
│  (learn)    │     │ (structure) │
└─────────────┘     └──────┬──────┘
                           │
                           ▼
                   ┌─────────────┐
                   │   Factors   │
                   │ (CPDs/dists)│
                   └──────┬──────┘
                           │
                           ▼
                   ┌─────────────┐
                   │  Inference  │◀── Evidence
                   │  (queries)  │
                   └──────┬──────┘
                           │
                           ▼
                    Posteriors / Predictions
```

## 🎯 Inheritance Hierarchy

### Models
```
NetworkX Graph
    │
    ├── DAG (pgmpy.base)
    │   ├── DiscreteBayesianNetwork
    │   ├── LinearGaussianBayesianNetwork
    │   ├── FunctionalBayesianNetwork
    │   ├── DynamicBayesianNetwork
    │   └── NaiveBayes
    │
    └── UndirectedGraph (pgmpy.base)
        ├── DiscreteMarkovNetwork
        ├── FactorGraph
        └── ClusterGraph
```

### Estimators
```
BaseEstimator
    │
    ├── ParameterEstimator
    │   ├── MLE
    │   ├── BayesianEstimator
    │   └── EM
    │
    └── StructureEstimator
        ├── PC (constraint-based)
        ├── HillClimbSearch (score-based)
        ├── GES (score-based)
        └── TreeSearch
```

## 🔄 Data Flow

| Stage | Input | Output |
|-------|-------|--------|
| Learning | DataFrame | Model with CPDs |
| Inference | Model + Evidence | Posteriors |
| Simulation | Model | DataFrame |
| Causal | Model + Intervention | Causal effects |

## 🔧 CI/CD Workflows

| File | Triggers | Matrix |
|------|----------|--------|
| `ci.yml` | push, PR | Py 3.10-3.14 × Ubuntu/Win/macOS |
| `lint.yml` | push, PR | Ruff checks |
| `codeql.yml` | push, PR | Security analysis |

## 📝 File Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `test_*.py` | Test files |
| `*CPD.py` | CPD implementations |
| `*Network.py` | Model implementations |
| `*Estimator.py` | Learning algorithms |
| `*Inference.py` | Inference algorithms |
