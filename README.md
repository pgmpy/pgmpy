<div>

<a href="https://www.pgmpy.org"><img src="https://raw.githubusercontent.com/pgmpy/pgmpy/dev/logo/logo_color.png" width="175" align="left" /></a>
pgmpy is a Python library for causal and probabilistic modeling using Bayesian Networks and related models. It provides a uniform API for building, learning, and analyzing models such as Bayesian Networks, Dynamic Bayesian Networks, Directed Acyclic Graphs (DAGs), and Structural Equation Models(SEMs). By integrating tools from both probabilistic inference and causal inference, pgmpy enables users to seamlessly transition between predictive and interventional analyses.
</div>

<br/>
<br/>

<div align="center">

![Build](https://github.com/pgmpy/pgmpy/actions/workflows/ci.yml/badge.svg?branch=dev)
[![codecov](https://codecov.io/gh/pgmpy/pgmpy/branch/dev/graph/badge.svg?token=UaJMCdHaEF)](https://codecov.io/gh/pgmpy/pgmpy)
[![Version](https://img.shields.io/pypi/v/pgmpy?color=blue)](https://pypi.org/project/pgmpy/)
[![!conda](https://img.shields.io/conda/vn/conda-forge/pgmpy)](https://anaconda.org/conda-forge/pgmpy) [![Python Version](https://img.shields.io/pypi/pyversions/pgmpy.svg?color=blue)](https://pypi.org/project/pgmpy/)
[![License](https://img.shields.io/github/license/pgmpy/pgmpy)](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/pgmpy.svg)](https://pypistats.org/packages/pgmpy)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](http://pgmpy.org/pgmpy-benchmarks/)

</div>

<div align="center">

[![Join the pgmpy Discord server](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/DRkdKaumBs)
[![Read the Docs](https://img.shields.io/badge/-Docs-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://inseq.org)](https://pgmpy.org)
[![Examples](https://img.shields.io/badge/-Examples-orange?style=for-the-badge&logo=Jupyter&logoColor=white&link=https://github.com/pgmpy/pgmpy/tree/dev/examples)](https://github.com/pgmpy/pgmpy/tree/dev/examples)
[![Tutorial](https://img.shields.io/badge/-Tutorial-orange?style=for-the-badge&logo=Jupyter&logoColor=white&link=https://github.com/pgmpy/pgmpy_notebook)](https://github.com/pgmpy/pgmpy_notebook)

</div>

### Key Features

| Feature | Description |
|--------|-------------|
| [**Causal Discovery / Structure Learning**](https://pgmpy.org/examples/Structure%20Learning%20in%20Bayesian%20Networks.html) | Learn the model structure from data, with optional integration of **expert knowledge**. |
| [**Causal Validation**](https://pgmpy.org/metrics/metrics.html) | Assess how compatible the causal structure is with the data. |
| [**Parameter Learning**](https://pgmpy.org/examples/Learning%20Parameters%20in%20Discrete%20Bayesian%20Networks.html) | Estimate model parameters (e.g., conditional probability distributions) from observed data. |
| [**Probabilistic Inference**](https://pgmpy.org/examples/Inference%20in%20Discrete%20Bayesian%20Networks.html) | Compute posterior distributions conditioned on observed evidence. |
| [**Causal Inference**](https://pgmpy.org/examples/Causal%20Inference.html) | Compute interventional and counterfactual distributions using do-calculus. |
| [**Simulations**](https://github.com/pgmpy/pgmpy/blob/dev/examples/Simulating_Data.ipynb) | Generate synthetic data under specified evidence or interventions. |

### Resources and Links
- **Example Notebooks:** [Examples](https://github.com/pgmpy/pgmpy/tree/dev/examples)
- **Tutorial Notebooks:** [Tutorials](https://github.com/pgmpy/pgmpy_notebook)
- **Blog Posts:** [Medium](https://medium.com/@ankurankan_23083)
- **Documentation:** [Website](https://pgmpy.org/)
- **Bug Reports and Feature Requests:** [GitHub Issues](https://github.com/pgmpy/pgmpy/issues)
- **Questions:** [discord](https://discord.gg/DRkdKaumBs) · [Stack Overflow](https://stackoverflow.com/questions/tagged/pgmpy)

## Quickstart

### Installation
pgmpy is available on both [PyPI](https://pypi.org/project/pgmpy/) and [anaconda](https://anaconda.org/conda-forge/pgmpy). To install from PyPI, use:

```bash
pip install pgmpy
```
To install from conda-forge, use:

```bash
conda install conda-forge::pgmpy
```

### Discrete Data
```python
from pgmpy.utils import get_example_model

# Load a Discrete Bayesian Network and simulate data.
discrete_bn = get_example_model('alarm')
alarm_df = discrete_bn.simulate(n_samples=100)

# Learn a network from simulated data.
from pgmpy.estimators import PC
dag = PC(data=alarm_df).estimate(ci_test='chi_square', return_type='dag')

# Learn the parameters from the data.
dag_fitted = dag.fit(alarm_df)
dag_fitted.get_cpds()

# Drop a column and predict using the learned model.
evidence_df = alarm_df.drop(columns=['FIO2'], axis=1)
pred_FIO2 = dag_fitted.predict(evidence_df)
```

### Linear Gaussian Data
```python
# Load an example Gaussian Bayesian Network and simulate data
gaussian_bn = get_example_model('ecoli70')
ecoli_df = gaussian_bn.simulate(n_samples=100)

# Learn the network from simulated data.
from pgmpy.estimators import PC
dag = PC(data=ecoli_df).estimate(ci_test='pearsonr', return_type='dag')

# Learn the parameters from the data.
from pgmpy.models import LinearGausianBayesianNetwork
gaussian_bn = LinearGausianBayesianNetwork(dag.edges())
dag_fitted = gaussian_bn.fit(ecoli_df)
dag_fitted.get_cpds()

# Drop a column and predict using the learned model.
evidence_df = ecoli_df.drop(columns=['ftsJ'], axis=1)
pred_ftsJ = dag_fitted.predict(evidence_df)
```

## Contributing

We welcome all contributions --not just code-- to pgmpy. Please refer out
[contributing guide](https://github.com/pgmpy/pgmpy/blob/dev/Contributing.md)
for more details. We also offer mentorship for new contributors and maintain a
list of potential [mentored
projects](https://github.com/pgmpy/pgmpy/wiki/Mentored-Projects). If you are
interested in contributing to pgmpy, please join our
[discord](https://discord.gg/DRkdKaumBs) server and introduce yourself. We will
be happy to help you get started.
