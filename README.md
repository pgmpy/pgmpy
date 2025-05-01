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
[![!conda](https://img.shields.io/conda/vn/conda-forge/pgmpy)](https://anaconda.org/conda-forge/pgmpy)
[![Python Version](https://img.shields.io/pypi/pyversions/pgmpy.svg?color=blue)](https://pypi.org/project/pgmpy/)
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

### Quick Links
- **Documentation:** https://pgmpy.org/
- **Installation:** https://pgmpy.org/started/install.html
- **Mailing List:** https://groups.google.com/forum/#!forum/pgmpy
- **Community chat:** [discord](https://discord.gg/DRkdKaumBs)

Contributing
============
We welcome all contributions --not just code-- to pgmpy. Please refer out
[contributing guide](https://github.com/pgmpy/pgmpy/blob/dev/Contributing.md)
for more details. We also offer mentorship for new contributors and maintain a
list of potential [mentored
projects](https://github.com/pgmpy/pgmpy/wiki/Mentored-Projects). If you are
interested in contributing to pgmpy, please join our
[discord](https://discord.gg/DRkdKaumBs) server and introduce yourself. We will
be happy to help you get started.
