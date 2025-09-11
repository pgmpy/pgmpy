<div>

<a href="https://www.pgmpy.org"><img src="https://raw.githubusercontent.com/pgmpy/pgmpy/dev/logo/logo_color.png" width="175" align="left" /></a>
pgmpy is a Python library for causal and probabilistic modeling using graphical models. It provides a uniform API for building, learning, and analyzing models such as Bayesian Networks, Dynamic Bayesian Networks, Directed Acyclic Graphs (DAGs), and Structural Equation Models(SEMs). By integrating tools from both probabilistic inference and causal inference, pgmpy enables users to seamlessly transition between predictive and interventional analyses.
</div>


|  | **[Documentation](https://pgmpy.org/)** · **[Examples](https://pgmpy.org/examples.html)** . **[Tutorials](https://github.com/pgmpy/pgmpy_tutorials)** |
|---|---|
| **Open&#160;Source** | [![GitHub License](https://img.shields.io/github/license/pgmpy/pgmpy)](https://github.com/pgmpy/pgmpy/blob/main/LICENSE) [![GC.OS Sponsored](https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) |
| **Tutorials** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pgmpy/pgmpy/dev?filepath=examples)
| **Community** | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.gg/DRkdKaumBs) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/pgmpy/)  |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/pgmpy/pgmpy/ci.yml?logo=github)](https://github.com/pgmpy/pgmpy/actions/workflows/ci.yml) [![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](http://pgmpy.org/pgmpy-benchmarks/) [![platform](https://img.shields.io/conda/pn/conda-forge/pgmpy)](https://github.com/pgmpy/pgmpy) |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/pgmpy?color=orange)](https://pypi.org/project/pgmpy/) [![!conda](https://img.shields.io/conda/vn/conda-forge/pgmpy)](https://anaconda.org/conda-forge/pgmpy) [![!python-versions](https://img.shields.io/pypi/pyversions/pgmpy)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  |
| **Downloads** | ![PyPI - Downloads](https://img.shields.io/pypi/dw/pgmpy) ![PyPI - Downloads](https://img.shields.io/pypi/dm/pgmpy) [![Downloads](https://static.pepy.tech/personalized-badge/pgmpy?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/pgmpy) |

## Key Features

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
### Examples
#### Discrete Data
```python
from pgmpy.utils import get_example_model

# Load a Discrete Bayesian Network and simulate data.
discrete_bn = get_example_model("alarm")
alarm_df = discrete_bn.simulate(n_samples=100)

# Learn a network from simulated data.
from pgmpy.estimators import PC

dag = PC(data=alarm_df).estimate(ci_test="chi_square", return_type="dag")

# Learn the parameters from the data.
dag_fitted = dag.fit(alarm_df)
dag_fitted.get_cpds()

# Drop a column and predict using the learned model.
evidence_df = alarm_df.drop(columns=["FIO2"], axis=1)
pred_FIO2 = dag_fitted.predict(evidence_df)
```

#### Linear Gaussian Data
```python
# Load an example Gaussian Bayesian Network and simulate data
gaussian_bn = get_example_model("ecoli70")
ecoli_df = gaussian_bn.simulate(n_samples=100)

# Learn the network from simulated data.
from pgmpy.estimators import PC

dag = PC(data=ecoli_df).estimate(ci_test="pearsonr", return_type="dag")

# Learn the parameters from the data.
from pgmpy.models import LinearGausianBayesianNetwork

gaussian_bn = LinearGausianBayesianNetwork(dag.edges())
dag_fitted = gaussian_bn.fit(ecoli_df)
dag_fitted.get_cpds()

# Drop a column and predict using the learned model.
evidence_df = ecoli_df.drop(columns=["ftsJ"], axis=1)
pred_ftsJ = dag_fitted.predict(evidence_df)
```

#### Mixture Data with Arbitrary Relationships
```python
import pyro.distributions as dist

from pgmpy.models import FunctionalBayesianNetwork
from pgmpy.factors.hybrid import FunctionalCPD

# Create a Bayesian Network with mixture of discrete and continuous variables.
func_bn = FunctionalBayesianNetwork(
    [
        ("x1", "w"),
        ("x2", "w"),
        ("x1", "y"),
        ("x2", "y"),
        ("w", "y"),
        ("y", "z"),
        ("w", "z"),
        ("y", "c"),
        ("w", "c"),
    ]
)

# Define the Functional CPDs for each node and add them to the model.
cpd_x1 = FunctionalCPD("x1", fn=lambda _: dist.Normal(0.0, 1.0))
cpd_x2 = FunctionalCPD("x2", fn=lambda _: dist.Normal(0.5, 1.2))

# Continuous mediator: w = 0.7*x1 - 0.3*x2 + ε
cpd_w = FunctionalCPD(
    "w",
    fn=lambda parents: dist.Normal(0.7 * parents["x1"] - 0.3 * parents["x2"], 0.5),
    parents=["x1", "x2"],
)

# Bernoulli target with logistic link: y ~ Bernoulli(sigmoid(-0.7 + 1.5*x1 + 0.8*x2 + 1.2*w))
cpd_y = FunctionalCPD(
    "y",
    fn=lambda parents: dist.Bernoulli(
        logits=(-0.7 + 1.5 * parents["x1"] + 0.8 * parents["x2"] + 1.2 * parents["w"])
    ),
    parents=["x1", "x2", "w"],
)

# Downstream Bernoulli influenced by y and w
cpd_z = FunctionalCPD(
    "z",
    fn=lambda parents: dist.Bernoulli(
        logits=(-1.2 + 0.8 * parents["y"] + 0.2 * parents["w"])
    ),
    parents=["y", "w"],
)

# Continuous outcome depending on y and w: c = 0.2 + 0.5*y + 0.3*w + ε
cpd_c = FunctionalCPD(
    "c",
    fn=lambda parents: dist.Normal(0.2 + 0.5 * parents["y"] + 0.3 * parents["w"], 0.7),
    parents=["y", "w"],
)

func_bn.add_cpds(cpd_x1, cpd_x2, cpd_w, cpd_y, cpd_z, cpd_c)
func_bn.check_model()

# Simulate data from the model
df_func = func_bn.simulate(n_samples=1000, seed=123)

# For learning and inference in Functional Bayesian Networks, please refer to the example notebook: https://github.com/pgmpy/pgmpy/blob/dev/examples/Functional_Bayesian_Network_Tutorial.ipynb
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
