<div align="center">

<a href="https://pgmpy.org/"><img src="https://raw.githubusercontent.com/pgmpy/pgmpy/dev/logo/logo_color.png" width="175" alt="pgmpy logo"></a>

**Python toolkit for causal and probabilistic reasoning with Bayesian networks**

</div>

pgmpy helps you answer cause-and-effect questions from data: *What causes what? What do I need to adjust for to estimate
the effect of X on Y? What happens to Y if we intervene on X?* Write down your assumptions as a causal graph, or learn
one from data. Give each variable a conditional probability distribution (CPD) and the graph becomes a causal Bayesian
network, which answers both probabilistic questions (*given what we observed, how likely is Y?*) and causal ones (*what
happens to Y if we set X?*). Give each variable a structural equation with explicit noise instead and it becomes a
structural causal model (SCM), which can also answer counterfactual questions (*what would have happened to this unit
had X been different?*). SCM support is experimental and under active development.

|  | **[Documentation](https://pgmpy.org/)** · **[Examples](https://pgmpy.org/examples.html)** · **[Tutorials](https://github.com/pgmpy/pgmpy_tutorials)** |
|---|---|
| **Open&#160;Source** | [![GitHub License](https://img.shields.io/github/license/pgmpy/pgmpy)](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE) |
| **Tutorials** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pgmpy/pgmpy/dev?filepath=examples)
| **Community** | [![Discord Online](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fguilds%2F1248540985894633492%2Fwidget.json&query=%24.presence_count&label=online&logo=discord&style=flat-square&labelColor=E0E3FF&color=5865F2&logoColor=white)](https://discord.gg/DRkdKaumBs) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/pgmpy/)  |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/pgmpy/pgmpy/ci.yml?logo=github)](https://github.com/pgmpy/pgmpy/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/pgmpy/pgmpy/graph/badge.svg?token=UaJMCdHaEF)](https://codecov.io/gh/pgmpy/pgmpy) [![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](http://pgmpy.org/pgmpy-benchmarks/) [![platform](https://img.shields.io/conda/pn/conda-forge/pgmpy)](https://github.com/pgmpy/pgmpy) |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/pgmpy?color=orange)](https://pypi.org/project/pgmpy/) [![!conda](https://img.shields.io/conda/vn/conda-forge/pgmpy)](https://anaconda.org/conda-forge/pgmpy) [![!python-versions](https://img.shields.io/pypi/pyversions/pgmpy)](https://www.python.org/) [![!ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)  |
| **Downloads** |  ![PyPI - Downloads](https://img.shields.io/pypi/dm/pgmpy) [![Downloads](https://static.pepy.tech/personalized-badge/pgmpy?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/pgmpy) |
| **Supported By** | <a href="https://nlnet.nl/commonsfund/"><img src="https://nlnet.nl/image/logos/NGI0CommonsFund_tag_black_mono.svg" height="20" alt="NGI Zero Commons"></a> <a href="https://www.openscience.nl/en/news/45-projects-strengthen-dutch-open-science-infrastructure"><img src="https://www.openscience.nl/sites/open_science/files/media-files/L1_%5BZWART_DONKERPAARS%5D_OSNL_LOGO_GROOT.png" height="20" alt="Open Science NL"> </a> [![GC.OS Sponsored](https://img.shields.io/badge/GC.OS-Sponsored%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) [![FLOSS/FUND](https://floss.fund//static/badge.svg)](https://floss.fund/) [![Affiliated with NumFOCUS](https://img.shields.io/badge/NumFOCUS-affiliated%20project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects) |

## What you can do with pgmpy

<p align="center">
  <a href="https://pgmpy.org/documentation.html">
    <img src="https://raw.githubusercontent.com/pgmpy/pgmpy/dev/docs/_static/images/pgmpy_workflow.svg" width="820"
      alt="The pgmpy workflow. Learn a causal graph from data or draw it from expert knowledge, and test it against
      the data. Use the graph to identify what to adjust for and estimate the effect of X on Y, or fit it into a causal
      model, a Bayesian network or an experimental structural causal model, to answer seeing, doing and imagining
      questions and to simulate data.">
  </a>
</p>

| Your question | What pgmpy does | Guide |
|---|---|---|
| What causes what? | Causal discovery from data, optionally guided by expert knowledge | [Causal discovery](https://pgmpy.org/guides/causal_discovery.html) |
| Does my causal graph fit the data? | Model testing: checks the independencies the graph implies, or compares it with a known graph | [Metrics](https://pgmpy.org/guides/metrics.html) |
| What do I need to adjust for? | Causal identification: adjustment sets and the front-door criterion | [Causal identification](https://pgmpy.org/guides/causal_identification.html) |
| How large is the effect of X on Y? | Causal effect estimation with adjustment, instrumental variables or double machine learning | [Causal estimation](https://pgmpy.org/guides/causal_estimation.html) |
| How does each variable depend on its causes? | Parameter learning: fits a CPD for each variable | [Parameter estimation](https://pgmpy.org/guides/parameter_estimation.html) |
| Given what we observed, how likely is Y? | Probabilistic inference, P(Y \| evidence), exact or approximate | [Probabilistic inference](https://pgmpy.org/guides/probabilistic_inference.html) |
| What happens to Y if we set X? | Interventional queries, P(Y \| do(X)) | [Causal estimation](https://pgmpy.org/guides/causal_estimation.html) |
| What would have happened had X been different? | Counterfactuals with structural causal models (experimental) | – |
| What would the data look like under an intervention? | Simulation of observational and interventional data | [Simulations](https://pgmpy.org/guides/simulations.html) |

pgmpy builds on pandas, NetworkX and scikit-learn, and follows scikit-learn's `fit` / `predict` conventions where it
can. It also supports dynamic Bayesian networks, Markov networks and factor graphs;
[reads and writes](https://pgmpy.org/guides/io.html) BIF, XMLBIF, UAI, NET and XDSL files; and comes with 50
[datasets](https://pgmpy.org/guides/datasets.html) and over 250
[example models](https://pgmpy.org/guides/example_models.html) to practice on.

## Installation

pgmpy is available on [PyPI](https://pypi.org/project/pgmpy/) and [conda-forge](https://anaconda.org/conda-forge/pgmpy):

```bash
pip install pgmpy
```

```bash
conda install conda-forge::pgmpy
```

Some features need optional dependencies: PyTorch and Pyro for functional Bayesian networks, Matplotlib and Graphviz
for plotting, LiteLLM for LLM-assisted causal discovery, and skpro for the experimental structural causal models. To
install all of them, use `pip install "pgmpy[optional]"`. See the
[installation guide](https://pgmpy.org/started/install.html) for details.

## Quickstart: seeing vs. doing

The same Bayesian network answers two different questions. *Seeing:* how likely is Akt to be at level 3 in cells where
we observe Erk at level 3? *Doing:* how likely is it if we set Erk to level 3, for example with a drug? The data are
protein-signalling measurements from human immune cells (Sachs et al., 2005), discretized into levels 1 to 3, and the
graph is the signalling network known from the biology literature.

```python
>>> from pgmpy.datasets import load_dataset
>>> from pgmpy.inference import CausalInference, VariableElimination
>>> from pgmpy.models import DiscreteBayesianNetwork

>>> # Fit a Bayesian network on the known causal graph: graph + data = causal Bayesian network.
>>> sachs = load_dataset("sachs_discrete")
>>> model = DiscreteBayesianNetwork(sachs.ground_truth.edges()).fit(sachs.data)

>>> seeing = VariableElimination(model).query(["akt"], evidence={"erk": 3})  # observe Erk at level 3
>>> doing = CausalInference(model).query(["akt"], do={"erk": 3})  # set Erk to level 3
>>> print(f"P(akt=3 | erk=3) = {seeing.get_value(akt=3):.2f}, P(akt=3 | do(erk=3)) = {doing.get_value(akt=3):.2f}")
P(akt=3 | erk=3) = 0.31, P(akt=3 | do(erk=3)) = 0.18

```

Observing Erk at level 3 makes Akt at level 3 more likely (0.31) than setting Erk does (0.18), because part of their
association comes from PKA, which regulates both. `VariableElimination` answers the seeing question with probabilistic
inference, and `CausalInference` answers the doing question by removing that confounding.

### More examples

<details>
<summary><b>Learn a causal graph from data</b></summary>

```python
>>> from pgmpy.causal_discovery import PC
>>> from pgmpy.example_models import load_model
>>> from pgmpy.metrics import SHD

>>> # Simulate 5,000 records from ALARM, a 37-variable patient-monitoring network, and try to recover its graph.
>>> alarm = load_model("bnlearn/alarm")
>>> data = alarm.simulate(n_samples=5000, seed=42)
>>> pc = PC().fit(data)  # picks a conditional independence test that suits the data
>>> SHD()(true_causal_graph=alarm, est_causal_graph=pc.causal_graph_)  # edges to add, remove or reverse
14

```

To guide the search with what you already know, pass required or forbidden edges, or a temporal order, through
`ExpertKnowledge`. Without a known graph to compare against, test the learned graph against the data with the
[unsupervised metrics](https://pgmpy.org/guides/metrics.html).

</details>

<details>
<summary><b>Estimate a causal effect from data</b></summary>

```python
>>> import numpy as np
>>> import pandas as pd
>>> from pgmpy.base import DAG
>>> from pgmpy.identification import Adjustment
>>> from pgmpy.inference import CausalInference

>>> # Simulated data: older patients get higher doses and recover less well. The true effect of dose is 2.
>>> rng = np.random.default_rng(42)
>>> age = rng.normal(50, 10, size=2000)
>>> dose = 0.1 * age + rng.normal(size=2000)
>>> recovery = 2 * dose - 0.3 * age + rng.normal(size=2000)
>>> data = pd.DataFrame({"age": age, "dose": dose, "recovery": recovery})

>>> graph = DAG(
...     [("age", "dose"), ("age", "recovery"), ("dose", "recovery")],
...     roles={"exposures": "dose", "outcomes": "recovery"},
... )
>>> identified_graph, is_identified = Adjustment().identify(graph)
>>> identified_graph.get_role("adjustment")
['age']
>>> naive = np.polyfit(data["dose"], data["recovery"], deg=1)[0]
>>> adjusted = CausalInference(graph).estimate_ate("dose", "recovery", data=data)
>>> print(f"naive: {naive:.2f}, adjusted for age: {adjusted:.2f}")
naive: 0.51, adjusted for age: 2.04

```

</details>

<details>
<summary><b>Model continuous and mixed data</b></summary>

A linear Gaussian Bayesian network for continuous data:

```python
>>> import numpy as np
>>> from pgmpy.example_models import load_model
>>> from pgmpy.models import LinearGaussianBayesianNetwork

>>> # Fit to 1,000 records simulated from a 46-gene E. coli network, then predict one gene from the others.
>>> ecoli = load_model("bnlearn/ecoli70")
>>> data = ecoli.simulate(n_samples=1000, seed=42)
>>> model = LinearGaussianBayesianNetwork(ecoli.edges()).fit(data)
>>> predicted = model.predict(data.drop(columns="ftsJ"))
>>> print(f"correlation with the true values: {np.corrcoef(predicted['ftsJ'], data['ftsJ'])[0, 1]:.2f}")
correlation with the true values: 0.96

```

A functional Bayesian network for mixed data, with any [Pyro](https://pyro.ai/) distribution for each variable:

```python
>>> import pyro.distributions as dist
>>> from pgmpy import config
>>> from pgmpy.factors.hybrid import FunctionalCPD
>>> from pgmpy.models import FunctionalBayesianNetwork

>>> config.set_backend("torch")
>>> # x and w are continuous, y is binary.
>>> model = FunctionalBayesianNetwork([("x", "w"), ("x", "y"), ("w", "y")])
>>> model.add_cpds(
...     FunctionalCPD("x", fn=lambda _: dist.Normal(0.0, 1.0)),
...     FunctionalCPD("w", fn=lambda p: dist.Normal(0.7 * p["x"], 0.5), parents=["x"]),
...     FunctionalCPD("y", fn=lambda p: dist.Bernoulli(logits=-0.7 + 1.5 * p["x"] + 1.2 * p["w"]), parents=["x", "w"]),
... )
>>> observed = model.simulate(n_samples=1000, seed=123)
>>> intervened = model.simulate(n_samples=1000, seed=123, do={"w": 2.0})
>>> print(f"P(y=1) = {observed['y'].mean():.2f}, P(y=1 | do(w=2)) = {intervened['y'].mean():.2f}")
P(y=1) = 0.40, P(y=1 | do(w=2)) = 0.77
>>> config.set_backend("numpy")

```

To learn functional Bayesian networks from data and run inference on them, see the
[tutorial notebook](https://github.com/pgmpy/pgmpy/blob/dev/examples/Functional_Bayesian_Network_Tutorial.ipynb).

</details>

## Learn more

- **New to causal inference?** The [tutorials](https://github.com/pgmpy/pgmpy_tutorials) teach causal inference with
  pgmpy step by step.
- **Want a quick tour?** The [quickstart](https://pgmpy.org/started/quickstart.html) has a minimal example for every
  task.
- **Know what you need?** Go to the [user guide](https://pgmpy.org/documentation.html), the
  [example notebooks](https://pgmpy.org/examples.html) (also runnable on
  [Binder](https://mybinder.org/v2/gh/pgmpy/pgmpy/dev?filepath=examples)) or the
  [API reference](https://pgmpy.org/reference.html).
- **Blog posts** are on [Medium](https://medium.com/@ankurankan_23083).

**Upgrading from pgmpy 1.0?** Since 1.1, the classes in `pgmpy.estimators` have new homes in `pgmpy.causal_discovery`,
`pgmpy.parameter_estimator`, `pgmpy.structure_score` and `pgmpy.ci_tests`. The old imports still work but warn, and
will be removed in 2.0. See the [changelog](https://github.com/pgmpy/pgmpy/blob/dev/CHANGELOG.md) for details.

## Community and support

- **Questions:** ask on [Discord](https://discord.gg/DRkdKaumBs) or
  [Stack Overflow](https://stackoverflow.com/questions/tagged/pgmpy).
- **Bug reports and feature requests:** open a [GitHub issue](https://github.com/pgmpy/pgmpy/issues).
- **News:** follow pgmpy on [LinkedIn](https://www.linkedin.com/company/pgmpy/).

## Citing pgmpy

If you use pgmpy in your scientific work, please cite:

> Ankur Ankan and Johannes Textor (2024). pgmpy: A Python Toolkit for Bayesian Networks. *Journal of Machine Learning
> Research*, 25(265), 1–8.

```bibtex
@article{Ankan2024,
  author  = {Ankur Ankan and Johannes Textor},
  title   = {pgmpy: A Python Toolkit for Bayesian Networks},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {265},
  pages   = {1--8},
  url     = {http://jmlr.org/papers/v25/23-0487.html}
}
```

## Contributing

We welcome all contributions to pgmpy, not just code. Please read our
[contributing guide](https://github.com/pgmpy/pgmpy/blob/dev/CONTRIBUTING.md) to get started. To add a new algorithm,
metric, dataset or example model, start from the matching
[extension template](https://github.com/pgmpy/pgmpy/tree/dev/devtools/extension_templates). We also mentor new
contributors and keep a list of [mentored projects](https://github.com/pgmpy/pgmpy/wiki/Mentored-Projects). If you are
interested, join our [Discord](https://discord.gg/DRkdKaumBs) server and introduce yourself.
