<div align="center">
  <img src="https://raw.githubusercontent.com/pgmpy/pgmpy/dev/logo/logo_color.png" width="318" height="300"/>
</div>
<br/>
<div align="center">

![Build](https://github.com/pgmpy/pgmpy/actions/workflows/ci.yml/badge.svg?branch=dev)
[![Downloads](https://img.shields.io/pypi/dm/pgmpy.svg)](https://pypistats.org/packages/pgmpy)
[![codecov](https://codecov.io/gh/pgmpy/pgmpy/branch/dev/graph/badge.svg?token=UaJMCdHaEF)](https://codecov.io/gh/pgmpy/pgmpy)
[![Version](https://img.shields.io/pypi/v/pgmpy?color=blue)](https://pypi.org/project/pgmpy/)
[![Python Version](https://img.shields.io/pypi/pyversions/pgmpy.svg?color=blue)](https://pypi.org/project/pgmpy/)
[![License](https://img.shields.io/github/license/pgmpy/pgmpy)](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](http://pgmpy.org/pgmpy-benchmarks/)


</div>

<div align="center">

[![Join the pgmpy Discord server](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/DRkdKaumBs)
[![Read the Docs](https://img.shields.io/badge/-Docs-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://inseq.org)](https://pgmpy.org)
[![Examples](https://img.shields.io/badge/-Examples-orange?style=for-the-badge&logo=Jupyter&logoColor=white&link=https://github.com/pgmpy/pgmpy/tree/dev/examples)](https://github.com/pgmpy/pgmpy/tree/dev/examples)
[![Tutorial](https://img.shields.io/badge/-Tutorial-orange?style=for-the-badge&logo=Jupyter&logoColor=white&link=https://github.com/pgmpy/pgmpy_notebook)](https://github.com/pgmpy/pgmpy_notebook)

</div>

pgmpy is a Python package for working with Bayesian Networks and related models such as Directed Acyclic Graphs, Dynamic Bayesian Networks, and Structural Equation Models. It combines features from causal inference and probabilistic inference literature to allow users to seamlessly work between them. It implements algorithms for structure learning, causal discovery, parameter estimation, probabilistic and causal inference, and simulations.

- **Documentation:** https://pgmpy.org/
- **Installation:** https://pgmpy.org/started/install.html
- **Mailing List:** https://groups.google.com/forum/#!forum/pgmpy .
- **Community chat:** [discord](https://discord.gg/DRkdKaumBs) (Older chat at: [gitter](https://gitter.im/pgmpy/pgmpy))


Examples
--------
- Creating a Bayesian Network: [view](https://pgmpy.org/examples/Creating%20a%20Discrete%20Bayesian%20Network.html) | <a target="_blank" href="https://colab.research.google.com/github/ankurankan/pgmpy/blob/dev/examples/Creating%20a%20Discrete%20Bayesian%20Network.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
- Structure Learning/Causal Discovery: [view](https://pgmpy.org/examples/Structure%20Learning%20in%20Bayesian%20Networks.html) | <a target="_blank" href="https://colab.research.google.com/github/ankurankan/pgmpy/blob/dev/examples/Structure%20Learning%20in%20Bayesian%20Networks.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
- Parameter Learning: [view](https://pgmpy.org/examples/Learning%20Parameters%20in%20Discrete%20Bayesian%20Networks.html) | <a target="_blank" href="https://colab.research.google.com/github/ankurankan/pgmpy/blob/dev/examples/Learning%20Parameters%20in%20Discrete%20Bayesian%20Networks.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
- Probabilistic Inference: [view](https://pgmpy.org/examples/Inference%20in%20Discrete%20Bayesian%20Networks.html) | <a target="_blank" href="https://colab.research.google.com/github/ankurankan/pgmpy/blob/dev/examples/Inference%20in%20Discrete%20Bayesian%20Networks.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
- Causal Inference: [view](https://pgmpy.org/examples/Causal%20Inference.html) | <a target="_blank" href="https://colab.research.google.com/github/https://pgmpy.org/examples/Causal%20Inference.html"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
- Extending pgmpy: [view](https://pgmpy.org/examples/Extending%20pgmpy.html) | <a target="_blank" href="https://colab.research.google.com/github/ankurankan/pgmpy/blob/dev/examples/Extending%20pgmpy.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

<br/>

- Full List of Examples: https://github.com/pgmpy/pgmpy/tree/dev/examples
- Tutorials: https://github.com/pgmpy/pgmpy_notebook/

Citing
======
If you use `pgmpy` in your scientific work, please consider citing us:

```
Ankur Ankan, & Johannes Textor (2024). pgmpy: A Python Toolkit for Bayesian Networks. Journal of Machine Learning Research, 25(265), 1–8.
```

Bibtex:
```
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

Development
============

Code
----
The latest codebase is available in the `dev` branch of the repository.

Building from Source
--------------------
To install pgmpy from the source code:
```
$ git clone https://github.com/pgmpy/pgmpy
$ cd pgmpy/
$ pip install -r requirements.txt
$ python setup.py install
```

To run the tests, you can use pytest:
```
$ pytest -v pgmpy
```

If you face any problems during installation let us know, via issues, mail or at our discord channel.

Contributing
------------
Please feel free to report any issues on GitHub: https://github.com/pgmpy/pgmpy/issues.

Before opening a pull request, please have a look at our [contributing guide](
https://github.com/pgmpy/pgmpy/blob/dev/Contributing.md) If you face any
problems in pull request, feel free to ask them on the mailing list or gitter.

If you would like to implement any new features, please have a discussion about it before starting to work on it.
If you are looking for some ideas for projects, we a list of **mentored projects** available at: https://github.com/pgmpy/pgmpy/wiki/Mentored-Projects.

Building Documentation
----------------------
We use sphinx to build the documentation. Please refer: https://github.com/pgmpy/pgmpy/wiki/Maintenance-Guide#building-docs for steps to build docs locally.



License
=======
pgmpy is released under MIT License. You can read about our license at [here](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE)
