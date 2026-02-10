.. pgmpy documentation master file

:hide-toc:
:hide-navigation:

.. raw:: html

   <div class="hero-grid">
     <div class="hero-logo">
       <img src="_static/logo.png" alt="pgmpy logo">
     </div>
     <div class="hero-text">
       <h1>Welcome to pgmpy</h1>
       <p class="hero-subtitle"><em>Python Library for Causal AI</em></p>
       <p>pgmpy is a Python package for causal inference and probabilistic inference
       using Directed Acyclic Graphs (DAGs) and Bayesian Networks with a focus on
       modularity and extensibility. Implementations of various algorithms for Causal
       Discovery (a.k.a, Structure Learning), Parameter Estimation, Approximate
       (Sampling Based) and Exact inference, and Causal Inference are available.</p>
       <div class="badge-container" style="text-align: left;">
         <a href="https://pypi.org/project/pgmpy/"><img src="https://img.shields.io/pypi/v/pgmpy?style=flat-square&amp;color=2E8B8E" alt="PyPI version"></a>
         <a href="https://anaconda.org/conda-forge/pgmpy"><img src="https://img.shields.io/conda/vn/conda-forge/pgmpy?style=flat-square&amp;color=2E8B8E" alt="Conda version"></a>
         <a href="https://github.com/pgmpy/pgmpy"><img src="https://img.shields.io/github/stars/pgmpy/pgmpy?style=flat-square&amp;color=2E8B8E" alt="GitHub stars"></a>
         <a href="http://jmlr.org/papers/v25/23-0487.html"><img src="https://img.shields.io/badge/JMLR-2024-009688?style=flat-square" alt="JMLR 2024"></a>
       </div>
     </div>
   </div>

Key Features
------------

.. grid:: 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards

   .. grid-item-card:: Causal Discovery and Structure Learning
      :link: structure_estimator/base
      :link-type: doc
      :class-card: sd-card-hover

      Learn causal structure from data.

   .. grid-item-card:: Parameter Estimation
      :link: param_estimator/base
      :link-type: doc
      :class-card: sd-card-hover

      Estimate model parameters with Maximum Likelihood, Bayesian estimation, or EM.

   .. grid-item-card:: Probabilistic Inference
      :link: infer/base
      :link-type: doc
      :class-card: sd-card-hover

      Run exact inference (Variable Elimination, Belief Propagation) or approximate
      inference (sampling, Gibbs).

   .. grid-item-card:: Causal Inference
      :link: causal_infer/base
      :link-type: doc
      :class-card: sd-card-hover

      Perform interventional and counterfactual queries using do-calculus, backdoor,
      and frontdoor adjustment.

   .. grid-item-card:: Causal Identification
      :link: doc/causal_identification
      :link-type: doc
      :class-card: sd-card-hover

      Determine whether a causal effect is identifiable from observational data
      given the graph structure.

   .. grid-item-card:: Example Datasets and Models
      :link: examples
      :link-type: doc
      :class-card: sd-card-hover

      Explore built-in example Bayesian Networks and datasets to quickly prototype
      and test workflows.

Workflow
--------

.. figure:: pgmpy_workflow.png
   :alt: Possible Workflows in pgmpy for Directed Acyclic Graphs (DAGs) and Bayesian Networks (BNs).

   Possible Workflows in pgmpy for Directed Acyclic Graphs (DAGs) and Bayesian Networks (BNs).

.. toctree::
   :hidden:

   Getting Started <started/base>
   Documentation <documentation>
   Examples <examples>
   API Reference <api>
   Citation <citation>
   Getting Involved <development>
