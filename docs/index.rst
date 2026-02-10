.. pgmpy documentation master file

:hide-toc:
:hide-navigation:

Welcome to pgmpy
================

*Python library for Causal AI*

pgmpy is a Python package for causal inference and probabilistic inference
using Directed Acyclic Graphs (DAGs) and Bayesian Networks with a focus on
modularity and extensibility. Implementations of various algorithms for Causal
Discovery (a.k.a, Structure Learning), Parameter Estimation, Approximate
(Sampling Based) and Exact inference, and Causal Inference are available.

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
      :link: causal_infer/base
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
   Development <development>
