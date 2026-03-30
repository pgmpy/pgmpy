.. meta::
   :description: Public API reference for pgmpy graphs, models, factors, inference, discovery, metrics, datasets, example models, and model import or export.

API Reference
=============

Public API reference for pgmpy.

Browse by workflow or object family to find the public classes and helper
functions for that part of the library.

Deprecated compatibility aliases are intentionally omitted from the main API
listings in favor of the current class names.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Graph Classes
      :link: api/base
      :link-type: doc
      :class-card: sd-card-hover

      DAG, PDAG, MAG, PAG, and other base graph structures.

   .. grid-item-card:: Models
      :link: api/models
      :link-type: doc
      :class-card: sd-card-hover

      Bayesian networks, SEMs, Markov models, and derived graph structures.

   .. grid-item-card:: Factors And CPDs
      :link: api/factors
      :link-type: doc
      :class-card: sd-card-hover

      Discrete factors, CPDs, Gaussian CPDs, hybrid CPDs, and factor utilities.

   .. grid-item-card:: Inference And Sampling
      :link: api/inference
      :link-type: doc
      :class-card: sd-card-hover

      Exact inference, approximate inference, sampling, and elimination-order helpers.

   .. grid-item-card:: Causal Inference
      :link: api/causal_inference
      :link-type: doc
      :class-card: sd-card-hover

      Identification, interventional inference, and regression-based causal estimators.

   .. grid-item-card:: Parameter Estimation
      :link: api/parameter_estimation
      :link-type: doc
      :class-card: sd-card-hover

      Estimator base classes, MLE, Bayesian estimation, EM, SEM fitting, and mirror descent.

   .. grid-item-card:: Causal Discovery
      :link: api/structure_learning
      :link-type: doc
      :class-card: sd-card-hover

      Constraint-based, score-based, tree-based, and expert-guided discovery algorithms.

   .. grid-item-card:: Conditional Independence (CI) Tests
      :link: api/ci_test
      :link-type: doc
      :class-card: sd-card-hover

      Built-in CI test selectors and test implementations.

   .. grid-item-card:: Structure Scores
      :link: api/structure_score
      :link-type: doc
      :class-card: sd-card-hover

      Discrete, Gaussian, and conditional-Gaussian structure scoring classes.

   .. grid-item-card:: Metrics
      :link: api/metrics
      :link-type: doc
      :class-card: sd-card-hover

      Supervised and unsupervised metrics for evaluating learned graphs and fitted models.

   .. grid-item-card:: Reading/Writing
      :link: api/readwrite
      :link-type: doc
      :class-card: sd-card-hover

      Reader and writer classes for BIF, XMLBIF, XDSL, UAI, PomdpX, and related formats.

   .. grid-item-card:: Datasets And Examples
      :link: api/data
      :link-type: doc
      :class-card: sd-card-hover

      Built-in dataset loaders and example-model discovery helpers.

   .. grid-item-card:: Independencies
      :link: api/independencies
      :link-type: doc
      :class-card: sd-card-hover

      Independence assertions and collections used across discovery and testing workflows.

.. toctree::
   :hidden:

   api/base
   api/models
   api/undirected
   api/factors
   api/inference
   api/causal_inference
   api/parameter_estimation
   api/structure_learning
   api/ci_test
   api/structure_score
   api/metrics
   api/readwrite
   api/data
   api/independencies
