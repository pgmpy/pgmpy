.. meta::
   :description: Public API reference for pgmpy models, factors, inference, causal discovery, metrics, and model import or export.

API Reference
=============

Complete API reference for all pgmpy modules.

Use these section landing pages when you already know the workflow you need and
want the public classes, functions, and modules for that area.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards pgmpy-card-grid

   .. grid-item-card:: Graph Classes
      :link: api/base
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      DAG, PDAG, MAG, PAG, and other base graph structures.

   .. grid-item-card:: Models
      :link: api/models
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Bayesian networks, SEMs, and related structures.

   .. grid-item-card:: Parameterization
      :link: api/factors
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      CPDs, factors, and factor utilities for discrete, Gaussian, and hybrid models.

   .. grid-item-card:: Probabilistic Inference
      :link: api/inference
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Exact inference, approximate inference, sampling, and inference utilities.

   .. grid-item-card:: Causal Inference
      :link: api/causal_inference
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Causal identification, interventional inference, and causal estimators.

   .. grid-item-card:: Parameter Estimation
      :link: api/parameter_estimation
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      MLE, Bayesian estimation, EM, and SEM estimation workflows.

   .. grid-item-card:: Causal Discovery
      :link: api/structure_learning
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Structure learning algorithms, CI tests, and graph scoring methods.

   .. grid-item-card:: Conditional Independence (CI) Tests
      :link: api/ci_test
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      List of available CI tests

   .. grid-item-card:: Structure Scores
      :link: api/structure_score
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      List of available structure scoring methods.

   .. grid-item-card:: Metrics
      :link: api/metrics
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Supervised and unsupervised metrics for evaluating learned graphs and models.

   .. grid-item-card:: Reading/Writing
      :link: api/readwrite
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Readers and writers for BIF, XMLBIF, XDSL, UAI, PomdpX, and related formats.

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
