:hide-toc:

.. meta::
   :description: Task-oriented pgmpy guides for causal discovery, parameter estimation, probabilistic inference, simulations, and model building.

User Guide
==========

Use these guides when you want workflow-oriented documentation before diving into
the API reference. Each page focuses on a concrete task and links to the
relevant examples and public APIs.

Core Workflow
-------------

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Causal Discovery and Structure Learning
      :link: guides/causal_discovery
      :link-type: doc
      :class-card: sd-card-hover

      Learn causal graphs from data with constraint-based, score-based, and expert-guided algorithms.

   .. grid-item-card:: Parameter Estimation
      :link: guides/parameter_estimation
      :link-type: doc
      :class-card: sd-card-hover

      Learn CPDs from data using MLE, Bayesian priors, or EM for missing data.

   .. grid-item-card:: Probabilistic Inference
      :link: guides/probabilistic_inference
      :link-type: doc
      :class-card: sd-card-hover

      Compute posteriors, marginals, and MAP assignments with exact or approximate methods.

   .. grid-item-card:: Simulations
      :link: guides/simulations
      :link-type: doc
      :class-card: sd-card-hover

      Sample observational, interventional, and conditional data from fitted models.

Causal Inference
----------------

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Causal Identification
      :link: guides/causal_identification
      :link-type: doc
      :class-card: sd-card-hover

      Determine identifiability using backdoor adjustment and frontdoor criteria.

   .. grid-item-card:: Causal Estimation
      :link: guides/causal_estimation
      :link-type: doc
      :class-card: sd-card-hover

      Estimate treatment effects and interventional distributions from causal graphs and data.

Evaluation & Data
-----------------

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Metrics
      :link: guides/metrics
      :link-type: doc
      :class-card: sd-card-hover

      Evaluate learned graphs with supervised and unsupervised metrics.

   .. grid-item-card:: Example Datasets
      :link: guides/datasets
      :link-type: doc
      :class-card: sd-card-hover

      Curated benchmark datasets with ground-truth graphs and expert knowledge.

   .. grid-item-card:: Example Models
      :link: guides/example_models
      :link-type: doc
      :class-card: sd-card-hover

      Ready-made networks from bnlearn, bnrep, and dagitty for benchmarking and exploration.

Utilities
---------

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Exporting / Importing Models
      :link: guides/io
      :link-type: doc
      :class-card: sd-card-hover

      Save and load models in BIF, NET, XMLBIF, XDSL, and other formats.

   .. grid-item-card:: Defining a Custom Model
      :link: guides/custom_model
      :link-type: doc
      :class-card: sd-card-hover

      Define graphs and CPDs directly for discrete, continuous, or dynamic models.

   .. grid-item-card:: Plotting Models
      :link: guides/plotting
      :link-type: doc
      :class-card: sd-card-hover

      Visualize model structure with Graphviz, daft, and networkx backends.

Contributing
------------

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Extensibility
      :link: guides/extensibility
      :link-type: doc
      :class-card: sd-card-hover

      Add new algorithms, datasets, and metrics using the built-in extension templates.

.. toctree::
   :hidden:

   guides/causal_discovery
   guides/parameter_estimation
   guides/probabilistic_inference
   guides/causal_identification
   guides/causal_estimation
   guides/metrics
   guides/datasets
   guides/example_models
   guides/simulations
   guides/io
   guides/custom_model
   guides/plotting
   guides/extensibility
