:hide-toc:

.. meta::
   :description: pgmpy guides for causal discovery, parameter estimation, probabilistic inference, simulations, and custom models.

User Guide
==========

Each guide shows how to do one task with pgmpy. Read the guides before the API reference.
Each guide also links to the related examples and API pages.

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

      Learn CPDs from data with MLE, Bayesian priors, or EM for missing data.

   .. grid-item-card:: Probabilistic Inference
      :link: guides/probabilistic_inference
      :link-type: doc
      :class-card: sd-card-hover

      Calculate posteriors, marginals, and MAP states with exact or approximate methods.

   .. grid-item-card:: Simulations
      :link: guides/simulations
      :link-type: doc
      :class-card: sd-card-hover

      Draw observational, interventional, and conditional samples from fitted models.

Causal Inference
----------------

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Causal Identification
      :link: guides/causal_identification
      :link-type: doc
      :class-card: sd-card-hover

      Check identifiability with the backdoor and frontdoor criteria.

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

      Benchmark datasets with ground-truth graphs and expert knowledge.

   .. grid-item-card:: Example Models
      :link: guides/example_models
      :link-type: doc
      :class-card: sd-card-hover

      Networks from bnlearn, bnRep, and DAGitty for benchmarks and examples.

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

      Define the graphs and CPDs of discrete, continuous, or dynamic models yourself.

   .. grid-item-card:: Plotting Models
      :link: guides/plotting
      :link-type: doc
      :class-card: sd-card-hover

      Draw the model structure with Graphviz, daft, or networkx.

Contributing
------------

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Extensibility
      :link: guides/extensibility
      :link-type: doc
      :class-card: sd-card-hover

      Use the extension templates to add new algorithms, datasets, and metrics.

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
