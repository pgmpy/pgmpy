:hide-toc:

.. meta::
   :description: Task-oriented pgmpy guides for causal discovery, parameter estimation, probabilistic inference, simulations, and model building.

Guides
======

Use these guides when you want workflow-oriented documentation before diving into
the API reference. Each page focuses on a concrete task and links to the
relevant examples and public APIs.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Causal Discovery and Structure Learning
      :link: guides/causal_discovery
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Learn causal graph structure from data.

   .. grid-item-card:: Parameter Estimation
      :link: guides/parameter_estimation
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Estimate model parameters from data.

   .. grid-item-card:: Probabilistic Inference
      :link: guides/probabilistic_inference
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Query posterior probabilities given evidence.

   .. grid-item-card:: Causal Identification
      :link: guides/causal_identification
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Determine if a causal effect is identifiable from the graph.

   .. grid-item-card:: Causal Estimation
      :link: guides/causal_estimation
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Estimate causal effects from data.

   .. grid-item-card:: Metrics
      :link: guides/metrics
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Evaluate and compare learned models.

   .. grid-item-card:: Example Datasets
      :link: guides/datasets
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Built-in datasets for testing and experimentation.

   .. grid-item-card:: Example Models
      :link: guides/example_models
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Pre-built Bayesian Networks from standard repositories.

   .. grid-item-card:: Simulations
      :link: guides/simulations
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Generate synthetic data from Bayesian Networks.

   .. grid-item-card:: Exporting / Importing Models
      :link: guides/io
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Read and write models in various file formats.

   .. grid-item-card:: Defining a Custom Model
      :link: guides/custom_model
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Build models from scratch with custom structure and parameters.

   .. grid-item-card:: Plotting Models
      :link: guides/plotting
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Visualize graphs with pygraphviz, daft, and networkx.

   .. grid-item-card:: Extensibility
      :link: guides/extensibility
      :link-type: doc
      :class-card: sd-card-hover pgmpy-card

      Use repository templates to add datasets, models, metrics, and algorithms.

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
