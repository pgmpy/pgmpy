.. pgmpy documentation master file

:hide-toc:
:hide-navigation:

.. meta::
   :description: pgmpy documentation for causal discovery, model testing, causal effect estimation, parameter estimation, probabilistic and causal inference, and simulations in Python.

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: hero-grid

   .. grid-item::
      :class: hero-logo-panel

      .. image:: _static/images/logo.png
         :alt: pgmpy logo
         :width: 220px
         :align: center

   .. grid-item::
      :class: hero-copy-panel

      .. container:: hero-subtitle

         Python toolkit for causal and probabilistic reasoning

      pgmpy provides the building blocks for causal and probabilistic reasoning using graphical models. It implements data structures for a range of causal and graphical models such as DAGs, PDAGs, MAGs, PAGs, Bayesian Networks, Dynamic Bayesian Networks, and Structural Equation Models along with algorithms for various tasks such as causal discovery, causal identification, causal and probabilistic inference, model validation, parameter estimation, simulations, and more.

      Algorithms for each task follow a unified composable API, making them modular and extensible. They are also scikit-learn compatible when possible. They can be used directly, combined in sklearn pipelines, or used to build higher-level tools on top of them.


      .. container:: hero-actions

         .. button-ref:: started/index
            :ref-type: doc
            :color: primary
            :outline:
            :class: hero-action-button

            Getting Started

         .. button-ref:: documentation
            :ref-type: doc
            :color: primary
            :outline:
            :class: hero-action-button

            User Guide

         .. button-ref:: examples
            :ref-type: doc
            :color: primary
            :outline:
            :class: hero-action-button

            Example Notebooks

         .. button-ref:: reference
            :ref-type: doc
            :color: primary
            :outline:
            :class: hero-action-button

            API Reference

Key Features
------------

.. grid:: 1 1 2 4
   :gutter: 3
   :class-container: sd-shadow-hover-cards pgmpy-card-grid

   .. grid-item-card:: Causal Discovery / Structure Learning
      :link: quickstart-causal-discovery
      :link-type: ref
      :class-card: sd-card-hover pgmpy-card

      Learn causal graphs from data using scikit-learn compatible implementations.

   .. grid-item-card:: Parameter Estimation
      :link: quickstart-parameter-estimation
      :link-type: ref
      :class-card: sd-card-hover pgmpy-card

      Estimate conditional distributions for nodes in the model.

   .. grid-item-card:: Probabilistic Inference
      :link: quickstart-probabilistic-inference
      :link-type: ref
      :class-card: sd-card-hover pgmpy-card

      Compute posterior distributions from the learned model using exact or approximate inference.

   .. grid-item-card:: Causal Identification
      :link: quickstart-causal-identification
      :link-type: ref
      :class-card: sd-card-hover pgmpy-card

      Given a causal graph determine how to estimate the a causal query.

   .. grid-item-card:: Causal Inference
      :link: quickstart-causal-inference
      :link-type: ref
      :class-card: sd-card-hover pgmpy-card

      Compute interventional and counterfactual distributions from models.

   .. grid-item-card:: Example Datasets and Models
      :link: quickstart-example-data-models
      :link-type: ref
      :class-card: sd-card-hover pgmpy-card

      Built-in collection of example Bayesian Networks and datasets from different sources.

   .. grid-item-card:: Simulations
      :link: quickstart-simulations
      :link-type: ref
      :class-card: sd-card-hover pgmpy-card

      Simulate data from models under various scenarios.

   .. grid-item-card:: Extend pgmpy
      :link: quickstart-extensibility
      :link-type: ref
      :class-card: sd-card-hover pgmpy-card

      Write your own custom pgmpy plugable methods using our extension templates.


.. toctree::
   :hidden:

   Getting Started <started/index>
   User Guide <documentation>
   Examples <examples>
   API Reference <reference>
   Citation <citation>
   Getting Involved <development>
