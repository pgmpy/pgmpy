.. pgmpy documentation master file

.. title:: Documentation — pgmpy

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

      pgmpy is a Python library for causal and probabilistic reasoning with graphical models. It covers the full workflow from learning causal graphs from data to estimating causal effects, running probabilistic inference, and simulating data from fitted models. All algorithms follow a unified, composable API and are scikit-learn compatible where possible, so they work standalone, in sklearn pipelines, or as building blocks for higher-level tools.


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
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Causal Discovery / Structure Learning
      :link: quickstart-causal-discovery
      :link-type: ref
      :class-card: sd-card-hover

      Learn causal graphs from data with algorithms that are compatible with scikit-learn.

   .. grid-item-card:: Parameter Estimation
      :link: quickstart-parameter-estimation
      :link-type: ref
      :class-card: sd-card-hover

      Estimate the conditional distribution of each node in the model.

   .. grid-item-card:: Probabilistic Inference
      :link: quickstart-probabilistic-inference
      :link-type: ref
      :class-card: sd-card-hover

      Calculate posterior distributions from a model with exact or approximate inference.

   .. grid-item-card:: Causal Identification
      :link: quickstart-causal-identification
      :link-type: ref
      :class-card: sd-card-hover

      Use a causal graph to find how to estimate a causal effect.

   .. grid-item-card:: Causal Inference
      :link: quickstart-causal-inference
      :link-type: ref
      :class-card: sd-card-hover

      Calculate interventional and counterfactual distributions from models.

   .. grid-item-card:: Example Datasets and Models
      :link: quickstart-example-data-models
      :link-type: ref
      :class-card: sd-card-hover

      Use example Bayesian networks and datasets from different sources.

   .. grid-item-card:: Simulations
      :link: quickstart-simulations
      :link-type: ref
      :class-card: sd-card-hover

      Simulate data from models, with interventions, evidence, and missing values.

   .. grid-item-card:: Extend pgmpy
      :link: quickstart-extensibility
      :link-type: ref
      :class-card: sd-card-hover

      Use the extension templates to add your own methods to pgmpy.


.. toctree::
   :hidden:

   Getting Started <started/index>
   User Guide <documentation>
   Examples <examples>
   API Reference <reference>
   Citation <citation>
   Supported By <supported_by>
   Getting Involved <development>
   References <references>
