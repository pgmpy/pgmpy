Examples
========
A curated set of `Jupyter notebooks <https://github.com/pgmpy/pgmpy/tree/dev/examples>`_ that demonstrate the most common tasks in pgmpy - building models, learning from data, inference, and causal analysis.

Defining Bayesian Networks
""""""""""""""""""""""

See the :doc:`Defining a Custom Model <doc/custom_model>` guide for background.

.. grid:: 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards

   .. grid-item-card:: Creating Discrete BN
      :link: examples/Creating_Discrete_BN
      :link-type: doc
      :class-card: sd-card-hover

      Build a discrete Bayesian Network from scratch.

   .. grid-item-card:: Creating Linear BN
      :link: examples/Creating_Linear_BN
      :link-type: doc
      :class-card: sd-card-hover

      Define a linear Gaussian Bayesian Network.

   .. grid-item-card:: Dynamic BN
      :link: examples/Dynamic_BN
      :link-type: doc
      :class-card: sd-card-hover

      Model temporal dependencies with a Dynamic BN.

   .. grid-item-card:: Defining CPDs
      :link: examples/Defining_CPDs
      :link-type: doc
      :class-card: sd-card-hover

      Specify conditional probability distributions.

   .. grid-item-card:: Basic Operations on BN
      :link: examples/Basic_Operations_on_BN
      :link-type: doc
      :class-card: sd-card-hover

      Inspect, modify, and validate a BN.

Causal Discovery / Structure Learning
""""""""""""""""""""""""""""""

See the :doc:`Causal Discovery <doc/causal_discovery>` guide for background.

.. grid:: 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards

   .. grid-item-card:: Structure Learning
      :link: examples/Structure_Learning
      :link-type: doc
      :class-card: sd-card-hover

      Learn a graph structure from data.

   .. grid-item-card:: Chow-Liu Tree
      :link: examples/Structure_Learning_Chow_Liu
      :link-type: doc
      :class-card: sd-card-hover

      Learn tree-structured networks efficiently.

   .. grid-item-card:: TAN
      :link: examples/Structure_Learning_TAN
      :link-type: doc
      :class-card: sd-card-hover

      Learn a tree-augmented Naive Bayes model.

   .. grid-item-card:: Expert Knowledge
      :link: examples/Expert_Knowledge
      :link-type: doc
      :class-card: sd-card-hover

      Incorporate domain constraints into learning.

Parameter Estimation
""""""""""""""""""

See the :doc:`Parameter Estimation <doc/parameter_estimation>` guide for background.

.. grid:: 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards

   .. grid-item-card:: Discrete BN Parameters
      :link: examples/Parameter_Learning_Discrete_BN
      :link-type: doc
      :class-card: sd-card-hover

      Fit CPDs for a discrete BN.

   .. grid-item-card:: Factor Graph Parameters
      :link: examples/Parameter_Learning_Factor_Graphs
      :link-type: doc
      :class-card: sd-card-hover

      Estimate parameters for factor graphs.

Probabilistic Inference
"""""""""""""""""""""

See the :doc:`Probabilistic Inference <doc/probabilistic_inference>` guide for background.

.. grid:: 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards

   .. grid-item-card:: Inference in Discrete BN
      :link: examples/Inference_Discrete_BN
      :link-type: doc
      :class-card: sd-card-hover

      Query posterior probabilities with evidence.

   .. grid-item-card:: Monty Hall
      :link: examples/Monty_Hall
      :link-type: doc
      :class-card: sd-card-hover

      Solve the Monty Hall problem with a BN.

Causal Inference
""""""""""""""""""

See the :doc:`Causal Identification <doc/causal_identification>` and :doc:`Causal Estimation <doc/causal_estimation>` guides for background.

.. grid:: 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards

   .. grid-item-card:: Causal Inference
      :link: examples/Causal_Inference
      :link-type: doc
      :class-card: sd-card-hover

      Estimate causal effects from data.

   .. grid-item-card:: Causal Games
      :link: examples/Causal_Games
      :link-type: doc
      :class-card: sd-card-hover

      Explore causal reasoning via games.

Simulations
""""""""""

See the :doc:`Simulations <doc/simulations>` guide for background.

.. grid:: 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards

   .. grid-item-card:: Simulating Data
      :link: examples/Simulating_Data
      :link-type: doc
      :class-card: sd-card-hover

      Generate synthetic samples from a BN.

Extending pgmpy
""""""""""""""""""

.. grid:: 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards

   .. grid-item-card:: Extending pgmpy
      :link: examples/Extending_pgmpy
      :link-type: doc
      :class-card: sd-card-hover

      Add custom models, estimators, or utilities.

   .. grid-item-card:: Functional Bayesian Network
      :link: examples/Functional_Bayesian_Network_Tutorial
      :link-type: doc
      :class-card: sd-card-hover

      Build functional CPDs for hybrid models.

   .. grid-item-card:: Junction Tree Inference
      :link: examples/Junction_Tree_Inference
      :link-type: doc
      :class-card: sd-card-hover

      Perform inference using junction trees.

Tutorial Notebooks
""""""""""""""""""

A series of in-depth tutorial notebooks that walk through pgmpy's core concepts step by step — from
probabilistic graphical model basics to real-world applications.

.. grid:: 3
   :gutter: 3
   :class-container: sd-shadow-hover-cards

   .. grid-item-card:: 1. Introduction to PGMs
      :link: detailed_notebooks/1. Introduction to Probabilistic Graphical Models
      :link-type: doc
      :class-card: sd-card-hover

      Foundations of probabilistic graphical models.

   .. grid-item-card:: 2. Bayesian Networks
      :link: detailed_notebooks/2. Bayesian Networks
      :link-type: doc
      :class-card: sd-card-hover

      Build and reason with Bayesian Networks.

   .. grid-item-card:: 3. Causal Bayesian Networks
      :link: detailed_notebooks/3. Causal Bayesian Networks
      :link-type: doc
      :class-card: sd-card-hover

      Causal reasoning with Bayesian Networks.

   .. grid-item-card:: 4. Markov Models
      :link: detailed_notebooks/4. Markov Models
      :link-type: doc
      :class-card: sd-card-hover

      Undirected graphical models and Markov Networks.

   .. grid-item-card:: 5. Exact Inference
      :link: detailed_notebooks/5. Exact Inference in Graphical Models
      :link-type: doc
      :class-card: sd-card-hover

      Variable Elimination, Belief Propagation, and more.

   .. grid-item-card:: 6. Approximate Inference
      :link: detailed_notebooks/6. Approximate Inference in Graphical Models
      :link-type: doc
      :class-card: sd-card-hover

      Sampling-based and variational methods.

   .. grid-item-card:: 7. Continuous Variables
      :link: detailed_notebooks/7. Parameterizing with Continuous Variables
      :link-type: doc
      :class-card: sd-card-hover

      Linear Gaussian and hybrid models.

   .. grid-item-card:: 8. Sampling Algorithms
      :link: detailed_notebooks/8. Sampling Algorithms
      :link-type: doc
      :class-card: sd-card-hover

      Forward, rejection, and likelihood-weighted sampling.

   .. grid-item-card:: 9. Reading & Writing Models
      :link: detailed_notebooks/9. Reading and Writing from pgmpy file formats
      :link-type: doc
      :class-card: sd-card-hover

      Import and export models in various formats.

   .. grid-item-card:: 10. Learning BNs from Data
      :link: detailed_notebooks/10. Learning Bayesian Networks from Data
      :link-type: doc
      :class-card: sd-card-hover

      Structure and parameter learning end-to-end.

   .. grid-item-card:: 11. Energy & Greenhouse Gases
      :link: detailed_notebooks/11. A Bayesian Network to model the influence of energy consumption on greenhouse gases in Italy
      :link-type: doc
      :class-card: sd-card-hover

      Real-world case study with Italian energy data.

.. toctree::
   :hidden:

   examples/Creating_Discrete_BN
   examples/Creating_Linear_BN
   examples/Dynamic_BN
   examples/Defining_CPDs
   examples/Basic_Operations_on_BN
   examples/Structure_Learning
   examples/Structure_Learning_Chow_Liu
   examples/Structure_Learning_TAN
   examples/Expert_Knowledge
   examples/Parameter_Learning_Discrete_BN
   examples/Parameter_Learning_Factor_Graphs
   examples/Inference_Discrete_BN
   examples/Monty_Hall
   examples/Causal_Inference
   examples/Causal_Games
   examples/Simulating_Data
   examples/Extending_pgmpy
   examples/Functional_Bayesian_Network_Tutorial
   examples/Junction_Tree_Inference
   tutorial
