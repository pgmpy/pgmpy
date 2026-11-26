.. meta::
   :description: pgmpy example notebooks about Bayesian networks, structure learning, inference, simulations, and causal analysis.

Example Notebooks
=================

These `Jupyter notebooks <https://github.com/pgmpy/pgmpy/tree/dev/examples>`_ show the most
common tasks in pgmpy: model definition, learning from data, inference, and causal analysis.
Find a notebook on this page. For more details, go to the related guide or API section.

Defining Bayesian Networks
""""""""""""""""""""""""""

See the :doc:`Defining a Custom Model <guides/custom_model>` guide for background.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Creating Discrete BN
      :link: examples/Creating_Discrete_BN
      :link-type: doc
      :class-card: sd-card-hover

      Define a discrete Bayesian network.

   .. grid-item-card:: Creating Linear BN
      :link: examples/Creating_Linear_BN
      :link-type: doc
      :class-card: sd-card-hover

      Define a linear Gaussian Bayesian network.

   .. grid-item-card:: Dynamic BN
      :link: examples/Dynamic_BN
      :link-type: doc
      :class-card: sd-card-hover

      Model temporal dependencies with a Dynamic BN.

   .. grid-item-card:: Defining CPDs
      :link: examples/Defining_CPDs
      :link-type: doc
      :class-card: sd-card-hover

      Define conditional probability distributions.

   .. grid-item-card:: Basic Operations on BN
      :link: examples/Basic_Operations_on_BN
      :link-type: doc
      :class-card: sd-card-hover

      Examine, change, and validate a BN.

Causal Discovery / Structure Learning
"""""""""""""""""""""""""""""""""""""

See the :doc:`Causal Discovery <guides/causal_discovery>` guide for background.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Structure Learning
      :link: examples/Structure_Learning
      :link-type: doc
      :class-card: sd-card-hover

      Learn a graph structure from data.

   .. grid-item-card:: Chow-Liu Tree
      :link: examples/Structure_Learning_Chow_Liu
      :link-type: doc
      :class-card: sd-card-hover

      Learn networks with a tree structure.

   .. grid-item-card:: TAN
      :link: examples/Structure_Learning_TAN
      :link-type: doc
      :class-card: sd-card-hover

      Learn a tree-augmented Naive Bayes model.

   .. grid-item-card:: Expert Knowledge
      :link: examples/Expert_Knowledge
      :link-type: doc
      :class-card: sd-card-hover

      Add domain constraints to structure learning.

Parameter Estimation
""""""""""""""""""""

See the :doc:`Parameter Estimation <guides/parameter_estimation>` guide for background.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Discrete BN Parameters
      :link: examples/Parameter_Learning_Discrete_BN
      :link-type: doc
      :class-card: sd-card-hover

      Fit the CPDs of a discrete BN.

   .. grid-item-card:: Factor Graph Parameters
      :link: examples/Parameter_Learning_Factor_Graphs
      :link-type: doc
      :class-card: sd-card-hover

      Estimate the parameters of factor graphs.

Probabilistic Inference
"""""""""""""""""""""""

See the :doc:`Probabilistic Inference <guides/probabilistic_inference>` guide for background.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

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

See the :doc:`Causal Identification <guides/causal_identification>` and :doc:`Causal Estimation <guides/causal_estimation>` guides for background.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Causal Inference
      :link: examples/Causal_Inference
      :link-type: doc
      :class-card: sd-card-hover

      Estimate causal effects from data.

   .. grid-item-card:: Causal Games
      :link: examples/Causal_Games
      :link-type: doc
      :class-card: sd-card-hover

      Learn causal reasoning with games.

Simulations
"""""""""""

See the :doc:`Simulations <guides/simulations>` guide for background.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Simulating Data
      :link: examples/Simulating_Data
      :link-type: doc
      :class-card: sd-card-hover

      Generate synthetic samples from a BN.

Extending pgmpy
""""""""""""""""""

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: Extending pgmpy
      :link: examples/Extending_pgmpy
      :link-type: doc
      :class-card: sd-card-hover

      Add custom models, estimators, or utilities.

   .. grid-item-card:: Functional Bayesian Network
      :link: examples/Functional_Bayesian_Network_Tutorial
      :link-type: doc
      :class-card: sd-card-hover

      Make functional CPDs for hybrid models.

   .. grid-item-card:: Junction Tree Inference
      :link: examples/Junction_Tree_Inference
      :link-type: doc
      :class-card: sd-card-hover

      Do inference with junction trees.

Tutorial Notebooks
""""""""""""""""""

These tutorial notebooks explain the core concepts of pgmpy step by step. They start with the
basics of probabilistic graphical models and end with real-world applications.

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: pgmpy-card-grid

   .. grid-item-card:: 1. Introduction to PGMs
      :link: detailed_notebooks/1.\ Introduction\ to\ Probabilistic\ Graphical\ Models
      :link-type: doc
      :class-card: sd-card-hover

      The basics of probabilistic graphical models.

   .. grid-item-card:: 2. Bayesian Networks
      :link: detailed_notebooks/2.\ Bayesian\ Networks
      :link-type: doc
      :class-card: sd-card-hover

      Make Bayesian networks and use them for reasoning.

   .. grid-item-card:: 3. Causal Bayesian Networks
      :link: detailed_notebooks/3.\ Causal\ Bayesian\ Networks
      :link-type: doc
      :class-card: sd-card-hover

      Causal reasoning with Bayesian networks.

   .. grid-item-card:: 4. Markov Models
      :link: detailed_notebooks/4.\ Markov\ Models
      :link-type: doc
      :class-card: sd-card-hover

      Undirected graphical models and Markov networks.

   .. grid-item-card:: 5. Exact Inference
      :link: detailed_notebooks/5.\ Exact\ Inference\ in\ Graphical\ Models
      :link-type: doc
      :class-card: sd-card-hover

      Variable Elimination, Belief Propagation, and other methods.

   .. grid-item-card:: 6. Approximate Inference
      :link: detailed_notebooks/6.\ Approximate\ Inference\ in\ Graphical\ Models
      :link-type: doc
      :class-card: sd-card-hover

      Sampling-based and variational methods.

   .. grid-item-card:: 7. Continuous Variables
      :link: detailed_notebooks/7.\ Parameterizing\ with\ Continuous\ Variables
      :link-type: doc
      :class-card: sd-card-hover

      Linear Gaussian and hybrid models.

   .. grid-item-card:: 8. Sampling Algorithms
      :link: detailed_notebooks/8.\ Sampling\ Algorithms
      :link-type: doc
      :class-card: sd-card-hover

      Forward, rejection, and likelihood-weighted sampling.

   .. grid-item-card:: 9. Reading & Writing Models
      :link: detailed_notebooks/9.\ Reading\ and\ Writing\ from\ pgmpy\ file\ formats
      :link-type: doc
      :class-card: sd-card-hover

      Read and write models in different file formats.

   .. grid-item-card:: 10. Learning BNs from Data
      :link: detailed_notebooks/10.\ Learning\ Bayesian\ Networks\ from\ Data
      :link-type: doc
      :class-card: sd-card-hover

      The full process of structure and parameter learning.

   .. grid-item-card:: 11. Energy & Greenhouse Gases
      :link: detailed_notebooks/11.\ A\ Bayesian\ Network\ to\ model\ the\ influence\ of\ energy\ consumption\ on\ greenhouse\ gases\ in\ Italy
      :link-type: doc
      :class-card: sd-card-hover

      A real-world case study with Italian energy data.

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
