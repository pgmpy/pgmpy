Examples
========
A curated set of `Jupyter notebooks <https://github.com/pgmpy/pgmpy/tree/dev/examples>`_ that demonstrate the most common tasks in pgmpy - building models, learning from data, inference, and causal analysis.

Defining Bayesian Networks
""""""""""""""""""""""

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
