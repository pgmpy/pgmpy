Probabilistic Inference
=======================

.. meta::
   :description: Compute posterior probabilities with exact or approximate inference algorithms.

Probabilistic inference computes the probability of variables of interest
given observed evidence.

Formally, inference computes a posterior such as
:math:`P(X \mid e) = \frac{P(X, e)}{P(e)}`, often by summing out hidden
variables or using sampling to approximate the result.

Example
-------

.. code-block:: python

    from pgmpy.inference import VariableElimination
    from pgmpy.utils import get_example_model

    model = get_example_model("alarm")
    infer = VariableElimination(model)
    variable = list(model.nodes())[0]
    query = infer.query(variables=[variable])
    print(query)

When to use which
-----------------

- **Variable Elimination** -- General-purpose exact inference. Good default for
  small to medium networks.
- **Belief Propagation** -- Efficient for tree-structured networks or when
  multiple queries share computation.
- **MPLP** -- MAP (most probable explanation) inference via linear programming
  relaxation.
- **Approximate / Gibbs** -- Use when the network is too large for exact
  inference or when approximate answers are acceptable.

Algorithms
----------

Exact Inference
^^^^^^^^^^^^^^^

Exact inference computes the true posterior distribution. These methods are
suitable for small to moderately sized networks.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Algorithm
     - API Reference
   * - Variable Elimination
     - :class:`pgmpy.inference.ExactInference.VariableElimination`
   * - Belief Propagation
     - :class:`pgmpy.inference.ExactInference.BeliefPropagation`
   * - Max-Product Linear Programming (MPLP)
     - :class:`pgmpy.inference.mplp.Mplp`
   * - Dynamic Bayesian Network Inference
     - :class:`pgmpy.inference.dbn_inference.DBNInference`

Approximate Inference
^^^^^^^^^^^^^^^^^^^^^

Approximate inference uses sampling to estimate posterior distributions. These
methods scale better to large networks where exact inference is intractable.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Algorithm
     - API Reference
   * - Approximate Inference (Sampling)
     - :class:`pgmpy.inference.ApproxInference.ApproxInference`
   * - Forward Sampling
     - :class:`pgmpy.sampling.Sampling.BayesianModelSampling`
   * - Gibbs Sampling
     - :class:`pgmpy.sampling.Sampling.GibbsSampling`

See Also
--------

- **Examples:** :doc:`Inference in Discrete BN <../examples/Inference_Discrete_BN>` | :doc:`Monty Hall <../examples/Monty_Hall>` | :doc:`Junction Tree Inference <../examples/Junction_Tree_Inference>`
- **API Reference:** :doc:`Inference API <../infer/base>`
- **Previous:** :doc:`parameter_estimation` -- estimate model parameters
- **Next:** :doc:`causal_identification` -- check whether a causal effect is identifiable
