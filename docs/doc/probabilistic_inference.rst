Probabilistic Inference
=======================

Probabilistic inference is the task of computing posterior probability
distributions over variables of interest given observed evidence. For example,
given a medical diagnosis model and observed symptoms, inference computes the
probability distribution over possible diseases.

pgmpy provides both exact and approximate inference algorithms.

Exact Inference
---------------

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
---------------------

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
