Metrics
=======

.. meta::
   :description: Evaluate learned graphs and models with supervised and unsupervised metrics in pgmpy.

Metrics help quantify how good a learned model is, either by comparing it to a
known ground-truth graph or by checking how well it matches the data.

For example, Structural Hamming Distance (SHD) counts the number of edge
additions, deletions, and reversals needed to transform an estimated graph into
the true graph, where :math:`R` denotes reversed edges:

.. math::

   SHD(G, \hat{G}) = |E \setminus \hat{E}| + |\hat{E} \setminus E| + |R|

Example
-------

.. code-block:: python

    from pgmpy.base import DAG
    from pgmpy.metrics import SHD

    true_graph = DAG([("A", "B"), ("B", "C")])
    est_graph = DAG([("B", "A"), ("B", "C")])

    shd = SHD()
    print(shd(true_causal_graph=true_graph, est_causal_graph=est_graph))

When to use which
-----------------

- **SHD** -- The standard metric for comparing a learned graph against the
  ground truth. Counts edge additions, deletions, and reversals needed.
- **Structure Score** -- Compares the score (BIC, BDeu, etc.) of the learned
  structure against the true structure.
- **Correlation Score** -- Compares observed correlations in data against those
  implied by the model. Does not require a ground-truth graph.
- **Implied CIs** -- Tests whether the conditional independencies implied by the
  model hold in the data.
- **Fisher C** -- Combines p-values from independence tests implied by the
  model into an overall goodness-of-fit statistic.

Algorithms
----------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Metric
     - Type
     - API Reference
   * - Structural Hamming Distance (SHD)
     - Supervised
     - :class:`pgmpy.metrics.SHD`
   * - Structure Score
     - Supervised
     - :class:`pgmpy.metrics.StructureScore`
   * - Correlation Score
     - Unsupervised
     - :class:`pgmpy.metrics.CorrelationScore`
   * - Implied Conditional Independencies
     - Unsupervised
     - :class:`pgmpy.metrics.ImpliedCIs`
   * - Fisher C
     - Unsupervised
     - :class:`pgmpy.metrics.FisherC`

See Also
--------

- **API Reference:** :doc:`Metrics API <../metrics/metrics>`
- **Previous:** :doc:`causal_estimation` -- estimate causal effects
- **Next:** :doc:`simulations` -- generate synthetic data from a model
