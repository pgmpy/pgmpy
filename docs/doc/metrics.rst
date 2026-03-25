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

When to use
-----------

- Use SHD when you have a ground-truth graph and want a simple structural error
  count.
- Use structure score comparisons when you care about how well the learned
  graph explains the data under a scoring function.
- Use correlation score or implied conditional independencies when no
  ground-truth graph is available.
- Use Fisher C when you want a single omnibus fit diagnostic from implied
  independencies.

Example
-------

.. code-block:: python

    from pgmpy.base import DAG
    from pgmpy.metrics import SHD

    true_graph = DAG([("A", "B"), ("B", "C")])
    est_graph = DAG([("B", "A"), ("B", "C")])

    shd = SHD()
    print(shd(true_causal_graph=true_graph, est_causal_graph=est_graph))

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

- **API Reference:** :doc:`Metrics API <../api/metrics>`
- **Previous:** :doc:`causal_estimation` -- estimate causal effects
- **Next:** :doc:`simulations` -- generate synthetic data from a model
