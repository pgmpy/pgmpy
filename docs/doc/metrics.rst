Metrics
=======

pgmpy provides metrics to evaluate learned models. Metrics fall into two
categories:

- **Supervised** metrics require a ground-truth graph for comparison (useful
  when the true structure is known, e.g., in simulation studies).
- **Unsupervised** metrics evaluate a model using only the data (useful for
  real-world applications where the true graph is unknown).

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

When to use which
-----------------

- **SHD** -- The standard metric for comparing a learned graph against the
  ground truth. Counts edge additions, deletions, and reversals needed.
- **Structure Score** -- Compares the score (BIC, BDeu, etc.) of the learned
  structure against the true structure.
- **Correlation Score** -- Compares observed correlations in data against those
  implied by the model. Does not require a ground-truth graph.
- **Implied CIs** -- Tests whether the conditional independencies implied by
  the model hold in the data.
- **Fisher C** -- Combines p-values from independence tests implied by the
  model into an overall goodness-of-fit statistic.
