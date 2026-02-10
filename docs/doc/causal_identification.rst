Causal Identification
=====================

Causal identification determines whether a causal effect can be uniquely
computed from observational data given a causal graph. Before estimating a
causal effect, it is important to verify that the effect is identifiable --
that is, the causal quantity can be expressed in terms of the observed data
distribution.

pgmpy implements standard graphical criteria for identification:

- **Backdoor criterion** -- Identifies a set of variables to condition on that
  blocks all backdoor (confounding) paths between treatment and outcome.
- **Frontdoor criterion** -- Used when the backdoor criterion fails, by finding
  mediator variables that satisfy the frontdoor conditions.

Algorithms
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - API Reference
   * - Adjustment (Backdoor)
     - :class:`pgmpy.identification.adjustment.Adjustment`
   * - Frontdoor
     - :class:`pgmpy.identification.frontdoor.Frontdoor`
