Causal Identification
=====================

.. meta::
   :description: Check whether a causal effect is identifiable from a causal graph using backdoor and frontdoor criteria.

Causal identification checks whether a causal effect can be computed from
observational data given a causal graph.

Formally, if a valid adjustment set :math:`Z` exists, the effect is
identifiable and can be written as:

.. math::

   P(Y \mid do(X)) = \sum_Z P(Y \mid X, Z) P(Z)

Example
-------

.. code-block:: python

    from pgmpy.base import DAG
    from pgmpy.identification import Adjustment

    dag = DAG(
        [("X", "Y"), ("Z", "X"), ("Z", "Y")],
        roles={"exposures": "X", "outcomes": "Y"},
    )
    identified = Adjustment(variant="minimal").identify(dag)
    print(identified.get_role("adjustment"))

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

See Also
--------

- **Examples:** :doc:`Causal Inference <../examples/Causal_Inference>`
- **API Reference:** :doc:`Causal Inference API <../causal_infer/base>`
- **Previous:** :doc:`probabilistic_inference` -- query posterior probabilities
- **Next:** :doc:`causal_estimation` -- estimate causal effects from data
