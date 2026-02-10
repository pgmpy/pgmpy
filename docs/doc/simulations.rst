Simulations
===========

.. meta::
   :description: Generate synthetic data by sampling from Bayesian Networks.

Simulation generates synthetic data by sampling from a Bayesian Network.

Given a DAG with CPDs, the joint distribution factorizes as
:math:`P(X_1, \ldots, X_n) = \prod_i P(X_i \mid Pa_i)`. Forward sampling draws
variables in a topological order to produce i.i.d. samples from this joint
distribution.

Example
-------

.. code-block:: python

    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.utils import get_example_model

    model = get_example_model("asia")
    sampler = BayesianModelSampling(model)
    data = sampler.forward_sample(size=1000, show_progress=False)
    print(data.head())

Algorithms
----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Algorithm
     - API Reference
   * - Forward Sampling
     - :class:`pgmpy.sampling.Sampling.BayesianModelSampling`
   * - Rejection Sampling
     - :class:`pgmpy.sampling.Sampling.BayesianModelSampling`
   * - Likelihood-Weighted Sampling
     - :class:`pgmpy.sampling.Sampling.BayesianModelSampling`
   * - Gibbs Sampling
     - :class:`pgmpy.sampling.Sampling.GibbsSampling`

See Also
--------

- **Examples:** :doc:`Simulating Data <../examples/Simulating_Data>`
- **Previous:** :doc:`metrics` -- evaluate model quality
- **Next:** :doc:`datasets` -- built-in datasets for testing
