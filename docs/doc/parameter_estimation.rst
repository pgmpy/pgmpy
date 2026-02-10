Parameter Estimation
====================

.. meta::
   :description: Estimate CPDs for known structures using MLE, Bayesian, or EM methods.

Once a model structure is known, parameter estimation learns the numerical
parameters associated with that structure. In a Bayesian network, this means
estimating one conditional probability distribution (CPD) for each variable,
conditioned on every parent configuration in the graph. After this step, the
graph is no longer just a dependency structure: together with the learned CPDs,
it defines a full probabilistic model that can be used for inference, sampling,
and prediction.

For fully observed discrete data, maximum likelihood estimation (MLE) computes
each CPD entry from empirical counts. For a variable :math:`X`, one of its
states :math:`x`, and a parent configuration :math:`Pa = p`,
:math:`N(x, p)` is the number of rows where both conditions hold and
:math:`N(p) = \sum_{x'} N(x', p)` is the number of rows with parent state
:math:`p`. The MLE estimate is:

.. math::

   \hat{P}(X = x \mid Pa = p) = \frac{N(x, p)}{N(p)}

Example
-------

.. code-block:: python

    import pandas as pd
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork

    data = pd.DataFrame(
        {
            "PKA": [0, 0, 1, 1],
            "ERK": [0, 0, 1, 1],
            "Akt": [0, 1, 1, 0],
        }
    )
    model = DiscreteBayesianNetwork([("PKA", "ERK"), ("ERK", "Akt")])

    mle = MaximumLikelihoodEstimator(model, data)
    cpds = mle.get_parameters()
    model.add_cpds(*cpds)

    print("Learned CPDs:")
    for cpd in model.get_cpds():
        print(cpd)

This learns one CPD for each node in the model:

- :math:`P(\text{PKA})`
- :math:`P(\text{ERK} \mid \text{PKA})`
- :math:`P(\text{Akt} \mid \text{ERK})`

Example output:

.. code-block:: text

    +--------+-----+
    | PKA(0) | 0.5 |
    +--------+-----+
    | PKA(1) | 0.5 |
    +--------+-----+

    +--------+--------+--------+
    | PKA    | PKA(0) | PKA(1) |
    +--------+--------+--------+
    | ERK(0) | 1.0    | 0.0    |
    +--------+--------+--------+
    | ERK(1) | 0.0    | 1.0    |
    +--------+--------+--------+

    +--------+--------+--------+
    | ERK    | ERK(0) | ERK(1) |
    +--------+--------+--------+
    | Akt(0) | 0.5    | 0.5    |
    +--------+--------+--------+
    | Akt(1) | 0.5    | 0.5    |
    +--------+--------+--------+

Algorithms
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - API Reference
   * - Maximum Likelihood Estimation (MLE)
     - :class:`pgmpy.estimators.MLE.MaximumLikelihoodEstimator`
   * - Bayesian Estimation
     - :class:`pgmpy.estimators.BayesianEstimator.BayesianEstimator`
   * - Expectation Maximization (EM)
     - :class:`pgmpy.estimators.EM.ExpectationMaximization`
   * - SEM Estimator
     - :class:`pgmpy.estimators.SEMEstimator.SEMEstimator`
   * - IV Estimator
     - :class:`pgmpy.estimators.SEMEstimator.IVEstimator`

See Also
--------

- **Examples:** :doc:`Discrete BN Parameters <../examples/Parameter_Learning_Discrete_BN>` | :doc:`Factor Graph Parameters <../examples/Parameter_Learning_Factor_Graphs>`
- **API Reference:** :doc:`Parameter Estimation API <../param_estimator/base>`
- **Previous:** :doc:`causal_discovery` -- learn graph structure from data
- **Next:** :doc:`probabilistic_inference` -- query the fitted model
