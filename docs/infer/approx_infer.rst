Approximate Inference Using Sampling
====================================

.. autoclass:: pgmpy.inference.ApproxInference.ApproxInference
   :members:

.. note::

    The `query` method in `ApproxInference` currently does **not** support datasets generated using **Weighted Likelihood Sampling (WLS)**. 
    Any sample weights will be ignored during inference.
    Ensure that you fit the model and use evidence-based queries without passing weighted datasets.
