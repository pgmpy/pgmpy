# This extension template provides instructions to add new metrics to pgmpy.
#
# Please follow the following steps:
# 1. Copy this file to `pgmpy/metrics` and rename the file as `your_metric_name.py` (e.g., `my_metric.py`).
#    Note: Do NOT start the filename with an underscore `_`, otherwise it won't be discovered.
# 2. Go through the file and address all the TODOs.
# 3. Add an import statement in the `pgmpy/metrics/__init__.py` file (e.g. `from .my_metric import MyMetric`).
# 4. Add the metric name to the `__all__` list in `pgmpy/metrics/__init__.py` file.
# 5. If you would like to contribute the metric to pgmpy, please add tests for the metric in
#   `pgmpy/tests/test_metrics/test_your_metric_name.py` file.


# TODO: Add any other necessary imports here.
# import pandas as pd
# import numpy as np

from pgmpy.base import DAG  # noqa: F401

# TODO: Choose the appropriate base class based on whether your metric requires ground truth:
# For metrics that compare against a ground truth graph, import _BaseSupervisedMetric
# For metrics that evaluate a graph against data without ground truth, import _BaseUnsupervisedMetric
from pgmpy.metrics import _BaseSupervisedMetric, _BaseUnsupervisedMetric

# TODO: Add any additional imports needed for your specific metric implementation.
# Common imports might include:
# from sklearn.metrics import f1_score, precision_score, recall_score
# from pgmpy.estimators.CITests import ci_registry
# from itertools import combinations


# TODO: Choose ONE of the following class definitions and remove the other.
# Option 1: For supervised metrics (requires ground truth)
class MyMetric(_BaseSupervisedMetric):
    """
    [One line description of the metric.]

    [Detailed description of the metric, including what it measures, how it works,
    and when to use it.]

    Parameters
    ----------
    param1: type, optional (default=value)
        Description of param1.

    param2: type, optional (default=value)
        Description of param2.

    Returns
    -------
    metric_value: float
        [Description of what the returned value represents and its interpretation.
        E.g., "Higher values indicate better performance" or "Lower values indicate better fit"]

    Examples
    --------
    >>> from pgmpy.metrics import MyMetric
    >>> from pgmpy.base import DAG
    >>> # TODO: Provide a complete working example
    >>> true_graph = DAG([("A", "B"), ("B", "C")])
    >>> est_graph = DAG([("A", "B"), ("A", "C")])
    >>> metric = MyMetric()
    >>> metric(true_causal_graph=true_graph, est_causal_graph=est_graph)
    # Expected output value

    References
    ----------
    .. [1] TODO: Add citation for the metric
    """

    # TODO: Fill in the tags for your metric. This is mandatory.
    _tags = {
        "name": "my_metric",  # Change to your metric name (lowercase, underscores allowed)
        "requires_true_graph": True,  # Set to True for supervised metrics
        "requires_data": False,  # Set to True if metric needs data in addition to graphs
        "lower_is_better": True,  # Set to False if higher values are better
        "is_symmetric": False,  # Set to True if metric(A, B) == metric(B, A)
        "supported_graph_types": (DAG,),  # Add supported graph types (DAG, PDAG, etc.)
    }

    # TODO: Add all parameters required for the metric in the init method.
    def __init__(self, param1=None, param2=None):
        # TODO: Assign all parameters as attributes to the instance.
        self.param1 = param1
        self.param2 = param2

    def _evaluate(self, true_causal_graph, est_causal_graph, **kwargs):
        """
        Internal method to compute the metric value.

        Parameters
        ----------
        true_causal_graph: Instance of supported graph type
            The ground truth causal graph.

        est_causal_graph: Instance of supported graph type
            The estimated causal graph.

        **kwargs: dict
            Additional keyword arguments passed from the evaluate method.

        Returns
        -------
        metric_value: float
            The computed metric value.
        """
        # TODO: Implement the metric computation logic here.
        # Common patterns include:
        # 1. Validate that graphs have the same nodes if required
        # 2. Convert graphs to appropriate representations (e.g., adjacency matrices)
        # 3. Compute the metric based on differences/similarities
        # 4. Return the computed value

        # Example validation (remove if not needed):
        if set(true_causal_graph.nodes()) != set(est_causal_graph.nodes()):
            raise ValueError("The graphs must have the same nodes.")

        # TODO: Replace this placeholder with actual implementation
        # Remove the following line once you implement the actual metric
        _ = kwargs  # Suppress unused parameter warning
        metric_value = 0.0
        return metric_value


# TODO: Choose ONE of the following class definitions and remove the other.
# Option 2: For unsupervised metrics (no ground truth required)
class MyUnsupervisedMetric(_BaseUnsupervisedMetric):
    """
    [One line description of the metric.]

    [Detailed description of the metric, including what it measures, how it works,
    and when to use it.]

    Parameters
    ----------
    param1: type, optional (default=value)
        Description of param1.

    param2: type, optional (default=value)
        Description of param2.

    Returns
    -------
    metric_value: float or pandas.DataFrame
        [Description of what the returned value represents and its interpretation.
        Can return a DataFrame if return_summary=True or similar parameter is used.]

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> from pgmpy.metrics import MyUnsupervisedMetric
    >>> # TODO: Provide a complete working example
    >>> model = load_model("bnlearn/alarm")
    >>> data = model.simulate(int(1e4))
    >>> metric = MyUnsupervisedMetric()
    >>> metric(X=data, causal_graph=model)
    # Expected output value

    References
    ----------
    .. [1] TODO: Add citation for the metric
    """

    # TODO: Fill in the tags for your metric. This is mandatory.
    _tags = {
        "name": "my_unsupervised_metric",  # Change to your metric name
        "requires_true_graph": False,  # Always False for unsupervised metrics
        "requires_data": True,  # Always True for unsupervised metrics
        "lower_is_better": False,  # Set based on your metric's interpretation
        "supported_graph_types": (DAG,),  # Add supported graph types
    }

    # TODO: Add all parameters required for the metric in the init method.
    def __init__(self, param1=None, param2=None):
        # TODO: Assign all parameters as attributes to the instance.
        self.param1 = param1
        self.param2 = param2

    def _evaluate(self, X, causal_graph, **kwargs):
        """
        Internal method to compute the metric value.

        Parameters
        ----------
        X: pandas.DataFrame
            The data used for evaluation.

        causal_graph: Instance of supported graph type
            The causal graph to be evaluated.

        **kwargs: dict
            Additional keyword arguments passed from the evaluate method.

        Returns
        -------
        metric_value: float or pandas.DataFrame
            The computed metric value.
        """
        # TODO: Implement the metric computation logic here.
        # Common patterns include:
        # 1. Validate parameter values and configurations
        # 2. Extract necessary information from the graph and data
        # 3. Perform statistical tests or computations
        # 4. Return the computed value or summary dataframe

        # TODO: Replace this placeholder with actual implementation
        # Remove the following lines once you implement the actual metric
        _ = X  # Suppress unused parameter warning
        _ = causal_graph  # Suppress unused parameter warning
        _ = kwargs  # Suppress unused parameter warning
        metric_value = 0.0
        return metric_value


# TODO: After implementing your metric, remove the unused class definition above
# and update the class name in docstring examples.
# TODO: Test your implementation thoroughly with different graph types and edge cases.
