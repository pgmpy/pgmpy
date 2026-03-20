from itertools import combinations

import pandas as pd
from sklearn.metrics import f1_score

from pgmpy.base import DAG
from pgmpy.ci_tests import get_ci_test
from pgmpy.metrics import _BaseUnsupervisedMetric


class CorrelationScore(_BaseUnsupervisedMetric):
    """
    Score to compute how well the model structure represents the correlations
    in the data. The model doesn't need to be parameterized for this score.

    A Bayesian Network or DAG has d-connection property which can be used to
    determine which variables are correlated according to the model. This
    function uses this d-connection/d-separation property to compare the model
    with variable correlations in a given dataset. For every pair of variables
    in the dataset, a correlation test (specified by `test` argument) is done.
    We say that any two variables are correlated if the test's p-value <
    significance_level. The same pair of variables are then tested whether they
    are d-connected in the network structure or not. Finally, a metric specified
    by `score` is computed by using the correlation test as the true value and
    d-connections as predicted values.

    Absence of correlation/d-separation is considered as the positive class for
    computing the metrics.

    Parameters
    ----------
    ci_test: str or function
        The statistical tests to use for determining whether the variables in data
        are correlated or not. For discrete variables, the options are: 1) chi_square
        2) g_sq 3) log_likelihood 4) freeman_tuckey 5) modified_log_likelihood 6) neyman
        7) cressie_read. For continuous variables only one test is available: 1) pearsonr.
        A function with the signature fun(X, Y, Z, data) can also be passed which
        returns True for uncorrelated and False otherwise.

    significance_level: float
        A value between 0 and 1. If p_value < significance_level, the variables are
        considered uncorrelated.

    score: fun (default: f1-score)
        Any classification scoring metric from scikit-learn.
        https://scikit-learn.org/stable/modules/classes.html#classification-metrics

    return_summary: boolean (default: False)
        If True, returns a dataframe with details for each of the conditions checked.

    Returns
    -------
    The specified metric: float
        The metric specified by the `score` argument. By defaults returns the f1-score.

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> from pgmpy.metrics import CorrelationScore
    >>> alarm = load_model("bnlearn/alarm")
    >>> data = alarm.simulate(int(1e4))
    >>> scorer = CorrelationScore(
    ...     ci_test="chi_square", significance_level=0.05, return_summary=False
    ... )
    >>> scorer(X=data, causal_graph=alarm)
    0.911957950065703

    >>> scorer = CorrelationScore(
    ...     ci_test="chi_square", significance_level=0.05, return_summary=True
    ... )
    >>> scorer(X=data, causal_graph=alarm).head()
        var1            var2  stat_test  d_connected
    0   HISTORY          CVP      False        False
    1   HISTORY         PCWP      False        False
    2   HISTORY  HYPOVOLEMIA       True         True
    3   HISTORY   LVEDVOLUME      False        False
    4   HISTORY    LVFAILURE      False        False
    """

    _tags = {
        "name": "correlation_score",
        "requires_true_graph": False,
        "requires_data": True,
        "lower_is_better": False,
        "supported_graph_types": (DAG,),
        "is_default": True,
    }

    def __init__(
        self,
        ci_test=None,
        score=f1_score,
        significance_level=0.05,
        return_summary=False,
    ):
        self.ci_test = ci_test
        self.score = score
        self.significance_level = significance_level
        self.return_summary = return_summary

    def _evaluate(self, X, causal_graph):
        # Step 1: Validate inputs
        num_nodes = causal_graph.number_of_nodes()
        if num_nodes < 2:
            raise ValueError(
                "The causal graph must have at least 2 nodes to compute the"
                f" correlation score. Got {num_nodes} node(s)."
            )

        if not callable(self.score):
            raise ValueError(f"score should be scikit-learn classification metric. Got {self.score}")

        ci_test = get_ci_test(test=self.ci_test, data=X)

        # Step 2: Create a dataframe of every 2 combination of variables
        results = []
        for i, j in combinations(causal_graph.nodes(), 2):
            test_result = ci_test(
                X=i,
                Y=j,
                Z=[],
                significance_level=self.significance_level,
            )
            d_connected = not causal_graph.is_dconnected(start=i, end=j)

            results.append(
                {
                    "var1": i,
                    "var2": j,
                    "stat_test": test_result,
                    "d_connected": d_connected,
                }
            )

        results = pd.DataFrame(results)

        # Step 3: Return summary or metric
        if self.return_summary:
            return results
        else:
            return self.score(y_true=results["stat_test"].values, y_pred=results["d_connected"].values)
