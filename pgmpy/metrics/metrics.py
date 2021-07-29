from itertools import combinations

import pandas as pd
from sklearn.metrics import f1_score

from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork


def correlation_score(
    model,
    data,
    test="chi_square",
    significance_level=0.05,
    score=f1_score,
    return_summary=False,
):
    """
    Function to score how well the model structure represents the correlations
    in the data. The model doesn't need to be parameterized for this score.

    A Bayesian Network or DAG has d-connection property which can be used to
    determine which variables are correlated according to the model. This
    function uses this d-connection/d-separation property to compare the model
    with variable correlations in a given dataset. For every pair of variables
    in the dataset, a correlation test (specified by `test` argument) is done.
    We say that any two variables are correlated if the test's p-value <
    significance_level. The same pair of variables are then tested whether they
    are d-connected in the network structure or not. Finally a metric specified
    by `score` is computed by using the correlation test as the true value and
    d-connections as predicted values.

    Absense of correlation/d-separation is considered as the positive class for
    computing the metrics.

    Parameters
    ----------
    model: Instance of pgmpy.base.DAG or pgmpy.models.BayesianNetwork
        The model which needs to be tested.

    data: pandas.DataFrame instance
        The dataset against which to test the model structure.

    test: str or function
        The statistical tests to use for determining whether the variables in data
        are correlated or not. For discrete variables, the options are: 1) chi_square
        2) g_sq 3) log_likelihood 4) freeman_tuckey 5) modified_log_likelihood 6) neyman
        7) cressie_read. For continuous variables only one test is available: 1) pearsonr.
        A function with the signature fun(X, Y, Z, data) can also be passed which
        returns True for uncorrelated and False otherwise.

    significance_level: float
        A value between 0 and 1. If p_value < significance_level, the variables are
        considered uncorrelated.

    score: function (default: f1-score)
        Any classification scoring metric from scikit-learn.
        https://scikit-learn.org/stable/modules/classes.html#classification-metrics

    return_summary: boolean (default: False)
        If True, returns a dataframe with details for each of the conditions checked.

    Returns
    -------
    float: The metric specified by the `score` argument. By defults returns the f1-score.

    Examples
    --------
    >>> from pgmpy.utils import get_examples_model
    >>> from pgmpy.model_testing import correlation_score
    >>> alarm = get_example_model("alarm")
    >>> data = alarm.simulate(int(1e4))
    >>> correlation_score(alarm, data, test="chi_square", significance_level=0.05)
    0.911957950065703
    """
    from pgmpy.estimators.CITests import (
        chi_square,
        g_sq,
        log_likelihood,
        freeman_tuckey,
        modified_log_likelihood,
        neyman,
        cressie_read,
        pearsonr,
    )

    # Step 1: Checks for input arguments.
    supported_tests = {
        "chi_square": chi_square,
        "g_sq": g_sq,
        "log_likelihood": log_likelihood,
        "freeman_tuckey": freeman_tuckey,
        "modified_log_likelihood": modified_log_likelihood,
        "neyman": neyman,
        "cressie_read": cressie_read,
        "pearsonr": pearsonr,
    }

    if not isinstance(model, (DAG, BayesianNetwork)):
        raise ValueError(
            f"model must be an instance of pgmpy.base.DAG or pgmpy.models.BayesianNetwork. Got {type(model)}"
        )
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas.DataFrame instance. Got {type(data)}")
    elif set(model.nodes()) != set(data.columns):
        raise ValueError(
            f"Missing columns in data. Can't find values for the following variables: { set(model.nodes()) - set(data.columns) }"
        )
    elif (test not in supported_tests.keys()) and (not callable(test)):
        raise ValueError(f"test not supported and not a callable")

    elif not callable(score):
        raise ValueError(f"score should be scikit-learn classification metric.")

    # Step 2: Create a dataframe of every 2 combination of variables
    results = []
    for i, j in combinations(model.nodes(), 2):
        test_result = supported_tests[test](
            X=i,
            Y=j,
            Z=[],
            data=data,
            boolean=True,
            significance_level=significance_level,
        )
        d_connected = not model.is_dconnected(start=i, end=j)

        results.append(
            {"var1": i, "var2": j, "stat_test": test_result, "d_connected": d_connected}
        )

    results = pd.DataFrame(results)
    metric = score(
        y_true=results["stat_test"].values, y_pred=results["d_connected"].values
    )

    if return_summary:
        return results
    else:
        return metric


def log_probability_score(model, data):
    from pgmpy.metrics import BayesianModelProbability

    infer = BayesianModelProbability(model)
    return infer.score(data)


def structure_score(model, data, score_method, **kwargs):
    from pgmpy.estimators import K2Score, BDeuScore, BDsScore, BicScore

    supported_methods = {
        "k2": K2Score,
        "bdeu": BDeuScore,
        "bds": BDsScore,
        "bic": BicScore,
    }

    return supported_methods[score_method](data).score(model)
