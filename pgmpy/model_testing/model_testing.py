from itertools import combinations

import pandas as pd
from sklearn.metrics import f1_score

from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork


def structure_test(
    model,
    data,
    test="chi_square",
    significance_level=0.05,
    return_summary=False,
    score=f1_score,
):
    """
    Function to test how well the model structure represents the data. This function
    compares whether the correlated variables in the data are also d-connected in
    the model structure or not. No parameters for the model is required for
    running this test.

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
        A function with the signature fun(X, Y, Z, data, boolean) can also be passed which
        returns True for uncorrelated and False otherwise.

    significance_level: float
        A value between 0 and 1. If p_value > significance_level, the variables are
        considered uncorrelated.

    return_summary: boolean (default: False)
        If True, returns a dataframe with details for each of the conditions checked.

    Returns
    -------
    float: F1-score

    Examples
    --------
    >>>

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
        return f"F1-score: {metric}"


def log_probability_score(model, data):
    from pgmpy.model_testing import BayesianModelProbability

    infer = BayesianModelProbability(model)
    return infer.score(data)
