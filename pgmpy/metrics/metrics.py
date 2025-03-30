import math
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.models import DiscreteBayesianNetwork


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
    are d-connected in the network structure or not. Finally, a metric specified
    by `score` is computed by using the correlation test as the true value and
    d-connections as predicted values.

    Absense of correlation/d-separation is considered as the positive class for
    computing the metrics.

    Parameters
    ----------
    model: Instance of pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork
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
    The specified metric: float
        The metric specified by the `score` argument. By defults returns the f1-score.

    Examples
    --------
    >>> from pgmpy.utils import get_examples_model
    >>> from pgmpy.metrics import correlation_score
    >>> alarm = get_example_model("alarm")
    >>> data = alarm.simulate(int(1e4))
    >>> correlation_score(alarm, data, test="chi_square", significance_level=0.05)
    0.911957950065703
    """
    from pgmpy.estimators.CITests import get_ci_test

    # Step 1: Checks for input arguments.
    if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
        raise ValueError(
            f"model must be an instance of pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork. Got {type(model)}"
        )
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas.DataFrame instance. Got {type(data)}")
    elif set(model.nodes()) != set(data.columns):
        raise ValueError(
            f"Missing columns in data. Can't find values for the following variables: { set(model.nodes()) - set(data.columns) }"
        )

    supported_test = get_ci_test(test)

    if not callable(score):
        raise ValueError(f"score should be scikit-learn classification metric.")

    # Step 2: Create a dataframe of every 2 combination of variables
    results = []
    for i, j in combinations(model.nodes(), 2):
        test_result = supported_test(
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


def log_likelihood_score(model, data):
    """
    Computes the log-likelihood of a given dataset i.e. P(data | model).

    The log-likelihood measure can be used to check how well the specified
    model describes the data. This method requires the parameters of the model to be
    specified as well. Direct interpretation of this score is difficult but can
    be used to compare the fit of two or more models. A higher score means ab
    better fit.

    Parameters
    ----------
    model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork instance
        The model whose score needs to be computed.

    data: pd.DataFrame instance
        The dataset against which to score the model.

    Examples
    --------
    >>> from pgmpy.metrics import log_likelihood_score
    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model("alarm")
    >>> data = model.simulate(int(1e4))
    >>> log_likelihood_score(model, data)
    -103818.57516969478
    """
    # Step 1: Check the inputs
    if not isinstance(model, DiscreteBayesianNetwork):
        raise ValueError(f"Only Bayesian Networks are supported. Got {type(model)}.")
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas.DataFrame instance. Got {type(data)}")
    elif set(model.nodes()) != set(data.columns):
        raise ValueError(
            f"Missing columns in data. Can't find values for the following variables: { set(model.nodes()) - set(data.columns) }"
        )

    model.check_model()

    # Step 2: Compute the log-likelihood
    from pgmpy.metrics import BayesianModelProbability

    return BayesianModelProbability(model).score(data)


def structure_score(model, data, scoring_method="bic-g", **kwargs):
    """
    Uses the standard model scoring methods to give a score for each structure.
    The score doesn't have very straight forward interpretebility but can be
    used to compare different models. A higher score represents a better fit.
    This method only needs the model structure to compute the score and parameters
    aren't required.

    Parameters
    ----------
    model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork instance
        The model whose score needs to be computed.

    data: pd.DataFrame instance
        The dataset against which to score the model.

    scoring_method: str
        Options are: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g, ll-cg, aic-cg, bic-cg

    kwargs: kwargs
        Any additional parameters that needs to be passed to the
        scoring method. Check pgmpy.estimators.StructureScore for details.

    Returns
    -------
    Model score: float
        A score value for the model.

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.metrics import structure_score
    >>> model = get_example_model('alarm')
    >>> data = model.simulate(int(1e4))
    >>> structure_score(model, data, scoring_method="bic-g")
    -106665.9383064447
    """
    from pgmpy.estimators import (
        AIC,
        BIC,
        K2,
        AICCondGauss,
        AICGauss,
        BDeu,
        BDs,
        BICCondGauss,
        BICGauss,
        LogLikelihoodCondGauss,
        LogLikelihoodGauss,
    )

    supported_methods = {
        "k2": K2,
        "bdeu": BDeu,
        "bds": BDs,
        "bic-d": BIC,
        "aic-d": AIC,
        "ll-g": LogLikelihoodGauss,
        "aic-g": AICGauss,
        "bic-g": BICGauss,
        "ll-cg": LogLikelihoodCondGauss,
        "aic-cg": AICCondGauss,
        "bic-cg": BICCondGauss,
    }

    # Step 1: Test the inputs
    if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
        raise ValueError(
            f"model must be an instance of pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork. Got {type(model)}"
        )
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas.DataFrame instance. Got {type(data)}")
    elif set(model.nodes()) != set(data.columns):
        raise ValueError(
            f"Missing columns in data. Can't find values for the following variables: { set(model.nodes()) - set(data.columns) }"
        )
    elif (scoring_method not in supported_methods.keys()) and (
        not callable(scoring_method)
    ):
        raise ValueError(f"scoring method not supported and not a callable")

    # Step 2: Compute the score and return
    return supported_methods[scoring_method](data, **kwargs).score(model)


def implied_cis(model, data, ci_test, show_progress=True):
    """
    Tests the implied Conditional Independences (CI) of the DAG in the given data.

    Each missing edge in a model structure implies a CI statement. If the
    distribution of the data is faithful to the constraints of the model
    structure, these CI statements should hold in the data as well. This
    function runs statistical tests for each implied CI on the given data.

    Parameters
    ----------
    model: pgmpy.base.DAG or any Bayesian Network
        The model whose structure need to be tested against the given data.

    data: pd.DataFrame
        Dataset to use for testing.

    ci_test: function
        The function for statistical test. Can be either any of the tests in
        pgmpy.estimators.CITests or any custom function of the same form.

    show_progress: bool (default: True)
        Whether to show the progress of testing.

    Returns
    -------
    pd.DataFrame: Returns a dataframe with each implied CI of the model and a p-value
        corresponding to it from the statistical test. A low p-value (e.g. <0.05)
        represents that the CI does not hold in the data.

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.metrics import implied_cis
    >>> from pgmpy.estimators.CITests import chi_square
    >>> model = get_example_model('cancer')
    >>> df = model.simulate(int(1e3))
    >>> implied_cis(model=model, data=df, ci_test=chi_square, show_progress=False)
           u         v cond_vars   p-value
    0  Pollution    Smoker        []  0.189851
    1  Pollution      Xray  [Cancer]  0.404149
    2  Pollution  Dyspnoea  [Cancer]  0.613370
    3     Smoker      Xray  [Cancer]  0.352665
    4     Smoker  Dyspnoea  [Cancer]  1.000000
    5       Xray  Dyspnoea  [Cancer]  0.888619
    """
    if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
        raise ValueError(
            f"model must be an instance of DAG or DiscreteBayesianNetwork. Got {type(model)}"
        )

    cis = []

    if show_progress and config.SHOW_PROGRESS:
        comb_iter = tqdm(
            combinations(model.nodes(), 2), total=math.comb(len(model.nodes()), 2)
        )
    else:
        comb_iter = combinations(model.nodes(), 2)

    for u, v in comb_iter:
        if not ((u in model[v]) or (v in model[u])):
            Z = list(model.minimal_dseparator(u, v))
            test_results = ci_test(X=u, Y=v, Z=Z, data=data, boolean=False)
            cis.append([u, v, Z, test_results[1]])
    cis = pd.DataFrame(cis, columns=["u", "v", "cond_vars", "p-value"])
    return cis


def fisher_c(model, data, ci_test, show_progress=True):
    """
    Returns a p-value for testing whether the given data is faithful to the
    model structure's constraints.

    Each missing edge in a model structure implies a CI statement. This test
    uses constructs implied CIs such that they are independent of each other,
    run statistical tests for each of them on the data, and finally combines
    them using the Fisher's method.

    Parameters
    ----------
    model: pgmpy.base.DAG or any Bayesian Network
        The model whose structure need to be tested against the given data.

    data: pd.DataFrame
        Dataset to use for testing.

    ci_test: function
        The function for statistical test. Can be either any of the tests in
        pgmpy.estimators.CITests or any custom function of the same form.

    show_progress: bool (default: True)
        Whether to show the progress of testing.

    Returns
    -------
    float: The p-value for the fit of the model structure to the data. A low
        p-value (e.g. <0.05) represents that the model structure doesn't fit the
        data well.

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.metrics import implied_cis
    >>> from pgmpy.estimators.CITests import chi_square
    >>> model = get_example_model('cancer')
    >>> df = model.simulate(int(1e3))
    >>> fisher_c(model=model, data=df, ci_test=chi_square, show_progress=False)
    0.7504
    """
    if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
        raise ValueError(
            f"model must be an instance of DAG or DiscreteBayesianNetwork. Got {type(model)}"
        )

    if len(model.latents) > 0:
        raise ValueError(
            f"This test can not be performed on models with latent variables."
        )

    cis = []

    if show_progress and config.SHOW_PROGRESS:
        comb_iter = tqdm(
            combinations(model.nodes(), 2), total=math.comb(len(model.nodes()), 2)
        )
    else:
        comb_iter = combinations(model.nodes(), 2)

    for u, v in comb_iter:
        if not ((u in model[v]) or (v in model[u])):
            Z = set(model.predecessors(u)).union(model.predecessors(v))
            test_results = ci_test(X=u, Y=v, Z=Z, data=data, boolean=False)
            cis.append([u, v, Z, test_results[1]])
    cis = pd.DataFrame(cis, columns=["u", "v", "cond_vars", "p_value"])
    cis.loc[:, "p_value"] = cis.loc[:, "p_value"].clip(lower=1e-6)

    C = -2 * np.log(cis.loc[:, "p_value"]).sum()
    p_value = 1 - stats.chi2.cdf(C, df=2 * cis.shape[0])
    return p_value


def SHD(true_model, est_model):
    """
    Computes the Structural Hamming Distance between `true_model` and `est_model`.

    SHD is defined as total number of basic operations: adding edges, removing
    edges, and reversing edges required to transform one graph to the other. It
    is a symmetrical measure.

    The code first accounts for edges that need to be deleted (from true_model),
    added (to true_model) and finally edges that need to be reversed. All operations
    count as 1.

    Parameters
    ----------
    true_model: pgmpy.base.DAG or pgmpy.base.CPDAG or pgmpy.models.DiscreteBayesianNetwork
        The first model to compare.
    est_model: pgmpy.base.DAG or pgmpy.base.CPDAG or pgmpy.models.DiscreteBayesianNetwork
        The second model to compare.

    Returns
    -------
    int:
        If both true_model and est_model are DAGs or Bayesian Networks returns
        an integer.

    Examples
    --------
    >>> from pgmpy.metrics import SHD
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> dag1 = DiscreteBayesianNetwork([(1, 2), (2, 3)])
    >>> dag2 = DiscreteBayesianNetwork([(2, 1), (2, 3)])
    >>> SHD(dag1, dag2)
    1
    """
    if set(true_model.nodes()) != set(est_model.nodes()):
        raise ValueError("The graphs must have the same nodes.")

    nodes_list = true_model.nodes()

    dag_true = nx.DiGraph(true_model.edges())
    m1 = nx.adjacency_matrix(dag_true, nodelist=nodes_list).todense()

    dag_est = nx.DiGraph(est_model.edges())
    m2 = nx.adjacency_matrix(dag_est, nodelist=nodes_list).todense()

    shd = 0

    s1 = m1 + m1.T
    s2 = m2 + m2.T

    # Edges that are in m1 but not in m2 (deletions from m1)
    ds = s1 - s2
    ind = np.where(ds > 0)
    m1[ind] = 0
    shd = shd + (len(ind[0]) / 2)

    # Edges that are in m2 but not in m1 (additions to m1)
    ind = np.where(ds < 0)
    m1[ind] = m2[ind]
    shd = shd + (len(ind[0]) / 2)

    # Edges that need to be simply reversed
    d = np.abs(m1 - m2)
    shd = shd + (np.sum((d + d.T) > 0) / 2)

    return int(shd)
