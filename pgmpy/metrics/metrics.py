import math
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import DAG, PDAG
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
    from pgmpy.estimators.CITests import get_callable_ci_test

    # Step 1: Checks for input arguments.
    if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
        raise ValueError(
            f"model must be an instance of pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork. Got {type(model)}"
        )
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas.DataFrame instance. Got {type(data)}")
    elif set(model.nodes()) != set(data.columns):
        raise ValueError(
            "Missing columns in data. Can't find values for the following variables: "
            f" {set(model.nodes()) - set(data.columns)}"
        )

    supported_test = get_callable_ci_test(test)

    if not callable(score):
        raise ValueError(
            f"score should be scikit-learn classification metric. Got {score}"
        )

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
            f"Missing columns in data. Can't find values for the following variables: "
            f" {set(model.nodes()) - set(data.columns)}"
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
            f"Missing columns in data. Can't find values for the following variables: "
            f" {set(model.nodes()) - set(data.columns)}"
        )
    elif (scoring_method not in supported_methods.keys()) and (
        not callable(scoring_method)
    ):
        raise ValueError(
            f"scoring method not supported and not a callable. Got {scoring_method}"
        )

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
            "This test can not be performed on models with latent variables."
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
    dag_true.add_nodes_from(list(nx.isolates(true_model)))
    m1 = nx.adjacency_matrix(dag_true, nodelist=nodes_list).todense()

    dag_est = nx.DiGraph(est_model.edges())
    dag_est.add_nodes_from(list(nx.isolates(est_model)))
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


def _latent_admg(dag: DAG, observed: list) -> nx.DiGraph:
    """
    Compute the latent‐projection ADMG L(G, observed_set) of a DAG G onto a subset observed_set ⊂ V.
    (Helper function for self_compatibility_graphical)

    Implements Definition 5 (latent ADMG) from [1] (Faller et al., AISTATS 2024)

    Parameters
    ----------
    dag : DAG
        The full DAG G on variables V (may include latent nodes).
    observed : list
        Subset observed_set ⊂ V to project onto (observed variables).

    Returns
    -------
    nx.DiGraph
    An ADMG over the observed nodes, where:
        - X→Y encodes a latent‐only directed chain X→…→Y
        - X↔Y is encoded by having both X→Y and Y→X in the grap.

    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.metrics.metrics import _latent_admg
    >>> # Example 1: A → H → B, observe only A,B
    >>> dag1 = DiscreteBayesianNetwork([('A', 'H'), ('H', 'B')])
    >>> admg1 = _latent_admg(dag1, observed=['A', 'B'])
    >>> set(admg1.edges()) == {('A', 'B')}
    True

    >>> # Example 2: Latent confounder H→A and H→B, observe only A,B
    >>> dag2 = DiscreteBayesianNetwork([('H','A'), ('H','B')])
    >>> admg2 = _latent_admg(dag2, observed=['A', 'B'])
    >>> set(admg2.edges()) == {('A','B'), ('B','A')}
    True

    >>> # Example 3: Unshielded collider with extension A→B←C→D, observe only A,C
    >>> dag3 = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D')])
    >>> admg3 = _latent_admg(dag3, observed=['A', 'C'])
    >>> set(admg3.edges()) == set()
    True

    >>> # Example 4: Pure confounding A→B and A→C, observe only B,C
    >>> dag4 = DiscreteBayesianNetwork([('A','B'), ('A','C')])
    >>> admg4 = _latent_admg(dag4, observed=['B', 'C'])
    >>> set(admg4.edges()) == {('B','C'), ('C','B')}
    True

    References
    ----------
    [1] Faller, P. M., et al. (2024).
    “Self‐compatibility: Evaluating Causal Discovery without Ground Truth.”
    In AISTATS. arXiv:2307.09552
    """
    full_directed = dag
    observed_set = set(observed)
    latent_set = set(dag.nodes()) - observed_set

    directed_edges = set()
    bidirected_edges = set()

    # 1) Preserve original observed→observed edges
    for u, v in dag.edges():
        if u in observed_set and v in observed_set:
            directed_edges.add((u, v))

    # 2) Add latent‐only directed chains via active trail + directed‐path check
    for u, v in combinations(observed_set, 2):
        #  take all the observed nodes except u and v
        observed_excluding_pair = list(observed_set - {u, v})
        reach_u = dag.active_trail_nodes(
            [u], observed=observed_excluding_pair, include_latents=False
        )[u]
        reach_v = dag.active_trail_nodes(
            [v], observed=observed_excluding_pair, include_latents=False
        )[v]

        # u→v?
        if v in reach_u:
            sub = full_directed.subgraph(latent_set | {u, v})
            if nx.has_path(sub, u, v):
                parents_v = {p for p in dag.predecessors(v) if p in observed_set}
                if parents_v <= {u}:
                    directed_edges.add((u, v))
        # v→u?
        if u in reach_v:
            sub = full_directed.subgraph(latent_set | {u, v})
            if nx.has_path(sub, v, u):
                parents_u = {p for p in dag.predecessors(u) if p in observed_set}
                if parents_u <= {v}:
                    directed_edges.add((v, u))

    # 3) Add bidirected edges only for *common‐parent* confounders, not colliders
    for u, v in combinations(observed_set, 2):
        # skip if already a two‐way directed link
        if (u, v) in directed_edges and (v, u) in directed_edges:
            continue

        # common‐parent latent l → u and l → v?
        for latent in latent_set:
            if dag.has_edge(latent, u) and dag.has_edge(latent, v):
                bidirected_edges.add((u, v))
                bidirected_edges.add((v, u))
                break
        else:
            # mixed non‐collider via observed child w:
            for w in observed_set - {u, v}:
                sub_uw = full_directed.subgraph(latent_set | {u, w})
                sub_vw = full_directed.subgraph(latent_set | {v, w})
                # here w → … → u and w → … → v through latents
                if nx.has_path(sub_uw, w, u) and nx.has_path(sub_vw, w, v):
                    bidirected_edges.add((u, v))
                    bidirected_edges.add((v, u))
                    break

    # 4) Assemble final ADMG
    admg = nx.DiGraph()
    admg.add_nodes_from(observed_set)
    admg.add_edges_from(directed_edges)
    admg.add_edges_from(bidirected_edges)
    return admg


def self_compatibility_graphical(
    estimator_class,
    data: pd.DataFrame,
    num_subsets: int = 50,
    subset_fraction: float = 0.8,
    random_state: int = None,
    **estimator_kwargs,
) -> float:
    """
    Implements Definition 6 from [1] (Faller et al., AISTATS 2024)

    Computes graphical self-compatibility by:
      1) Fitting a joint DAG on all variables.
      2) Projecting it to each random subset via Definition 5’s latent ADMG.
      3) Fitting a marginal DAG on each subset.
      4) Measuring SHD between the projected joint and the marginal.
      5) Averaging those SHDs.

    Parameters
    ----------
    estimator_class : class
        A pgmpy StructureEstimator (e.g. PC, HillClimbSearch).
    data : pd.DataFrame, shape (n_samples, n_vars)
        Observed dataset.
    num_subsets : int, default=50
        Number of random subsets to draw.
    subset_fraction : float in (0,1], default=0.8
        Fraction of variables to include in each subset.
    random_state : int or None
        RNG seed for reproducibility.
    **estimator_kwargs
        Keyword args forwarded to estimator_class(...).estimate().

    Returns
    -------
    float
        Mean SHD between latent-projected joint and marginal DAGs.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.factors.discrete import TabularCPD
    >>> from pgmpy.sampling import BayesianModelSampling
    >>> from pgmpy.estimators import HillClimbSearch
    >>> from pgmpy.metrics import self_compatibility_graphical
    >>> # 1) Build a simple BN X→Y
    >>> model = DiscreteBayesianNetwork([("X", "Y")])
    >>> model.add_cpds(
    ...     TabularCPD("X", 2, [[0.5], [0.5]]),
    ...     TabularCPD("Y", 2,
    ...                [[0.8, 0.2],
    ...                 [0.2, 0.8]],
    ...                evidence=["X"], evidence_card=[2])
    ... )
    >>> # 2) Sample 100 rows
    >>> df = BayesianModelSampling(model).forward_sample(size=100)
    >>> # 3) Compute graphical self-compatibility
    >>> score = self_compatibility_graphical(
    ...     HillClimbSearch,
    ...     df,
    ...     num_subsets=3,
    ...     subset_fraction=0.5,
    ...     scoring_method="bic-d"
    ... )
    >>> isinstance(score, float)
    True

    References
    ----------
    [1] Faller, P. M., et al. (2024).
        “Self‐compatibility: Evaluating Causal Discovery without Ground Truth.”
        In AISTATS. arXiv:2307.09552
    """
    if random_state is None:
        rng = np.random.RandomState()
    else:
        rng = np.random.RandomState(random_state)

    all_variables = list(data.columns)
    variable_count = len(all_variables)
    subset_size = max(2, int(np.floor(subset_fraction * variable_count)))

    # 1) Fit joint model on all variables
    joint_learner = estimator_class(data)
    joint = joint_learner.estimate(**estimator_kwargs)
    if isinstance(joint, PDAG):
        joint = joint.to_dag()
    shd_values = []

    for i in range(num_subsets):
        # a) sample subset of columns
        observed_set = rng.choice(
            all_variables, size=subset_size, replace=False
        ).tolist()
        sub_data = data[observed_set]
        joint_proj = _latent_admg(joint, observed_set)
        # b) fit marginal model
        marg_learner = estimator_class(sub_data)
        marginal = marg_learner.estimate(**estimator_kwargs)
        if isinstance(marginal, PDAG):
            marginal = marginal.to_dag()
        # d) project marginal onto observed_set
        marg_proj = _latent_admg(marginal, observed_set)
        # e) compute SHD
        shd_values.append(SHD(joint_proj, marg_proj))

    # 3) average SHD
    return float(np.mean(shd_values)) if shd_values else 0.0
