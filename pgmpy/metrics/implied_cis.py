from pgmpy.metrics import _BaseMetric


class ImpliedCIs(_BaseMetric):
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
    >>> model = get_example_model("cancer")
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

    def __init__(self, ci_test, show_progress=True):
        self.ci_test = ci_test

    def evaluate(X, estimated_causal_graph):
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

    def __call__(self, X, estimated_causal_graph):
        return self.evaluate(X, estimated_causal_graph)
