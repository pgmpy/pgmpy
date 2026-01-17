from pgmpy.metrics import _BaseMetric


class FisherC(_BaseMetric):
    """
    Returns a p-value for testing whether the given data is faithful to the
    model structure's constraints.

    Each missing edge in a model structure implies a CI statement. This test
    uses constructs implied CIs such that they are independent of each other,
    run statistical tests for each of them on the data, and finally combines
    them using the Fisher's method.

    Parameters
    ----------
    ci_test: function
        The function for statistical test. Can be either any of the tests in
        pgmpy.estimators.CITests or any custom function of the same form.

    compute_rmsea: bool (default: False)
        While calculating Fisher C statistic if RMSEA value required should be
        included in method call as True. Returns a tuple of (p-value, rmsea) if
        True otherwise only the p-value.

    show_progress: bool (default: True)
        Whether to show the progress of testing.

    Returns
    -------
    float (default): The p-value for the fit of the model structure to the data. A low
        p-value (e.g. <0.05) represents that the model structure doesn't fit the
        data well. This is returned if the compute_rmsea parameter is False.

    tuple: A (float, float) tuple packing p-value and rmsea value is returned if RMSEA
            computation is necessary, i.e., compute_rmsea is True in the method call

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.metrics import implied_cis
    >>> from pgmpy.estimators.CITests import chi_square
    >>> model = get_example_model("cancer")
    >>> df = model.simulate(int(1e3))
    >>> fisher_c(model=model, data=df, ci_test=chi_square, show_progress=False)
    0.7504
    """
    def __init__(self, ci_test=None, compute_rmsea=False, show_progress=True):
        self.ci_test = ci_test
        self.compute_rmsea = compute_rmsea
        self.show_progress = show_progress

    def evaluate(self, X, estimated_causal_model):
        if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
            raise ValueError(
                f"model must be an instance of DAG or DiscreteBayesianNetwork. Got {type(model)}"
            )

        if len(model.latents) > 0:
            raise ValueError(
                "This test can not be performed on models with latent variables."
            )

        cis = []

        if self.show_progress and config.SHOW_PROGRESS:
            comb_iter = tqdm(
                combinations(model.nodes(), 2), total=math.comb(len(model.nodes()), 2)
            )
        else:
            comb_iter = combinations(model.nodes(), 2)

        for u, v in comb_iter:
            if not ((u in model[v]) or (v in model[u])):
                Z = set(model.predecessors(u)).union(model.predecessors(v))
                test_results = self.ci_test(X=u, Y=v, Z=Z, data=data, boolean=False)
                cis.append([u, v, Z, test_results[1]])
        cis = pd.DataFrame(cis, columns=["u", "v", "cond_vars", "p_value"])
        cis.loc[:, "p_value"] = cis.loc[:, "p_value"].clip(lower=1e-6)

        C = -2 * np.log(cis.loc[:, "p_value"]).sum()
        p_value = 1 - stats.chi2.cdf(C, df=2 * cis.shape[0])
        rmsea = np.nan

        if self.compute_rmsea:
            if len(data) != 1 and len(cis) != 0:
                rmsea = np.sqrt(
                    max((C - 2 * len(cis)) / (2 * len(cis) * (len(data) - 1)), 0)
                )
            return (p_value, rmsea)

        return p_value
