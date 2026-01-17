from pgmpy.metrics import _BaseMetric


class StructureScore(_BaseMetric):
    """
    Uses the standard model scoring methods to give a score for each structure.
    The score doesn't have very straight forward interpretebility but can be
    used to compare different models. A higher score represents a better fit.
    This method only needs the model structure to compute the score and parameters
    aren't required.

    Parameters
    ----------
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
    >>> model = get_example_model("alarm")
    >>> data = model.simulate(int(1e4))
    >>> structure_score(model, data, scoring_method="bic-g")
    -106665.9383064447
    """
    def __init__(self, scoring_method=None, **kwargs):
        self.scoring_method = scoring_methods

    def evaluate(X, estimated_causal_graph):
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

    def __call__(self, X, estimated_causal_graph):
        return self.evaluate(X, estimated_causal_graph)
