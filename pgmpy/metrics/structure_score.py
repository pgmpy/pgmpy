from pgmpy.base import DAG
from pgmpy.estimators.StructureScore import get_scoring_method
from pgmpy.metrics import _BaseUnsupervisedMetric


class StructureScore(_BaseUnsupervisedMetric):
    """
    Uses the standard model scoring methods to give a score for each structure.
    The score doesn't have very straight forward interpretability but can be
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
    >>> from pgmpy.metrics import StructureScore
    >>> model = get_example_model("alarm")
    >>> data = model.simulate(int(1e4))
    >>> scorer = StructureScore(scoring_method="bic-g")
    >>> scorer(X=data, causal_graph=model)
    -106665.9383064447
    """

    _tags = {
        "name": "structure_score",
        "requires_true_graph": False,
        "requires_data": True,
        "lower_is_better": False,
        "supported_graph_types": (DAG,),
    }

    def __init__(self, scoring_method=None):
        self.scoring_method = scoring_method

    def _evaluate(self, X, causal_graph, **kwargs):
        scoring_method = get_scoring_method(
            self.scoring_method, data=X, use_cache=False, **kwargs
        )[0]
        return scoring_method.score(causal_graph)
