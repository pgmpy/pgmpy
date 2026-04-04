from pgmpy.base import DAG
from pgmpy.metrics import _BaseUnsupervisedMetric
from pgmpy.structure_score import get_scoring_method


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

    Returns
    -------
    Model score: float
        A score value for the model.

    Examples
    --------
    >>> from pgmpy.factors.discrete import TabularCPD
    >>> from pgmpy.global_vars import config
    >>> from pgmpy.metrics import StructureScore
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> config.SHOW_PROGRESS = False
    >>> model = DiscreteBayesianNetwork([("A", "B")])
    >>> model.add_cpds(
    ...     TabularCPD("A", 2, [[0.6], [0.4]]),
    ...     TabularCPD(
    ...         "B", 2, [[0.7, 0.2], [0.3, 0.8]], evidence=["A"], evidence_card=[2]
    ...     ),
    ... )
    >>> model.check_model()
    True
    >>> data = model.simulate(int(10000), seed=42)
    >>> scorer = StructureScore(scoring_method="bic-d")
    >>> scorer(X=data, causal_graph=model)
    np.float64(-12425.623548763984)
    """

    _tags = {
        "name": "structure_score",
        "requires_true_graph": False,
        "requires_data": True,
        "lower_is_better": False,
        "supported_graph_types": (DAG,),
        "is_default": False,
    }

    def __init__(self, scoring_method=None):
        self.scoring_method = scoring_method

    def _evaluate(self, X, causal_graph):
        scoring_method = get_scoring_method(self.scoring_method, data=X)
        return scoring_method.score(causal_graph)
