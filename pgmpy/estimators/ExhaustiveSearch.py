"""Deprecated compatibility shim for :class:`pgmpy.causal_discovery.ExhaustiveSearch`."""

import warnings

from pgmpy.causal_discovery import ExhaustiveSearch as _ExhaustiveSearch
from pgmpy.estimators import StructureEstimator
from pgmpy.structure_score import get_scoring_method


class ExhaustiveSearch(StructureEstimator):
    """
    Deprecated: use :class:`pgmpy.causal_discovery.ExhaustiveSearch` instead.

    Search class for exhaustive searches over all DAGs with a given set of variables.
    This class delegates to the canonical implementation in `pgmpy.causal_discovery`.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.NaN`.
        Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

    scoring_method: Instance of a `StructureScore`-subclass (`K2` is used as default)
        An instance of `K2`, `BDeu`, `BIC` or 'AIC'.
        This score is optimized during structure estimation by the `estimate`-method.

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    use_cache: bool (default: True)
        Retained for backwards compatibility; canonical structure scores cache
        local scores internally.
    """

    def __init__(self, data, scoring_method=None, use_cache=True, **kwargs):
        warnings.warn(
            """ExhaustiveSearch is deprecated and will be removed in v2.0. Please use
            pgmpy.causal_discovery.ExhaustiveSearch instead.""",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(data, **kwargs)
        self.scoring_method = get_scoring_method(scoring_method, self.data)

    def all_dags(self, nodes=None):
        """Legacy forwarder to the canonical `all_dags`.

        Refer to :meth:`pgmpy.causal_discovery.ExhaustiveSearch.all_dags` for
        parameter and return value details.
        """
        est = _ExhaustiveSearch()
        est.variables_ = self.variables
        return est.all_dags(nodes)

    def all_scores(self):
        """Legacy forwarder to the canonical `all_scores`.

        Refer to :meth:`pgmpy.causal_discovery.ExhaustiveSearch.all_scores` for
        parameter and return value details.
        """
        est = _ExhaustiveSearch(scoring_method=self.scoring_method)
        est.fit(self.data)
        return est.all_scores()

    def estimate(self):
        """
        Estimates the `DAG` structure that fits best to the given data set,
        according to the scoring method supplied in the constructor.
        Delegates to :class:`pgmpy.causal_discovery.ExhaustiveSearch`; refer to
        its documentation for details.

        Returns
        -------
        Estimated Model: pgmpy.base.DAG
            A `DAG` with maximal score.
        """
        est = _ExhaustiveSearch(scoring_method=self.scoring_method, return_type="dag")
        return est.fit(self.data).causal_graph_
