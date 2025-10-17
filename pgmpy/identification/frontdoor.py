import networkx as nx

from pgmpy.base import DAG
from pgmpy.identification import Adjustment, BaseIdentification
from pgmpy.utils.sets import _powerset


class Frontdoor(BaseIdentification):
    """
    Given a causal graph, finds the set of variables satisfying frontdoor criterion.

    Given a causal graph with `exposure` and `outcome` roles specified, the
    `FrontdoorIdentification` class provides methods to find the set of variables
    satisfying the frontdoor criterion with respect to `exposure` and `outcome` in
    the causal graph.

    Parameters
    ----------
    variant: all | None
        If all, returns all possible frontdoor identification causal graphs.
        If None, returns one at random.

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> dag = DAG(
    ...     ebunch=[
    ...         ("X", "M"),
    ...         ("M", "Y"),
    ...         ("U", "X"),
    ...         ("U", "Y"),
    ...     ],
    ...     roles={"exposure": "X", "outcome": "Y"},
    ... )
    >>> dag_with_adj, is_identified = FrontdoorIdentification().identify(dag)
    >>> dag_with_adj.roles
    {'exposure': 'x1', 'outcome': 'y1', 'frontdoor': ['M']}
    >>> FrontdoorIdentification.validate(dag)
    True
    """

    def __init__(self, variant=None):
        self.supported_graph_types = (DAG,)
        self.variant = variant

    def _identify(self, causal_graph):
        exposure = causal_graph.get_role("exposure")
        outcome = causal_graph.get_role("outcome")

        possible_frontdoor_vars = (
            set(causal_graph.observed) - set(exposure) - set(outcome)
        )

        valid_frontdoor_graphs = []
        for s in _powerset(possible_frontdoor_vars):
            updated_causal_graph = causal_graph.with_role("frontdoor", s, inplace=False)
            if self.validate(updated_causal_graph):
                if self.variant is None:
                    return updated_causal_graph, True
                elif self.variant == "all":
                    valid_frontdoor_graphs.append(updated_causal_graph)
        if len(valid_frontdoor_graphs) > 0:
            return valid_frontdoor_graphs, True
        else:
            return causal_graph, False

    @staticmethod
    def _is_valid_adjustment_set(causal_graph, X, Y, Z):
        causal_graph_copy = causal_graph.copy()
        causal_graph_copy.without_role("exposure", inplace=True)
        causal_graph_copy.without_role("outcome", inplace=True)
        causal_graph_copy.without_role("adjustment", inplace=True)

        causal_graph_copy.with_role("exposure", X, inplace=True)
        causal_graph_copy.with_role("outcome", Y, inplace=True)
        causal_graph_copy.with_role("adjustment", Z, inplace=True)

        return Adjustment().validate(causal_graph_copy)

    def _validate(self, causal_graph):
        """ """
        exposure = causal_graph.get_role("exposure")[0]
        outcome = causal_graph.get_role("outcome")[0]
        Z = causal_graph.get_role("frontdoor")

        # 0. Get all directed paths from X to Y.  Don't check further if there aren't any.
        directed_paths = list(nx.all_simple_paths(causal_graph, exposure, outcome))

        if len(directed_paths) == 0:
            return False

        # 1. Z intercepts all directed paths from X to Y
        unblocked_directed_paths = [
            path for path in directed_paths if not any(zz in path for zz in Z)
        ]

        if len(unblocked_directed_paths) > 0:
            return False

        # 2. There is no backdoor path from X to Z.
        unblocked_backdoor_paths_X_Z = [
            zz
            for zz in Z
            if not self._is_valid_adjustment_set(
                causal_graph, X=exposure, Y=zz, Z=set()
            )
        ]

        if unblocked_backdoor_paths_X_Z:
            return False

        # 3. All back-door paths from Z to Y are blocked by X
        valid_backdoor_sets = []

        for zz in Z:
            valid_backdoor_sets.append(
                self._is_valid_adjustment_set(causal_graph, X=zz, Y=outcome, Z=exposure)
            )
        if not all(valid_backdoor_sets):
            return False

        return True
