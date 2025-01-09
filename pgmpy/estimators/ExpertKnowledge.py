class ExpertKnowledge:
    """
    Class to specify expert knowledge for causal discovery algorithms.

    Expert knowledge is the prior knowledge about edges in the final structure of the
    graph learned by causal discovery algorithms. Currently, expert knowledge can
    provide information about edges that have to be present/absent in the final
    learned graph and a limited search space for edges.

    Parameters
    ----------
    forbidden_edges: iterable
            The set of directed edges that must be absent in the final
            graph structure. Defaults to None.
    required_edges: iterable
            The set of directed edges that must be present in the final
            graph structure. Defaults to None.
    max_cond_vars: int
            The maximum number of conditional variables to be used for statistical
            independce tests (e.g. PC algorithm). Default is 5.
    """

    def _validate_edges(self, edge_list):
        if not hasattr(edge_list, "__iter__"):
            raise TypeError(
                f"expected iterator type for edge information. Recieved {type(edge_list)} instead"
            )
        elif type(edge_list) != set:
            return set(edge_list)
        else:
            return edge_list

    def __init__(
        self,
        forbidden_edges=None,
        required_edges=None,
        temporal_order=None,
        max_cond_vars=5,
        **kwargs,
    ):
        self.forbidden_edges = (
            self._validate_edges(forbidden_edges)
            if forbidden_edges is not None
            else set()
        )
        self.required_edges = (
            self._validate_edges(required_edges)
            if required_edges is not None
            else set()
        )
        self.max_cond_vars = max_cond_vars

    def check_against_dag(self):
        pass
