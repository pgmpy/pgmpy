class ExpertKnowledge:
    """
    Class to specify expert knowledge for causal discovery algorithms.

    Expert knowledge is the prior knowledge about edges in the final structure of the
    graph learned by causal discovery algorithms. Currently, expert knowledge can
    provide information about edges that have to be present/absent in the final
    learned graph and a limited search space for edges.

    Parameters
    ----------
    white_list: list or None
            If a list of edges is provided as `white_list`, the search is
            limited to those edges. The resulting model will then only contain
            edges that are in `white_list`. Default: None

    black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded
            from the search and the resulting model will not contain any of
            those edges. Default: None

    fixed_edges: iterable
            A list of edges that will always be there in the final learned
            model. The algorithm will add these edges at the start of the
            algorithm and will never change it.
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

    def __init__(self, white_list=None, black_list=None, fixed_edges=None, **kwargs):
        self.white_list = (
            self._validate_edges(white_list) if white_list is not None else None
        )
        self.black_list = (
            self._validate_edges(black_list) if black_list is not None else set()
        )
        self.fixed_edges = (
            self._validate_edges(fixed_edges) if fixed_edges is not None else set()
        )

    def check_against_dag(self):
        pass
