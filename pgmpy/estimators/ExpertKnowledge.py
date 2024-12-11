class ExpertKnowledge:
    """
    Class to specify expert knowledge for causal discovery algorithms.
    Currently, expert knowledge can provide information about edges that have to be present/absent in
    the final learned graph and a limited search space for edges.
    Parameters
    ----------
    white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None
    black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None
    fixed_edges: iterable
            A list of edges that will always be there in the final learned model.
            The algorithm will add these edges at the start of the algorithm and
            will never change it.
    """

    def _check_list(self, edge_list):
        if type(edge_list) not in [list, set]:
            raise TypeError(f"TypeError: expected list data type for white/black list")

    def _check_set(self, edge_set):
        if not hasattr(edge_set, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")

    def __init__(self, white_list=None, black_list=None, fixed_edges=set(), **kwargs):

        if white_list:
            self._check_list(white_list)
        if black_list:
            self._check_list(black_list)
        if fixed_edges:
            self._check_set(fixed_edges)

        self.white_list = white_list if white_list is not None else None
        self.black_list = black_list if black_list is not None else None
        self.fixed_edges = set(fixed_edges)

    def check_against_dag(self):
        pass
