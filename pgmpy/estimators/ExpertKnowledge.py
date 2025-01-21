from pgmpy.global_vars import logger


class ExpertKnowledge:
    """
    Class to specify expert knowledge for causal discovery algorithms.

    Expert knowledge is the prior knowledge about edges in the final structure of the
    graph learned by causal discovery algorithms. Currently, expert knowledge can
    provide information about edges that have to be present/absent in the final
    learned graph, maximum number of conditioning variables and temporal ordering
    of the variables.

    Parameters
    ----------
    forbidden_edges: iterable
            The set of directed edges that are to be absent in the final
            graph structure. Refer to the algorithm documentation for details
            on what to expect from the output. Defaults to None.

    required_edges: iterable
            The set of directed edges that are to be present in the final
            graph structure. Refer to the algorithm documentation for details
            on what to expect from the output. Defaults to None.

    temporal order: list of lists
            The temporal ordering of variables according to prior knowledge.
            Each list in the list of lists contains variables with the same
            temporal significance; the more prior (parental) variables (list) are at
            the start while the prioority decreases as we go down the list. Defaul None.

    max_cond_vars: int
            The maximum number of conditional variables to be used for statistical
            independce tests (e.g. PC algorithm). Default is 5.

    Examples
    --------
    Import an example model from pgmpy.utils

    >>> from pgmpy.utils import get_example_model
    >>> asia_model = get_example_model("asia")

    **Required and forbidden edges**

    >>> forb_edges = [("tub", "asia"), ("lung", "smoke")]
    >>> req_edges = [("smoke","bronc")]
    >>> expert_knowledge = ExpertKnowledge(required_edges=req_edges, forbidden_edges,
                                           max_cond_vars=4)

    **Use during structure learning**

    >>> from pgmpy.estimators import PC
    >>> data = BayesianModelSampling(asia_model).forward_sample(size=int(1e4))
    >>> est = PC(data)
    >>> est.estimate(
                variant="stable",
                expert_knowledge=expert_knowledge,
                show_progress=False,
            )
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

    def check_edges(self):
        """
        Method to check whether the ExpertKnowledge instance has user specified fixed/required edges.

        Parameters
        ----------

        Returns
        --------
        edge information: boolean
            Boolean specifying whether the instance of the class has required and/or forbidden
            edges specified by the user.
        """

        if self.required_edges != set() or self.forbidden_edges != set():
            return True
        else:
            return False

    def orient_pdag(self, pdag):
        """
        Method to check consistency and orient edges in a graph based on expert knowledge.

        The required and forbidden edges, if specified by the user, are correctly
        oriented in the graph object passed. In case of any conflict between the
        graph structure and a required/forbidden edge, the edge is ignored and
        a warning is raised at the end.

        Parameters
        ----------
        pdag: pgmpy.base.PDAG
            A  partial DAG with directed and undirected edges.

        Returns
        --------
        Model after edge orientation: pgmpy.base.DAG
            The partial DAG after accounting for specified required
            and forbidden edges.

        References
        ----------
        [1] https://doi.org/10.48550/arXiv.2306.01638
        """

        for edge in self.forbidden_edges:
            u, v = edge

            if pdag.has_edge(u, v) and pdag.has_edge(v, u):
                pdag.remove_edge(u, v)
            elif pdag.has_edge(u, v):
                logger.warning(
                    f"Specified expert knowledge conflicts with learned structure. Ignoring edge {u}->{v} from forbidden edges"
                )

        for edge in self.required_edges:
            u, v = edge

            if pdag.has_edge(u, v) and pdag.has_edge(v, u):
                pdag.remove_edge(v, u)
            elif pdag.has_edge(u, v) is False:
                logger.warning(
                    f"Specified expert knowledge conflicts with learned structure. Ignoring edge {u}->{v} from required edges"
                )

        return pdag
