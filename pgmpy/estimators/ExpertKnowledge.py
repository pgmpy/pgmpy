from itertools import chain

from pgmpy.global_vars import logger


class ExpertKnowledge:
    """
    Class to specify expert knowledge for causal discovery / structure learning algorithms.

    Expert knowledge is the prior knowledge about edges in the final structure
    of the graph learned by causal discovery algorithms. Users can provide
    information about edges that have to be present/absent in the final learned
    graph and the temporal / causal ordering of the variables.

    Parameters
    ----------
    forbidden_edges: iterable (default: None)
            The set of directed edges that are to be absent in the final
            graph structure. Refer to the algorithm documentation for details
            on how the argument is handled.

    required_edges: iterable (default: None)
            The set of directed edges that are to be present in the final
            graph structure. Refer to the algorithm documentation for details
            on how the argument is handled.

    temporal order: iterator (default: None)
            The temporal ordering of variables according to prior knowledge.
            Each list/structure in the (2 dimensional) iterator contains
            variables with the same temporal significance; the more prior
            (parental) variables are at the start while the priority decreases
            as we go move towards the end of the structure (iterator).

    Examples
    --------
    Import an example model from pgmpy.utils

    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.estimators import ExpertKnowledge, PC
    >>> from pgmpy.sampling import BayesianModelSampling
    >>> asia_model = get_example_model("asia")
    >>> cancer_model = get_example_model("cancer")

    **Required and forbidden edges**

    >>> forb_edges = [("tub", "asia"), ("lung", "smoke")]
    >>> req_edges = [("smoke","bronc")]
    >>> expert_knowledge = ExpertKnowledge(required_edges=req_edges, forbidden_edges)

    **Use during structure learning**

    >>> data = BayesianModelSampling(asia_model).forward_sample(size=int(1e4))
    >>> est = PC(data)
    >>> est.estimate(
    ...         variant="stable",
    ...         expert_knowledge=expert_knowledge,
    ...         show_progress=False,
    ...     )

    **Temporal order**

    >>> expert_knowledge = ExpertKnowledge(temporal_order=[["Pollution", "Smoker"], ["Cancer"], ["Dyspnoea", "Xray"]])

    **Use during structure learning**

    >>> data = BayesianModelSampling(cancer_model).forward_sample(size=int(1e4))
    >>> est = PC(data)
    >>> est.estimate(
    ...         variant="stable",
    ...         expert_knowledge=expert_knowledge,
    ...         show_progress=False,
    ...     )
    """

    def _validate_edges(self, edge_list):
        if not hasattr(edge_list, "__iter__"):
            raise TypeError(
                f"Expected iterator type for edge information. Got {type(edge_list)} instead."
            )
        elif type(edge_list) != set:
            return set(edge_list)
        else:
            return edge_list

    def _validate_temporal_order(self, nodes):
        """
        Method to check consistency of temporal order with nodes of a graph.

        The temporal order, if specified by the user, is currently used by the PC
        algorithm. The temporal order of all nodes present in the graph/dataset
        need to be present in the ExpertKnowledge instance.

        Parameters
        ----------
        nodes: iterable
            A collection of nodes present in a dataset/graph object.
        """
        if self.temporal_order == [[]]:
            return

        # Check if no node is present in multiple tiers
        if len(set.intersection(*map(set, self.temporal_order))) != 0:
            raise ValueError("Node found in multiple tiers of temporal order.")

        # Check if all nodes are present in the temporal order
        if set(chain(*self.temporal_order)) != set(nodes):
            raise ValueError(
                f"Missing nodes in temporal order - {set(nodes) - set(chain(*self.temporal_order))}"
            )

    def _get_temporal_ordering(self, temporal_order):
        """
        Method to check consistency of temporal order with nodes of a graph.

        The temporal order, if specified by the user, is currently used by the PC
        algorithm. The temporal order of all nodes present in the graph/dataset
        need to be present in the ExpertKnowledge instance.

        Parameters
        ----------
        temporal_order: iterator
            The temporal ordering of variables according to prior knowledge.

        Returns
        --------
        temporal_ordering: dict
            Dictionary with the tier (0, 1, 2, 3 etc.) for each node.
        """
        if not hasattr(temporal_order, "__iter__"):
            raise TypeError(
                f"Expected iterator type for temporal order. Got {type(temporal_order)} instead."
            )

        temporal_ordering = dict()
        for order, tier in enumerate(self.temporal_order):
            for node in tier:
                if node in temporal_ordering:
                    raise ValueError(
                        f"Variable {node} present in multiple tiers. Aborting"
                    )
                temporal_ordering[node] = order

        return temporal_ordering

    def _orient_temporal_forbidden_edges(self, graph, only_edges=True):
        """
        Add edge directions forbidden by the temporal order to forbidden_edges.

        If the graph contains the edge information, the edges are checked against
        the temporal order. In case the edges are not contained in the graph,
        the temporal order is used to find the forbidden edge directions.

        Parameters
        ----------
        graph: variable
            The graph for which temporal order is specified.

        only_edges: boolean (default: True)
            Whether to only consider the edges in the graph for orientation. If
            False, considers all possible edges between the variables.
        """
        if self.temporal_ordering == dict():
            return

        forbidden_edges = []
        if only_edges:
            for node in graph.nodes:
                for neighbor in graph.neighbors(node):
                    if self.temporal_ordering[neighbor] < self.temporal_ordering[node]:
                        forbidden_edges.append((node, neighbor))
        else:
            for tier in range(1, len(self.temporal_order)):
                for node in self.temporal_order[tier]:
                    for lower_tier in range(tier):
                        for lower_node in self.temporal_order[lower_tier]:
                            forbidden_edges.append((node, lower_node))

        self.forbidden_edges = self.forbidden_edges.union(forbidden_edges)

    def __init__(
        self,
        forbidden_edges=None,
        required_edges=None,
        temporal_order=None,
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

        self.temporal_order = temporal_order if temporal_order is not None else [[]]
        self.temporal_ordering = self._get_temporal_ordering(self.temporal_order)

    def apply_expert_knowledge(self, pdag):
        """
        Method to check consistency and orient edges in a graph based on expert knowledge.

        The required and forbidden edges, if specified by the user,
        are correctly oriented in the graph object passed. Temporal order,
        as specified, is also taken into account. In case of any conflict
        between the graph structure and a required/forbidden edge, the edge is
        ignored and a warning is raised.

        Parameters
        ----------
        pdag: pgmpy.base.PDAG
            A partial DAG with directed and undirected edges.

        Returns
        --------
        Model after edge orientation: pgmpy.base.DAG
            The partial DAG after accounting for specified required
            and forbidden edges.

        References
        ----------
        [1] https://doi.org/10.48550/arXiv.2306.01638
        """
        self._validate_temporal_order(pdag.nodes())
        self._orient_temporal_forbidden_edges(pdag)

        for edge in self.forbidden_edges:
            u, v = edge

            if pdag.has_edge(u, v) and pdag.has_edge(v, u):
                pdag.remove_edge(u, v)
            elif pdag.has_edge(u, v):
                logger.warning(
                    f"Specified expert knowledge conflicts with learned structure. Ignoring edge {u}->{v} from forbidden edges."
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
