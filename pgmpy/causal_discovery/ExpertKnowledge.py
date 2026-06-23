from itertools import chain, permutations

from sklearn.base import BaseEstimator

from pgmpy import logger
from pgmpy.ci_tests import get_ci_test


class ExpertKnowledge(BaseEstimator):
    """
    Class to specify expert knowledge for causal discovery / structure learning algorithms.

    Expert knowledge is the prior knowledge about edges in the final structure of the graph learned by causal discovery
    algorithms. Users can provide information about edges that have to be present/absent in the final learned graph and
    the temporal / causal ordering of the variables.

    Parameters
    ----------
    forbidden_edges: iterable (default: None)
            The set of directed edges that are to be absent in the final graph structure. Refer to the algorithm
            documentation for details on how the argument is handled.

    required_edges: iterable (default: None)
            The set of directed edges that are to be present in the final graph structure. Refer to the algorithm
            documentation for details on how the argument is handled.

    search_space: iterable (default: None)
            The set of directed edges that form the search space for the structure learning algorithm (a white list of
            all possible edges). Refer to the algorithm documentation for details on how the argument is handled.

            If both `search_space` and `screening_method` are specified, the generated search space is merged with the
            user-provided search space.

    temporal_order: iterator (default: None)
            The temporal ordering of variables according to prior knowledge. Each list/structure in the (2 dimensional)
            iterator contains variables with the same temporal significance. The order is defined as: [(variables at the
            root / 1st temporal order), (variables at 2nd temporal order), ... (leaf nodes / last temporal order)]

    screening_method: str | BaseCITest | callable (default: None)
            Conditional independence test used for generating a search space from data.

            If `search_space` is also specified, the generated search space is merged with the user-provided search
            space.

    significance_level: float (default: 0.05)
            Significance threshold used for screening variable pairs when generating a search space.

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> from pgmpy.causal_discovery import ExpertKnowledge, PC
    >>> asia_model = load_model("bnlearn/asia")
    >>> cancer_model = load_model("bnlearn/cancer")

    **Required and forbidden edges**

    >>> expert_knowledge = ExpertKnowledge(
    ...     required_edges=[("smoke", "bronc")],
    ...     forbidden_edges=[("tub", "asia"), ("lung", "smoke")],
    ... )
    >>> data = asia_model.simulate(n_samples=int(1e4), seed=42)
    >>> est = PC(expert_knowledge=expert_knowledge).fit(data)  # doctest: +SKIP

    **Temporal order**

    >>> expert_knowledge = ExpertKnowledge(
    ...     temporal_order=[["Pollution", "Smoker"], ["Cancer"], ["Dyspnoea", "Xray"]]
    ... )
    >>> data = cancer_model.simulate(n_samples=int(1e4), seed=42)
    >>> est = PC(expert_knowledge=expert_knowledge).fit(data)  # doctest: +SKIP

    **CI-test-based search space generation**

    The search space is generated from the data inside ``fit``; no manual step is required.

    >>> expert_knowledge = ExpertKnowledge(
    ...     screening_method="chi_square",
    ...     significance_level=0.05,
    ... )
    >>> est = PC(expert_knowledge=expert_knowledge).fit(data)  # doctest: +SKIP
    """

    def __init__(
        self,
        forbidden_edges=None,
        required_edges=None,
        temporal_order=None,
        search_space=None,
        screening_method=None,
        significance_level=0.05,
        **kwargs,
    ):
        self.forbidden_edges = self._validate_edges(forbidden_edges) if forbidden_edges is not None else set()
        self.required_edges = self._validate_edges(required_edges) if required_edges is not None else set()

        self.search_space = self._validate_edges(search_space) if search_space is not None else set()
        self.screening_method = screening_method
        self.significance_level = significance_level
        if not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1.")

        self.temporal_order = temporal_order if temporal_order is not None else [[]]
        self.temporal_ordering = self._get_temporal_ordering(self.temporal_order)

    def __repr__(self):
        # Calculate total number of nodes in temporal order
        n_temporal_nodes = sum(len(tier) for tier in self.temporal_order)

        return (
            f"Expert Knowledge: {len(self.required_edges)} required edges, "
            f"{len(self.forbidden_edges)} forbidden edges, "
            f"temporal order on {n_temporal_nodes} nodes, and "
            f"{len(self.search_space)} search space edges"
        )

    def __str__(self):
        lines = ["Expert Knowledge:"]

        if self.required_edges:
            lines.append(f"Required Edges: {self.required_edges}")
        if self.forbidden_edges:
            lines.append(f"Forbidden Edges: {self.forbidden_edges}")
        if self.search_space:
            lines.append(f"Search Space: {self.search_space}")
        if self.temporal_order and self.temporal_order != [[]]:
            lines.append(f"Temporal Order: {self.temporal_order}")

        return "\n".join(lines)

    def _validate_edges(self, edge_list):
        if not hasattr(edge_list, "__iter__"):
            raise TypeError(f"Expected iterator type for edge information. Got {type(edge_list)} instead.")
        elif not isinstance(edge_list, set):
            return set(edge_list)
        else:
            return edge_list

    def _get_temporal_ordering(self, temporal_order):
        """
        Build the mapping from each variable to its temporal tier.

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
            raise TypeError(f"Expected iterator type for temporal order. Got {type(temporal_order)} instead.")

        temporal_ordering = dict()
        for order, tier in enumerate(temporal_order):
            for node in tier:
                if node in temporal_ordering:
                    raise ValueError(f"Variable {node} present in multiple tiers. Aborting")
                temporal_ordering[node] = order

        return temporal_ordering

    def _screening_search_space(self, data):
        """
        Compute the search space implied by a marginal independence test (Z=[]).

        Variable pairs that reject marginal independence according to the specified CI
        test are returned (in both directions). Unlike
        :meth:`_generate_screening_search_space`, this helper does not mutate the object.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset used for evaluating variable dependencies.

        Returns
        -------
        set
            Variable pairs (both directions) that reject marginal independence.
        """
        ci_test = get_ci_test(test=self.screening_method, data=data)

        generated_search_space = set()

        columns = list(data.columns)

        for i, X in enumerate(columns):
            for Y in columns[i + 1 :]:
                if not ci_test.is_independent(
                    X=X,
                    Y=Y,
                    Z=[],
                    significance_level=self.significance_level,
                ):
                    generated_search_space.add((X, Y))
                    generated_search_space.add((Y, X))

        return generated_search_space

    def fit(self, data=None):
        """
        Resolve the expert knowledge into fitted attributes for structure learning.

        Computes sklearn-style fitted attributes (suffixed with ``_``) from the declarative constructor inputs and,
        when provided, the dataset.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Dataset used for structure learning. The column names define the variable set used to build the search-space
            complement and, when ``screening_method`` is specified, to run the marginal independence tests. ``data`` is
            required only when ``screening_method`` or ``search_space`` is specified; the forbidden/required/temporal
            knowledge is resolved without it.

        Returns
        -------
        self : ExpertKnowledge
            The instance with the fitted attributes set.

        Attributes
        ----------
        forbidden_edges_ : set
            Directed edges that must be absent: the union of the user-specified forbidden edges, the temporal-order
            complement (any edge from a later tier to an earlier tier), and the complement of the search space.

        required_edges_ : set
            Directed edges that must be present.

        search_space_ : set
            The resolved search space (user-provided whitelist merged with the screening result).

        temporal_ordering_ : dict
            Mapping from each variable to its temporal tier.
        """
        if data is None:
            if self.screening_method is not None:
                raise ValueError("`data` is required to fit when `screening_method` is specified.")
            if self.search_space:
                raise ValueError("`data` is required to fit when `search_space` is specified.")

        # Validate the temporal order (if given) covers exactly the data's variables.
        if self.temporal_order != [[]]:
            if len(set.intersection(*map(set, self.temporal_order))) != 0:
                raise ValueError("Node found in multiple tiers of temporal order.")
            if data is not None and set(chain(*self.temporal_order)) != set(data.columns):
                missing = set(data.columns) - set(chain(*self.temporal_order))
                raise ValueError(f"Missing nodes in temporal order - {missing}")

        self.temporal_ordering_ = dict(self.temporal_ordering)
        self.required_edges_ = set(self.required_edges)

        # Resolve the search space (user whitelist + screening) without mutating the inputs.
        self.search_space_ = set(self.search_space)
        if self.screening_method is not None:
            self.search_space_ |= self._screening_search_space(data)

        # forbidden_edges_ = user forbidden edges
        #   + temporal complement: any edge from a later tier to an earlier tier
        #   + search-space complement (when a search space is in play).
        forbidden = set(self.forbidden_edges)
        for tier in range(1, len(self.temporal_order)):
            for node in self.temporal_order[tier]:
                for lower_tier in range(tier):
                    for lower_node in self.temporal_order[lower_tier]:
                        forbidden.add((node, lower_node))
        if data is not None and (self.search_space or self.screening_method is not None):
            forbidden |= set(permutations(data.columns, 2)) - self.search_space_
        self.forbidden_edges_ = forbidden

        return self

    def apply_to(self, graph):
        """
        Orient the edges of ``graph`` according to the fitted expert knowledge.

        Uses the fitted ``forbidden_edges_`` and ``required_edges_`` attributes (set by
        :meth:`initialize`) to orient still-undirected edges of ``graph`` in place.
        Required edges ``(u, v)`` are oriented ``u -> v``; forbidden edges ``(u, v)`` are
        oriented away from the forbidden direction (``v -> u``). Edges that already
        conflict with the learned structure are left unchanged and a warning is logged.

        Unlike :meth:`apply_expert_knowledge`, this method does not mutate the expert
        knowledge object; temporal constraints are already resolved into
        ``forbidden_edges_`` by :meth:`initialize`.

        Parameters
        ----------
        graph : pgmpy.base.PDAG
            A partial DAG with directed and undirected edges. Modified in place.

        Returns
        -------
        graph : pgmpy.base.PDAG
            The same graph instance, after edge orientation.

        References
        ----------
        - :cite:p:`ankan_textor_2023`
        """
        for u, v in self.forbidden_edges_:
            if graph.has_undirected_edge(u, v):
                graph.orient_undirected_edge(v, u, inplace=True)
            elif graph.has_edge(u, v):
                logger.warning(
                    f"Specified expert knowledge conflicts with learned structure. "
                    f"Ignoring edge {u}->{v} from forbidden edges."
                )

        for u, v in self.required_edges_:
            if graph.has_undirected_edge(u, v):
                graph.orient_undirected_edge(u, v, inplace=True)
            elif graph.has_edge(u, v) is False:
                logger.warning(
                    f"Specified expert knowledge conflicts with learned structure. "
                    f"Ignoring edge {u}->{v} from required edges"
                )

        return graph
