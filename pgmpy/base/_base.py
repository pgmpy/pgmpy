from collections import deque
from collections.abc import Hashable, Iterable
from itertools import combinations
from typing import Any

import networkx as nx
import pandas as pd

from pgmpy.base._algorithms import _GraphAlgorithms
from pgmpy.base._mixin_roles import _GraphRolesMixin
from pgmpy.base._plotting import _GraphPlotting


class _CoreGraph(nx.MultiGraph, _GraphAlgorithms, _GraphRolesMixin, _GraphPlotting):
    """
    Base class for all graph types in pgmpy.

    This class provides a generalized representation for all graph `edge_types` in pgmpy. All specific graph classes
    (e.g., `DAG`, `PDAG`, ...) inherit from `_CoreGraph`. Subclasses are expected to define their own
    `SUPPORTED_EDGE_TYPES` to restrict the kinds of edges they can store. It also provides generalized algorithms and
    methods that work across all inheriting graphs.

    Parameters
    ----------
    edge_list : iterable of tuples, optional
        A list or iterable of edges of the form [(variable1, variable2, edge_type), ... ] to add at initialization.
        The edge type must be one of the following: ``"--"``, ``"-o"``, ``"o-"``, ``"->"``, ``"<-"``, ``"o>"``,
        ``"<o"``, ``"<>"``, ``"oo"``.

    exposures : set, (default=set())
        Set of exposure variables in the graph. These are the variables that represent the treatment or intervention
        being studied in a causal analysis. Default is an empty set.

    outcomes : set, (default=set())
        Set of outcome variables in the graph. These are the variables that represent the response or dependent
        variables being studied in a causal analysis. Default is an empty set.

    latents : set of nodes, (default=set())
        A set of latent variables in the graph. These represent the unobserved variables in the model, i.e., variables
        that we don't have data for.

    roles : dict, optional (default=None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).

    Notes
    -----
    **Internal edge representation.**

    Although ``_CoreGraph`` exposes edges through string ``edge_type`` codes such as ``"->"`` or ``"<>"``, it does *not*
    store them as directed edges. Internally the class subclasses an *undirected* ``networkx.MultiGraph``, and the
    orientation of every edge is encoded by a pair of per-endpoint **markers** stored on the edge's data dictionary,
    keyed by the two node identifiers.

    Each endpoint carries one of three markers: ``"-"`` (a tail / no arrowhead), ``">"`` (an arrowhead), or ``"o"`` (a
    circle endpoint whose orientation is unspecified, as used in PAGs). An ``edge_type`` is therefore just the pair of
    endpoint markers: for an edge between ``u`` and ``v`` the dict ``{u: <u-marker>, v: <v-marker>}`` is stored on the
    edge. For example ``"A->B"`` is stored as ``{"A": "-", "B": ">"}`` and ``"A<>B"`` as ``{"A": ">", "B": ">"}``. The
    nine supported codes map to the following ``{u, v}`` marker dicts::

        --  : {u: "-", v: "-"}      ->  : {u: "-", v: ">"}      <-  : {u: ">", v: "-"}
        <>  : {u: ">", v: ">"}      o-  : {u: "o", v: "-"}      -o  : {u: "-", v: "o"}
        o>  : {u: "o", v: ">"}      <o  : {u: ">", v: "o"}      oo  : {u: "o", v: "o"}

    ``_to_markers`` performs the ``edge_type`` to marker-dict conversion and ``_to_edge_type`` the inverse;
    ``get_edges`` / ``get_edge_type`` use the latter to present edges back as string codes.

    Examples
    --------
    Create an empty `_CoreGraph` with no nodes and no edges.

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()

    Edges and vertices can be passed to the constructor as an edge list.

    >>> edges = [("A", "B", "->"), ("B", "C", "->")]
    >>> G = _CoreGraph(edge_list=edges)
    >>> G.get_edges(data=True)
    [('A', 'B', '->'), ('B', 'C', '->')]
    >>> G.get_edges(data=False)
    [('A', 'B'), ('B', 'C')]

    **Nodes:**

    Add one node,

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_node("A")
    >>> G.nodes
    NodeView(('A',))

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_edge("A", "B", "->")
    >>> G.get_edges(data=True)
    [('A', 'B', '->')]

    Remove one edge,

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
    >>> G = _CoreGraph(edge_list=edges)
    >>> G.remove_edge("A", "B", "->")
    >>> G.get_edges(data=True)
    [('B', 'C', '->'), ('C', 'D', '--')]

    **Exposures, Outcomes, and Latents:**

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(edge_list=edges)

    **Roles:**

    Add node's role.

    >>> G.exposures = "A"
    >>> G.outcomes = "C"
    >>> G.latents = "D"

    Checks for the node's role.

    >>> G.exposures
    {'A'}
    >>> G.outcomes
    {'C'}
    >>> G.latents
    {'D'}

    In addition to 'exposures', 'outcomes', and 'latents', you can add custom roles.

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(edge_list=edges)
    >>> G = G.with_role("Custom_role", "A", inplace=False)
    >>> G = G.with_role("latents", "D", inplace=False)
    >>> G.get_role_dict() == {"latents": ["D"], "Custom_role": ["A"]}
    True
    >>> G = G.without_role("Custom_role", "A", inplace=False)
    >>> G.get_role_dict() == {"latents": ["D"]}
    True

    References
    ----------
    - :cite:p:`ali_2009`
    - :cite:p:`zhang_2008`
    """

    SUPPORTED_EDGE_TYPES = frozenset(["--", "-o", "o-", "->", "<-", "o>", "<o", "<>", "oo"])

    def __init__(
        self,
        edge_list: Iterable[tuple[Hashable, Hashable, str]] | None = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        latents: set[Hashable] | None = None,
        roles: dict[str, str | Iterable[Hashable]] | None = None,
    ):
        super().__init__()
        if edge_list:
            edge_list = list(edge_list)  # materialize: _validate_edges below would exhaust a generator
            self._validate_edges(edge_list=edge_list)
            for u, v, edge_type in edge_list:
                self.add_edge(u, v, edge_type=edge_type)

        self.exposures = set() if exposures is None else set(exposures)
        self.outcomes = set() if outcomes is None else set(outcomes)
        self.latents = set() if latents is None else set(latents)

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as a dictionary.")

        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    def add_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: str,
    ) -> None:
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are not already in the graph.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers. Nodes must be hashable (and not None) Python objects.

        edge_type : str
            Must be one of the values in `SUPPORTED_EDGE_TYPES`. This argument is required.

        Returns
        -------
        None

        See Also
        --------
        add_edges_from : Add multiple edges at once.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph()
        >>> G.add_edge("A", "B", "->")
        >>> G.get_edges(data=True)
        [('A', 'B', '->')]

        """
        self._validate_edges(edge_list=[(u, v, edge_type)])

        # A pair of nodes can have at most one edge of a given type.
        if self.has_edge(u, v, edge_type):
            raise ValueError(f"Edge ({u}, {v}) of type '{edge_type}' already exists in {self.__class__.__name__}.")

        # Directed edges must not close a directed cycle.
        if edge_type == "->" and self.has_node(u) and self.has_node(v) and self.has_path(v, u, edge_types="->"):
            raise ValueError(f"Adding edge ({u}, {v}, '{edge_type}') would create a directed cycle.")
        if edge_type == "<-" and self.has_node(u) and self.has_node(v) and self.has_path(u, v, edge_types="->"):
            raise ValueError(f"Adding edge ({u}, {v}, '{edge_type}') would create a directed cycle.")

        markers = self._to_markers(edge=(u, v, edge_type))
        key = super().add_edge(u, v)
        self.edges[u, v, key].update({u: markers[u], v: markers[v]})

    def add_edges_from(
        self,
        edge_list: Iterable[tuple[Hashable, Hashable, str]],
    ) -> None:
        """
        Add all the edges in edge_list.

        Parameters
        ----------
        edge_list : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...].

        Returns
        -------
        None

        See Also
        --------
        add_edge : Add a single edge.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        >>> G.add_edges_from(edge_list=edges)
        >>> G.get_edges(data=True)
        [('A', 'B', '->'), ('B', 'C', '->')]

        """
        edge_list = list(edge_list)  # materialize: _validate_edges below would exhaust a generator
        self._validate_edges(edge_list=edge_list)
        for u, v, edge_type in edge_list:
            self.add_edge(u, v, edge_type)

    def remove_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: str,
    ) -> None:
        """
        Remove an edge between u and v.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str
            One of the values in `SUPPORTED_EDGE_TYPES`. The edge of this type
            between `u` and `v` is removed. This argument is required.

        Returns
        -------
        None

        See Also
        --------
        remove_edges_from : Remove multiple edges at once.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> G.remove_edge("A", "B", "->")
        >>> G.get_edges(data=True)
        [('B', 'C', '->'), ('C', 'D', '--')]

        """
        self._validate_edges(edge_list=[(u, v, edge_type)])

        markers = self._to_markers(edge=(u, v, edge_type))
        # get_edge_data returns None when the pair has no edge (or a node is absent); treat as "not in graph".
        for key, data in (self.get_edge_data(u, v) or {}).items():
            if data[u] == markers[u] and data[v] == markers[v]:
                super().remove_edge(u, v, key=key)
                return
        raise ValueError(f"Edge ({u}, {v}, {edge_type}) not in graph.")

    def remove_edges_from(
        self,
        edge_list: Iterable[tuple[Hashable, Hashable, str]],
    ) -> None:
        """
        Remove all the edges in edge_list.

        Parameters
        ----------
        edge_list : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...].

        Returns
        -------
        None

        See Also
        --------
        remove_edge : Remove a single edge.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> remove_edges = [("C", "B", "<-"), ("C", "D", "--")]
        >>> G.remove_edges_from(edge_list=remove_edges)
        >>> G.get_edges(data=True)
        [('A', 'B', '->')]

        """
        edge_list = list(edge_list)  # materialize: _validate_edges below would exhaust a generator
        self._validate_edges(edge_list=edge_list)
        for u, v, edge_type in edge_list:
            self.remove_edge(u, v, edge_type)

    def get_edges(self, data: bool = True, edge_types: Iterable[str] | None = None) -> list[tuple[Any, ...]]:
        """
        Retrieve all edges of the specified edge types.

        Parameters
        ----------
        data : bool, optional (default=True)
            If True, each edge is returned as ``(u, v, edge_type)``; if False, as ``(u, v)``.

        edge_types : iterable of str, optional (default: all)
            If given, only edges whose (canonical) type is in `edge_types` are returned. The filter is matched on the
            canonical type, so ``{"->"}`` selects both ``"->"`` and ``"<-"`` edges.

        Returns
        -------
        list
            A list of edge tuples: ``(u, v, edge_type)`` if ``data`` else ``(u, v)``.

        See Also
        --------
        get_edge_type : Edge type(s) between a specific pair of nodes.
        get_unique_edge_types : The distinct (canonical) edge types present in the graph.

        Examples
        --------
        >>> edges = [("A", "B", "->"), ("A", "B", "<>"), ("C", "B", "<-")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> sorted(G.get_edges(data=True))
        [('A', 'B', '->'), ('A', 'B', '<>'), ('B', 'C', '->')]
        >>> sorted(G.get_edges(data=True, edge_types={"->"}))
        [('A', 'B', '->'), ('B', 'C', '->')]

        """
        canonical = {"<-": "->", "-o": "o-", "<o": "o>"}
        if edge_types is not None:
            edge_types = {canonical.get(edge_type, edge_type) for edge_type in edge_types}

        edges = []
        for u, v, markers in super().edges(data=True):
            edge_type = self._to_edge_type(u, v, markers)
            if edge_type in canonical:
                u, v, edge_type = v, u, canonical[edge_type]
            if edge_types is None or edge_type in edge_types:
                edges.append((u, v, edge_type) if data else (u, v))
        return edges

    def get_unique_edge_types(self) -> set[str]:
        """
        Returns the set of distinct edge types present in the graph.

        Orientation-reversed views of the same edge type are treated as one: ``"<-"`` is reported as
        ``"->"``, ``"-o"`` as ``"o-"``, and ``"<o"`` as ``"o>"`` (they differ only in which endpoint
        is listed first). The result is therefore a subset of the six edge types
        ``{"--", "->", "<>", "o-", "o>", "oo"}``.

        Returns
        -------
        edge_types : set of str
            The distinct edge types currently in the graph, each in its canonical orientation.

        See Also
        --------
        get_edges : All edges in the graph.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("A", "B", "->"), ("C", "B", "<-"), ("C", "D", "<>")])
        >>> G.get_unique_edge_types() == {"->", "<>"}
        True

        """
        return {edge_type for _, _, edge_type in self.get_edges(data=True)}

    def get_edge_type(self, u: Hashable, v: Hashable) -> set[str]:
        """
        Return the set of edge types connecting `u` and `v`.

        Each type is read from ``u``'s endpoint, so it is *not* canonicalized: a stored ``v -> u`` edge
        is reported as ``"<-"`` (call ``get_edge_type(v, u)`` for the mirror orientation). A multigraph
        may join a pair by several edges at once (e.g. an ADMG holding both ``"->"`` and ``"<>"``), so
        the result is a set.

        Parameters
        ----------
        u : Hashable
            The first endpoint of the edge.
        v : Hashable
            The second endpoint of the edge.

        Returns
        -------
        edge_types : set of str
            The edge type(s) between `u` and `v`, each oriented from `u` to `v`.

        Raises
        ------
        ValueError
            If there is no edge between `u` and `v`.

        See Also
        --------
        get_edges : All edges in the graph.
        has_edge : Whether a (typed) edge exists between two nodes.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph()
        >>> graph.add_edge("A", "B", "->")
        >>> graph.add_edge("A", "B", "<>")
        >>> graph.add_edge("B", "C", "--")
        >>> graph.get_edge_type("A", "B") == {"->", "<>"}
        True
        >>> graph.get_edge_type("B", "C")
        {'--'}

        """
        if not self.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) not in graph.")

        # `self[u][v]` maps each parallel edge's key to its markers dict; read each type from `u`.
        return {self._to_edge_type(u, v, markers) for markers in self[u][v].values()}

    def has_edge(self, u, v, edge_type=None):
        """
        Returns True if the graph has an edge between nodes u and v.

        Parameters
        ----------
        u : Hashable
            The source node of the edge.
        v : Hashable
            The target node of the edge.
        edge_type : str
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        Returns
        -------
        bool

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph()
        >>> graph.add_edge("A", "B", "->")
        >>> graph.has_edge("A", "B")
        True
        >>> graph.has_edge("A", "B", "->")
        True
        >>> graph.has_edge("A", "B", "--")
        False

        """
        if not super().has_edge(u, v):
            return False

        if edge_type is None:
            return True

        if edge_type not in self.SUPPORTED_EDGE_TYPES:
            raise ValueError(f"Types must be one of {self.SUPPORTED_EDGE_TYPES}.")
        return edge_type in self.get_edge_type(u, v)

    def replace_edge(
        self,
        u: Hashable,
        v: Hashable,
        old_type: str = "--",
        new_type: str = "->",
    ) -> None:
        """
        Replaces the type of an existing edge between two nodes with a new type.

        Parameters
        ----------
        u : Hashable
            The source node of the edge.
        v : Hashable
            The target node of the edge.
        old_type : str, optional (default="--")
            The current type of the edge that needs to be replaced.
        new_type : str, optional (default="->")
            The new edge type to be assigned.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph()
        >>> graph.add_edge("A", "B", "--")
        >>> graph.replace_edge("A", "B", old_type="--", new_type="->")
        >>> graph.has_edge("A", "B", "->")
        True
        >>> graph.has_edge("A", "B", "--")
        False

        """
        if (old_type not in self.SUPPORTED_EDGE_TYPES) or (new_type not in self.SUPPORTED_EDGE_TYPES):
            raise ValueError(
                f"Unsupported edge type(s) provided. "
                f"Got old_type='{old_type}', new_type='{new_type}'. "
                f"Supported types are: {self.SUPPORTED_EDGE_TYPES}."
            )
        if not self.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) not in graph.")

        self.remove_edge(u, v, old_type)
        try:
            self.add_edge(u, v, new_type)
        except ValueError:
            # The new edge was rejected (e.g. it would close a directed cycle); restore the old
            # edge so a failed replace leaves the graph unchanged.
            self.add_edge(u, v, old_type)
            raise

    def get_marker(self, u: Hashable, v: Hashable) -> str:
        """
        Return the endpoint marker at `v` on the edge between `u` and `v`.

        The marker is read at the **second** argument's endpoint: for an edge ``A o> B``,
        ``get_marker("A", "B")`` is ``">"`` and ``get_marker("B", "A")`` is ``"o"``.

        Parameters
        ----------
        u, v : Hashable
            The endpoints of the edge; the marker at `v` is returned.

        Returns
        -------
        marker : str
            One of ``"-"`` (tail), ``">"`` (arrowhead), or ``"o"`` (circle).

        Raises
        ------
        ValueError
            If there is no edge between `u` and `v`, or the pair is joined by parallel edges
            (use ``get_edge_type`` for multi-edge pairs).

        See Also
        --------
        set_marker : Re-orient a single endpoint.
        get_edge_type : The full edge type(s) between two nodes.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("A", "B", "o>")])
        >>> graph.get_marker("A", "B")
        '>'
        >>> graph.get_marker("B", "A")
        'o'

        """
        if not self.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) not in graph.")

        edges = list(self.get_edge_data(u, v).values())
        if len(edges) > 1:
            raise ValueError(f"There are parallel edges between {u} and {v}; use get_edge_type instead.")
        return edges[0][v]

    def set_marker(self, u: Hashable, v: Hashable, marker: str) -> None:
        """
        Set the endpoint marker at `v` on the edge between `u` and `v`, leaving the marker at
        `u` untouched.

        This is the single-endpoint orientation move of FCI-style algorithms: e.g. on an edge
        ``A oo B``, ``set_marker("A", "B", ">")`` orients it to ``A o> B``. The marker is set at
        the **second** argument's endpoint. The resulting edge type must be supported by the
        class and must not close a directed cycle; a failed call leaves the edge unchanged.

        Parameters
        ----------
        u, v : Hashable
            The endpoints of the edge; the marker at `v` is set.

        marker : str
            One of ``"-"`` (tail), ``">"`` (arrowhead), or ``"o"`` (circle).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `marker` is invalid; if there is no edge between `u` and `v` or the pair is
            joined by parallel edges; if the resulting edge type is not in
            ``SUPPORTED_EDGE_TYPES``; or if the re-orientation would create a directed cycle.

        See Also
        --------
        get_marker : Read a single endpoint marker.
        replace_edge : Replace the full edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("A", "B", "oo")])
        >>> graph.set_marker("A", "B", ">")
        >>> graph.get_edges(data=True)
        [('A', 'B', 'o>')]
        >>> graph.set_marker("B", "A", "-")
        >>> graph.get_edges(data=True)
        [('A', 'B', '->')]

        """
        if marker not in {"-", ">", "o"}:
            raise ValueError(f"marker must be one of '-', '>', 'o'. Got {marker!r}.")

        u_marker = self.get_marker(v, u)
        v_marker = self.get_marker(u, v)
        if v_marker == marker:
            return

        old_type = self._to_edge_type(u, v, {u: u_marker, v: v_marker})
        new_type = self._to_edge_type(u, v, {u: u_marker, v: marker})
        self.replace_edge(u, v, old_type=old_type, new_type=new_type)

    def copy(self):
        """
        Returns a deep copy of the graph object.

        Returns
        -------
        graph: graph object
            A copy of the graph object.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>")], latents={"B"})
        >>> G_copy = G.copy()
        >>> G_copy == G
        True
        >>> sorted(G_copy.get_edges(data=True))
        [('A', 'B', '->'), ('B', 'C', '<>')]
        >>> G_copy.latents
        {'B'}
        >>> G_copy.add_edge("C", "D", "->")
        >>> G.has_edge("C", "D")
        False

        """
        return self.get_subgraph(self.nodes())

    def get_subgraph(self, nodes: Iterable[Hashable] | None = None, edge_types: Iterable[str] | None = None):
        """
        Returns a subgraph as an independent copy of the same class, filtered by nodes and/or edge types.

        The returned graph keeps the requested `nodes` and every edge of edge type specified in `edge_types`. The edge
        types and node roles (exposures, outcomes, latents, ...) are preserved in the subgraph.

        Parameters
        ----------
        nodes : iterable of Hashable, optional (default: all nodes)
            The nodes to induce the subgraph on; all must be present in the graph. An edge is kept only
            if both of its endpoints are kept.

        edge_types : iterable of str, optional (default: all edge types)
            The edge types to keep. Orientation-reversed forms are matched by their canonical type, so
            ``{"->"}`` keeps both ``"->"`` and ``"<-"`` edges (see ``get_unique_edge_types``).

        Returns
        -------
        subgraph : graph object
            A new graph of the same class as `self`, restricted to the selected nodes and edge types.

        Raises
        ------
        ValueError
            If any of the requested nodes is not in the graph.

        See Also
        --------
        copy : Deep copy of the whole graph.
        get_ancestral_graph : Subgraph induced by a set of nodes and their ancestors.
        get_unique_edge_types : The distinct (canonical) edge types present in the graph.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "--")])
        >>> sorted(G.get_subgraph(["A", "B", "C"]).get_edges(data=True))
        [('A', 'B', '->'), ('B', 'C', '<>')]
        >>> sorted(G.get_subgraph(edge_types={"->"}).get_edges(data=True))
        [('A', 'B', '->')]
        """
        nodes = set(self.nodes()) if nodes is None else set(nodes)
        if missing := (nodes - set(self.nodes())):
            raise ValueError(f"Nodes {sorted(missing)} not in graph.")

        subgraph = self.__class__()
        subgraph.add_nodes_from(node for node in self.nodes() if node in nodes)
        for u, v, edge_type in self.get_edges(data=True, edge_types=edge_types):
            if u in nodes and v in nodes:
                subgraph.add_edge(u, v, edge_type=edge_type)

        for role, variables in self.get_role_dict().items():
            retained = [node for node in variables if node in nodes]
            if retained:
                subgraph.with_role(role=role, variables=retained, inplace=True)
        return subgraph

    def get_skeleton(self) -> nx.Graph:
        """
        Returns the undirected skeleton of the graph as a ``networkx.Graph``.

        The skeleton keeps the same nodes and has one undirected edge for every adjacent pair of
        nodes; all edge orientations and types are discarded, and coincident edges between a pair
        collapse to a single edge.

        Returns
        -------
        skeleton : networkx.Graph
            An undirected graph capturing only the adjacency structure.

        See Also
        --------
        get_directed_graph : The directed counterpart (``"->"`` edges only).
        get_subgraph : Class-preserving node/edge-type-filtered subgraph.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>")])
        >>> sorted(G.get_skeleton().edges())
        [('A', 'B'), ('B', 'C')]

        """
        skeleton = nx.Graph()
        skeleton.add_nodes_from(self.nodes())
        skeleton.add_edges_from(self.edges())
        return skeleton

    def get_directed_graph(self) -> nx.DiGraph:
        """
        Returns the directed projection of the graph as a ``networkx.DiGraph``.

        Keeps the same nodes and one arc ``u -> v`` for every directed (``"->"``) edge, taken in its
        canonical orientation; undirected, bidirected, and circle edges are dropped. This is the
        directed counterpart of ``get_skeleton``, meant for feeding networkx's directed algorithms
        (e.g. ``topological_sort``, ``is_directed_acyclic_graph``, ``has_path``).

        Returns
        -------
        directed_graph : networkx.DiGraph
            A directed graph over all nodes, containing only the ``"->"`` edges.

        See Also
        --------
        get_skeleton : The undirected counterpart (adjacency only).

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("A", "B", "->"), ("C", "B", "<-"), ("C", "D", "--")])
        >>> sorted(G.get_directed_graph().edges())
        [('A', 'B'), ('B', 'C')]

        """
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(self.nodes())
        directed_graph.add_edges_from((u, v) for u, v, _ in self.get_edges(edge_types={"->"}))
        return directed_graph

    def do(self, nodes: Hashable | Iterable[Hashable], inplace: bool = False):
        """
        Returns the graph after applying the do-operator to `nodes`.

        Intervening on a variable severs it from all of its causes, so every edge with an arrowhead
        at a do-variable is removed. Assuming a node `X` in `nodes`, the following types of edges
        will be removed:
            - ``Y -> X``
            - ``Y <-> X``
            - ``Y o> X``

        And the following edge types are kept:
            - ``X -> Y``
            - ``X -- Y``
            - ``X -o Y``

        The following throw a `ValueError` as the edge mark at `X` is ambiguous.
            - ``X o- Y``
            - ``X o> Y``
            - ``X oo Y``

        Parameters
        ----------
        nodes : Hashable or iterable of Hashable
            The variable(s) to intervene on; all must be present in the graph. A ``str``, ``int``
            or ``tuple`` is one node; a list, set or frozenset is a collection.

        inplace : bool (default: False)
            If True, modify and return this graph; otherwise return a modified copy.

        Returns
        -------
        graph : graph object
            The post-intervention graph (same class as `self`).

        Raises
        ------
        ValueError
            If any node is not present in the graph, or an edge has a circle endpoint at a
            do-variable (so the intervention is undetermined).

        References
        ----------
        - :cite:p:`pearl_2009` (page 70).

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("X", "A", "->"), ("A", "Y", "->")])
        >>> sorted(G.do("A").get_edges(data=True))
        [('A', 'Y', '->')]

        """
        graph = self if inplace else self.copy()
        nodes = list(nodes) if isinstance(nodes, (list, set, frozenset)) else [nodes]
        if missing := (set(nodes) - set(self.nodes())):
            raise ValueError(f"Nodes not found in the model: {missing}")

        arrowhead_types = {"<-", "<>", "<o"} & self.SUPPORTED_EDGE_TYPES
        circle_types = {"o-", "o>", "oo"} & self.SUPPORTED_EDGE_TYPES

        # Step 1: Check if any of the edges has `o` marker on `node`.
        for node in nodes:
            if self.get_neighbors(node, circle_types):
                raise ValueError(
                    f"do({node!r}) is undetermined: an edge has a circle endpoint at {node!r}, "
                    "so it cannot be classified as incoming or not."
                )

        # Step 2: Remove every incoming (arrowhead-at-`node`) edge.
        for node in nodes:
            for edge_type in arrowhead_types:
                for neighbor in self.get_neighbors(node, edge_type):
                    graph.remove_edge(node, neighbor, edge_type)
        return graph

    def get_roots(self) -> set[Hashable]:
        """
        Returns the set of root (source) nodes.

        A node is a root when every incident edge has a **tail** at it, i.e., it has no incoming
        arrowhead (`->`/`<>` pointing in) and no ambiguous circle endpoint; outgoing directed (`->`)
        and undirected (`--`) edges are allowed. Isolated nodes are considered as roots.
        On a purely directed graph this is exactly the set of nodes with in-degree 0.

        Returns
        -------
        roots : set
            The source nodes.

        See Also
        --------
        get_leaves : The sink counterpart.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("X", "A", "->"), ("A", "Y", "->")])
        >>> G.get_roots()
        {'X'}

        """
        incoming_or_ambiguous = {"<-", "<>", "<o", "o-", "o>", "oo"} & self.SUPPORTED_EDGE_TYPES
        return {node for node in self.nodes() if not self.get_neighbors(node, incoming_or_ambiguous)}

    def get_leaves(self) -> set[Hashable]:
        """
        Returns the set of leaf (sink) nodes.

        A node is a leaf when every incident edge has an **arrowhead** at it -- so it has no outgoing
        tail (`->`/`--` leaving it) and no ambiguous circle endpoint; incoming directed (`->`) and
        bidirected (`<>`) edges are allowed (a sink may have latent parents). Isolated nodes are
        leaves. On a purely directed graph this is exactly the set of nodes with out-degree 0.

        Returns
        -------
        leaves : set
            The sink nodes.

        See Also
        --------
        get_roots : The source counterpart.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("X", "A", "->"), ("A", "Y", "->")])
        >>> G.get_leaves()
        {'Y'}

        """
        outgoing_or_ambiguous = {"--", "->", "-o", "o-", "o>", "oo"} & self.SUPPORTED_EDGE_TYPES
        return {node for node in self.nodes() if not self.get_neighbors(node, outgoing_or_ambiguous)}

    def get_neighbors(self, node: Hashable, edge_types: str | Iterable[str] | None = None) -> set[Hashable]:
        """
        Returns a set of neighbors nodes in the graph that are connected through one of the specified edge types.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers. Nodes must be hashable (and not None) Python objects.

        edge_types : str or iterable of str, optional (default: None)
            A single edge type or a collection of them, each from ``SUPPORTED_EDGE_TYPES`` and read from ``node``'s
            endpoint. A neighbor is returned if connected by an edge of any of these types; ``None`` returns all
            neighbors. Raises ``ValueError`` on an unsupported type.

        Returns
        -------
        nodes : set
            Set of neighbors nodes.

        See Also
        --------
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_descendants : All descendants of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_neighbors("B", "->")))
        ['C']
        >>> print(sorted(G.get_neighbors("B", "<-")))
        ['A']
        >>> print(sorted(G.get_neighbors("B", {"->", "<-"})))
        ['A', 'C']

        """
        if node not in self.nodes():
            raise ValueError(f"Node {node} not in graph.")

        if edge_types is None:
            return set(self.neighbors(node))

        requested = {edge_types} if isinstance(edge_types, str) else set(edge_types)
        if unsupported := (requested - self.SUPPORTED_EDGE_TYPES):
            raise ValueError(f"Types must be one of {self.SUPPORTED_EDGE_TYPES}. Got {unsupported}.")

        neighbors = set()
        for neighbor in self.neighbors(node):
            connecting_types = {
                self._to_edge_type(node, neighbor, markers) for markers in self.get_edge_data(node, neighbor).values()
            }
            if connecting_types & requested:
                neighbors.add(neighbor)
        return neighbors

    def get_parents(self, nodes: Hashable | Iterable[Hashable]) -> set[Hashable]:
        """
        Returns the parents of a node, or the union of parents over an iterable of nodes.

        Parameters
        ----------
        nodes : Hashable or iterable of Hashable
            A single node, or a list/set of nodes whose parents are unioned. A ``str``, ``int`` or
            ``tuple`` is treated as one node; a list, set or frozenset is treated as a collection.

        Returns
        -------
        nodes : set
            Set of parent nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_children : Child nodes (via outgoing directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_descendants : All descendants of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("A", "C", "->"), ("B", "C", "->"), ("C", "D", "->")])
        >>> sorted(G.get_parents("C"))
        ['A', 'B']
        >>> sorted(G.get_parents(["C", "D"]))
        ['A', 'B', 'C']

        """
        nodes = list(nodes) if isinstance(nodes, (list, set, frozenset)) else [nodes]
        return set().union(*(self.get_neighbors(node=n, edge_types="<-") for n in nodes))

    def get_children(self, nodes: Hashable | Iterable[Hashable]) -> set[Hashable]:
        """
        Returns the children of a node, or the union of children over an iterable of nodes.

        Parameters
        ----------
        nodes : Hashable or iterable of Hashable
            A single node, or a list/set of nodes whose children are unioned. A ``str``, ``int`` or
            ``tuple`` is treated as one node; a list, set or frozenset is treated as a collection.

        Returns
        -------
        nodes : set
            Set of child nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_descendants : All descendants of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("A", "B", "->"), ("A", "C", "->"), ("C", "D", "->")])
        >>> sorted(G.get_children("A"))
        ['B', 'C']
        >>> sorted(G.get_children(["A", "C"]))
        ['B', 'C', 'D']

        """
        nodes = list(nodes) if isinstance(nodes, (list, set, frozenset)) else [nodes]
        return set().union(*(self.get_neighbors(node=n, edge_types="->") for n in nodes))

    def get_spouses(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of spouses nodes in the graph.

        The spouses of a ``node`` are nodes that are connected to it through bi-directed edges. Note that this
        definition is different from the definition commonly used for DAGs where other parents of the ``node``'s
        children are the spouses.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of spouses nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_descendants : All descendants of a node (including the node itself).
        get_ancestors : All ancestors of a node (including the node itself).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "<>")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_spouses("B")))
        ['C']
        >>> print(sorted(G.get_spouses("C")))
        ['B']

        """
        return self.get_neighbors(node=node, edge_types="<>")

    def get_ancestors(self, nodes: Hashable | Iterable[Hashable]) -> set[Hashable]:
        """
        Returns a set of ancestors nodes in the graph.

        Parameters
        ----------
        nodes : Hashable or iterable of Hashable
            A single node, or a list/set of nodes; ancestors are unioned over the collection. A
            ``str``, ``int`` or ``tuple`` is one node; a list, set or frozenset is a collection.

        Returns
        -------
        nodes : set
            Set of ancestor nodes (including the input node(s)).

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_descendants : All descendants of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_ancestors("C")))
        ['A', 'B', 'C']
        >>> print(sorted(G.get_ancestors("B")))
        ['A', 'B']

        """
        nodes = list(nodes) if isinstance(nodes, (list, set, frozenset)) else [nodes]
        for n in nodes:
            if n not in self.nodes():
                raise ValueError(f"Node {n} not in graph.")

        ancestors = set()
        queue = deque(nodes)

        while queue:
            current = queue.popleft()
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self.get_parents(current))
        return ancestors

    def get_descendants(self, nodes: Hashable | Iterable[Hashable]) -> set[Hashable]:
        """
        Returns a set of descendants nodes in the graph.

        Parameters
        ----------
        nodes : Hashable or iterable of Hashable
            A single node, or a list/set of nodes; descendants are unioned over the collection. A
            ``str``, ``int`` or ``tuple`` is one node; a list, set or frozenset is a collection.

        Returns
        -------
        nodes : set
            Set of descendants nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_descendants("A")))
        ['A', 'B', 'C']
        >>> print(sorted(G.get_descendants(["B", "C"])))
        ['B', 'C']

        """
        nodes = list(nodes) if isinstance(nodes, (list, set, frozenset)) else [nodes]
        for n in nodes:
            if n not in self.nodes():
                raise ValueError(f"Node {n} not in graph.")

        descendants = set()
        queue = deque(nodes)

        while queue:
            current = queue.popleft()
            if current not in descendants:
                descendants.add(current)
                queue.extend(self.get_children(current))
        return descendants

    def get_reachable_nodes(self, node: Hashable, edge_types: str | Iterable[str] | None = None) -> set[Hashable]:
        """
        Returns a set of reachable nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_types : str or iterable of str, optional (default: None)
            A single edge type or a collection of them; the traversal follows edges of any of these
            types (read from each node's endpoint). ``None`` follows any edge type.

        Returns
        -------
        nodes : set
            Set of reachable nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_descendants : All descendants of a node (including the node itself).

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [
        ...     ("A", "B", "->"),
        ...     ("B", "C", "->"),
        ...     ("C", "D", "--"),
        ...     ("D", "F", "<>"),
        ... ]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_reachable_nodes("A", "->")))
        ['A', 'B', 'C']
        >>> print(sorted(G.get_reachable_nodes("C", "--")))
        ['C', 'D']
        >>> print(sorted(G.get_reachable_nodes("D", "<>")))
        ['D', 'F']

        """
        if node not in self.nodes():
            raise ValueError(f"Node {node} not in graph.")

        reachable = set()
        queue = deque([node])

        while queue:
            current = queue.popleft()
            if current not in reachable:
                reachable.add(current)
                queue.extend(self.get_neighbors(current, edge_types))
        return reachable

    def get_district(self, nodes: Hashable | Iterable[Hashable]) -> set[Hashable]:
        """
        Returns the district of `nodes`: the maximal set of nodes connected to them through
        bidirected (``"<>"``) edges, including the input node(s) themselves.

        The district -- a *c-component* / confounded component -- groups variables that share latent
        confounding, directly or transitively through a chain of bidirected edges. A node with no
        bidirected edges (e.g. in a DAG or PDAG) is its own district.

        Parameters
        ----------
        nodes : Hashable or iterable of Hashable
            A single node, or a list/set of nodes; districts are unioned over the collection. A
            ``str``, ``int`` or ``tuple`` is one node; a list, set or frozenset is a collection.

        Returns
        -------
        nodes : set
            All nodes in the same bidirected-connected component(s) as `nodes`, including `nodes`.

        See Also
        --------
        get_spouses : Nodes joined to a node by a single bidirected edge.
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base import ADMG
        >>> admg = ADMG(edge_list=[("X", "Y", "->"), ("X", "Z", "<>")])
        >>> sorted(admg.get_district("X"))
        ['X', 'Z']
        >>> admg.get_district("Y")
        {'Y'}

        """
        nodes = list(nodes) if isinstance(nodes, (list, set, frozenset)) else [nodes]
        return set().union(*(self.get_reachable_nodes(node=n, edge_types="<>") for n in nodes))

    def to_adjacency(self, encoding: str = "edge_type", nodelist=None) -> pd.DataFrame:
        """
        Return the adjacency matrix of the graph as a ``pandas.DataFrame``.

        The value in each cell depends on ``encoding``, chosen to match the conventions of common ecosystem tools.

        Parameters
        ----------
        encoding : str (default="edge_type")
            One of:

            - ``"edge_type"`` : native, human-readable. ``M.loc[u, v]`` is the edge type oriented from ``u`` to ``v``
              (e.g. ``"->"``, ``"<-"``, ``"<>"``, ``"--"``, ``"o>"``), or ``0`` for no edge. When a pair is joined by
              more than one edge (e.g. an ADMG with a directed *and* a bidirected edge), the cell holds a tuple of edge
              types, aligned per edge with the mirror cell ``M.loc[v, u]``.
            - ``"binary"`` : ``0/1`` directed adjacency. ``M.loc[u, v] = 1`` iff there is a directed edge ``u`` -> ``v``
              (so directed edges are asymmetric); an undirected edge is symmetric (both ``1``). This is the
              DAG/CPDAG-style numeric matrix and is only supported for DAG/PDAG-style graphs, i.e. those whose
              ``SUPPORTED_EDGE_TYPES`` is a subset of ``{"->", "<-", "--"}``. ``"bnlearn"`` is accepted as an alias.
            - ``"causal-learn"`` : integer codes interoperable with causal-learn's ``GeneralGraph``. ``M.loc[u, v]`` is
              the mark at ``u`` (the row endpoint): ``0`` no edge, ``-1`` tail, ``1`` arrowhead, ``2`` circle, ``4``
              tail-and-arrowhead, ``5`` arrowhead-and-arrowhead (the last two for coincident directed + bidirected
              edges).
            - ``"pcalg"`` : integer codes of pcalg's ``amat.pag``. ``M.loc[u, v]`` is the mark at ``v`` (the column
              endpoint): ``0`` no edge, ``1`` circle, ``2`` arrowhead, ``3`` tail.

        nodelist : list, optional (default=None)
            The row/column ordering. If ``None``, the sorted node list is used.

        Returns
        -------
        pandas.DataFrame
            A square adjacency matrix indexed by the nodes.

        Raises
        ------
        ValueError
            If ``encoding`` is unknown; if ``encoding="binary"`` is requested on a graph that supports edge types
            beyond ``{"->", "<-", "--"}`` (i.e. anything other than a DAG/PDAG); or if the graph contains an edge the
            chosen encoding cannot represent (coincident edges for ``"binary"``/``"pcalg"``).

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>")])
        >>> G.to_adjacency().loc["A", "B"]
        '->'
        """
        # Step 0: Validate the inputs
        valid = {"edge_type", "binary", "bnlearn", "causal-learn", "pcalg"}
        if encoding not in valid:
            raise ValueError(f"encoding must be one of {sorted(valid)}. Got {encoding!r}.")
        if encoding == "bnlearn":
            # binary and bnlearn are equivalent
            encoding = "binary"
        if encoding == "binary" and not self.SUPPORTED_EDGE_TYPES <= {"->", "<-", "--"}:
            raise ValueError(
                "binary encoding is only supported for DAG and PDAG (directed/undirected edges); "
                f"{self.__class__.__name__} supports {sorted(self.SUPPORTED_EDGE_TYPES)}."
            )

        # Step 1: Define data structure and helper method required for constructing the adjacency matrix.
        nodes = sorted(self.nodes()) if nodelist is None else list(nodelist)
        adj = pd.DataFrame(0, index=nodes, columns=nodes, dtype=object if encoding == "edge_type" else int)

        pcalg_code = {("o",): 1, (">",): 2, ("-",): 3}
        causal_learn_code = {("-",): -1, (">",): 1, ("o",): 2, ("-", ">"): 4, (">", ">"): 5}

        def endpoint_code(code_map, marks, u, v):
            """
            Used for converting edge_type adjacency matrix to pcalg or causal_learn conventions.
            """
            key = tuple(sorted(marks))
            if key not in code_map:
                raise ValueError(f"{encoding} adjacency cannot represent the marks {key} between {u} and {v}.")
            return code_map[key]

        # Step 2: Iterate over edges and add the appropriate symbol in `adj` matrix.
        for u, v in dict.fromkeys(self.get_edges(data=False)):
            edges = list(self.get_edge_data(u, v).values())
            marks_u = [data[u] for data in edges]
            marks_v = [data[v] for data in edges]

            # Step 2.1: If encoding = "edge_type", add the edge type at both (u, v) and (v, u). Use a tuple if there are
            #           multiple edges between u and v.
            if encoding == "edge_type":
                if len(edges) == 1:
                    adj.at[u, v] = self._to_edge_type(u, v, edges[0])
                    adj.at[v, u] = self._to_edge_type(v, u, edges[0])
                else:
                    order = sorted(range(len(edges)), key=lambda i: (marks_u[i], marks_v[i]))
                    adj.at[u, v] = tuple(self._to_edge_type(u, v, edges[i]) for i in order)
                    adj.at[v, u] = tuple(self._to_edge_type(v, u, edges[i]) for i in order)

            # Step 2.2: If encoding = "binary", add directed and undirected edges.
            elif encoding == "binary":
                if len(edges) > 1:
                    raise ValueError(
                        f"binary adjacency cannot represent the {len(edges)} coincident edges between {u} and {v}."
                    )
                edge_type = self._to_edge_type(u, v, edges[0])
                if edge_type == "->":
                    adj.at[u, v] = 1
                elif edge_type == "<-":
                    adj.at[v, u] = 1
                else:
                    adj.at[u, v] = 1
                    adj.at[v, u] = 1

            # Step 2.3: If encoding = "pcalg", convert the edge type using `pcalg_code` dict.
            elif encoding == "pcalg":
                adj.at[u, v] = endpoint_code(pcalg_code, marks_v, u, v)
                adj.at[v, u] = endpoint_code(pcalg_code, marks_u, u, v)

            # Step 2.4: If encoding = "causal-learn", convert the edge type using `causal_learn_code` dict.
            else:
                adj.at[u, v] = endpoint_code(causal_learn_code, marks_u, u, v)
                adj.at[v, u] = endpoint_code(causal_learn_code, marks_v, u, v)

        # Step 3: Return the dataframe.
        return adj

    @classmethod
    def from_adjacency(cls, adj: pd.DataFrame, encoding: str = "edge_type"):
        """
        Construct a graph from an adjacency matrix -- the inverse of :meth:`to_adjacency`.

        The graph is created as an instance of the class this is called on, so the class's
        ``SUPPORTED_EDGE_TYPES`` (and acyclicity of directed edges) are enforced while decoding.
        Node roles are not part of an adjacency matrix; assign them on the returned graph if
        needed. For every encoding ``cls.from_adjacency(g.to_adjacency(e), encoding=e) == g``
        holds (up to roles).

        Parameters
        ----------
        adj : pandas.DataFrame
            A square adjacency matrix with identical index and columns, in one of the encodings
            produced by :meth:`to_adjacency`. For a numpy array ``arr`` from another library,
            wrap it first: ``pd.DataFrame(arr, index=nodes, columns=nodes)``.

        encoding : str (default="edge_type")
            One of ``"edge_type"``, ``"binary"`` (alias ``"bnlearn"``), ``"causal-learn"``,
            ``"pcalg"``; see :meth:`to_adjacency` for the cell conventions of each.

        Returns
        -------
        graph : instance of `cls`
            The decoded graph over the matrix's index as nodes.

        Raises
        ------
        ValueError
            If ``encoding`` is unknown; if ``encoding="binary"`` is requested for a class whose
            edge types exceed ``{"->", "<-", "--"}``; if index and columns differ; if a cell and
            its mirror cell are inconsistent; or if a decoded edge is not representable by `cls`.

        See Also
        --------
        to_adjacency : The encoding counterpart.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(edge_list=[("A", "B", "->"), ("B", "C", "--")])
        >>> PDAG.from_adjacency(pdag.to_adjacency()) == pdag
        True
        >>> adj = pdag.to_adjacency(encoding="binary")
        >>> sorted(PDAG.from_adjacency(adj, encoding="binary").get_edges(data=True))
        [('A', 'B', '->'), ('B', 'C', '--')]

        """
        # Step 0: Validate the inputs.
        valid = {"edge_type", "binary", "bnlearn", "causal-learn", "pcalg"}
        if encoding not in valid:
            raise ValueError(f"encoding must be one of {sorted(valid)}. Got {encoding!r}.")
        if encoding == "bnlearn":
            encoding = "binary"
        if encoding == "binary" and not cls.SUPPORTED_EDGE_TYPES <= {"->", "<-", "--"}:
            raise ValueError(
                "binary encoding is only supported for DAG and PDAG (directed/undirected edges); "
                f"{cls.__name__} supports {sorted(cls.SUPPORTED_EDGE_TYPES)}."
            )
        if list(adj.index) != list(adj.columns):
            raise ValueError("adj must be a square DataFrame with identical index and columns.")
        for node in adj.index:
            if adj.at[node, node] != 0:
                raise ValueError(f"Inconsistent matrix: the diagonal cell ({node}, {node}) must be 0.")

        # Step 1: Create the graph and the decoding tables (inverses of the `to_adjacency` codes).
        graph = cls()
        graph.add_nodes_from(adj.index)

        pcalg_mark = {1: "o", 2: ">", 3: "-"}
        causal_learn_mark = {-1: "-", 1: ">", 2: "o"}
        causal_learn_coincident = {(4, 5): ("->", "<>"), (5, 4): ("<-", "<>")}

        # Step 2: Decode each node pair from its cell and the mirror cell.
        for u, v in combinations(adj.index, 2):
            cell_uv, cell_vu = adj.at[u, v], adj.at[v, u]

            # Step 2.1: binary cells are 0/1; asymmetric = directed (so an asymmetric zero is
            #           valid here, unlike the mark-based encodings below).
            if encoding == "binary":
                if cell_uv not in (0, 1) or cell_vu not in (0, 1):
                    raise ValueError(f"binary cells must be 0 or 1. Got ({cell_uv!r}, {cell_vu!r}) for ({u}, {v}).")
                if cell_uv and cell_vu:
                    graph.add_edge(u, v, "--")
                elif cell_uv:
                    graph.add_edge(u, v, "->")
                elif cell_vu:
                    graph.add_edge(v, u, "->")
                continue

            # The mark-based encodings store an edge in both cells: zero in one but not the other
            # is inconsistent.
            if (cell_uv == 0) != (cell_vu == 0):
                raise ValueError(f"Inconsistent cells for ({u}, {v}): {cell_uv!r} vs {cell_vu!r}.")
            if cell_uv == 0:
                continue

            # Step 2.2: edge_type cells hold the type oriented u -> v (a tuple for parallel
            #           edges); the mirror cell must hold the reversed orientations.
            if encoding == "edge_type":
                types_uv = (cell_uv,) if isinstance(cell_uv, str) else tuple(cell_uv)
                types_vu = (cell_vu,) if isinstance(cell_vu, str) else tuple(cell_vu)
                graph._validate_edges([(u, v, edge_type) for edge_type in types_uv])
                mirrored = tuple(graph._to_edge_type(v, u, graph._to_markers((u, v, t))) for t in types_uv)
                if sorted(mirrored) != sorted(types_vu):
                    raise ValueError(
                        f"Inconsistent cells for ({u}, {v}): {cell_uv!r} mirrors to {mirrored!r}, got {cell_vu!r}."
                    )
                for edge_type in types_uv:
                    graph.add_edge(u, v, edge_type)

            # Step 2.3: pcalg cells hold the mark at the column endpoint.
            elif encoding == "pcalg":
                if cell_uv not in pcalg_mark or cell_vu not in pcalg_mark:
                    raise ValueError(f"Cannot decode pcalg marks ({cell_uv!r}, {cell_vu!r}) between {u} and {v}.")
                graph.add_edge(u, v, graph._to_edge_type(u, v, {u: pcalg_mark[cell_vu], v: pcalg_mark[cell_uv]}))

            # Step 2.4: causal-learn cells hold the mark at the row endpoint; the codes 4/5 encode
            #           coincident directed + bidirected edges.
            else:
                if cell_uv in causal_learn_mark and cell_vu in causal_learn_mark:
                    markers = {u: causal_learn_mark[cell_uv], v: causal_learn_mark[cell_vu]}
                    graph.add_edge(u, v, graph._to_edge_type(u, v, markers))
                elif (cell_uv, cell_vu) in causal_learn_coincident:
                    for edge_type in causal_learn_coincident[(cell_uv, cell_vu)]:
                        graph.add_edge(u, v, edge_type)
                else:
                    raise ValueError(
                        f"Cannot decode causal-learn marks ({cell_uv!r}, {cell_vu!r}) between {u} and {v}."
                    )

        # Step 3: Return the decoded graph.
        return graph

    def is_collider(self, u: Hashable, w: Hashable, v: Hashable, shielded: bool = True) -> bool:
        """
        Check whether `w` is a collider between `u` and `v`.

        `w` is a collider when both the edge from `u` and the edge from `v` have an arrowhead at
        `w` (e.g. ``u -> w <- v``, ``u <> w <- v``, ``u o> w <> v``).

        Parameters
        ----------
        u, v : Hashable
            The flanking nodes.
        w : Hashable
            The middle node to test as a collider.
        shielded : bool (default: True)
            If True, adjacency between `u` and `v` is irrelevant (the plain collider test). If
            False, `u` and `v` must additionally be non-adjacent, making this the
            unshielded-collider (v-structure) test.

        Returns
        -------
        bool

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("T", "M", "->"), ("M", "O", "->"), ("M", "I", "<-"), ("T", "I", "->")])
        >>> graph.is_collider("T", "M", "O")
        False
        >>> graph.is_collider("T", "M", "I")
        True
        >>> graph.is_collider("T", "M", "I", shielded=False)  # T and I are adjacent
        False
        """
        if not {u, v, w}.issubset(self.nodes):
            raise ValueError(f"{u}, {v}, {w} must be present in the graph.")

        if not shielded and self.has_edge(u, v):
            return False

        arrowhead_types = {"<-", "<>", "<o"} & self.SUPPORTED_EDGE_TYPES
        into_w = self.get_neighbors(w, arrowhead_types)
        return (u in into_w) and (v in into_w)

    def __eq__(self, other):
        """
        Checks if two graphs are equal. Two graphs are considered equal if they
        have the same nodes, edges, exposures, outcomes, latent variables, and variable roles.

        Parameters
        ----------
        other : graph object
            The other graph to compare with.

        Returns
        -------
        bool :
            True if the graphs are equal, False otherwise.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = _CoreGraph()
        >>> G1.__eq__(G2)
        True

        """
        if type(other) is not type(self):
            return False

        def canonical(graph):
            return (
                set(graph.nodes()),
                frozenset(frozenset({(u, markers[u]), (v, markers[v])}) for u, v, markers in graph.edges(data=True)),
                frozenset((role, frozenset(nodes)) for role, nodes in graph.get_role_dict().items()),
            )

        return canonical(self) == canonical(other)

    def __hash__(self):
        """
        Returns a hash derived from the same things ``__eq__`` compares: the class, the nodes, the
        edges, and the variable roles. Equal graphs therefore hash equally, and graphs of different
        types (e.g. an ``ADMG`` and a ``MAG``) with the same structure get distinct hashes.

        Returns
        -------
        int
            The hash value of the graph.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G1 = _CoreGraph(edge_list=[("A", "B", "->")])
        >>> G2 = _CoreGraph(edge_list=[("B", "A", "<-")])
        >>> hash(G1) == hash(G2)
        True

        """
        edges = frozenset(frozenset({(u, markers[u]), (v, markers[v])}) for u, v, markers in self.edges(data=True))
        roles = frozenset((role, frozenset(nodes)) for role, nodes in self.get_role_dict().items())
        return hash((type(self), frozenset(self.nodes()), edges, roles))

    def _validate_edges(
        self,
        edge_list,
    ):
        """
        Validates the value input by the user, then either raises an error.

        Parameters
        ----------
        edge_list : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...].
        """
        for edge in edge_list:
            if len(edge) != 3:
                raise ValueError(f"Edge tuple must have 3 elements. Edge {edge} is of length {len(edge)}.")

            u, v, edge_type = edge

            if (u is None) or (v is None):
                raise ValueError(f"Nodes cannot be None. Got {(u, v, edge_type)}.")
            if u == v:
                raise ValueError(f"Nodes cannot be the same for an edge. Got {(u, v, edge_type)}.")
            if not isinstance(edge_type, str) or edge_type not in self.SUPPORTED_EDGE_TYPES:
                raise ValueError(f"edge_type must be one of {self.SUPPORTED_EDGE_TYPES}. Got {(u, v, edge_type)}.")

    def _to_markers(
        self,
        edge: tuple[Hashable, Hashable, str],
    ) -> dict[Hashable, str]:
        """
        The `_to_markers` method converts the user's `edge_type` input into an internal representation.
        """
        u, v, edge_type = edge
        if edge_type == "<-":
            return {v: "-", u: ">"}
        elif edge_type == "o-":
            return {v: "-", u: "o"}
        elif edge_type == "<o":
            return {v: "o", u: ">"}
        elif edge_type == "<>":
            return {u: ">", v: ">"}
        else:
            return {u: edge_type[0], v: edge_type[1]}

    def _to_edge_type(
        self,
        u: Hashable,
        v: Hashable,
        markers: dict,
    ) -> str:
        """
        The `_to_edge_type` method converts the internal representation into the user's `edge_type` input.
        """
        u_marker = markers[u]
        v_marker = markers[v]

        marker_map = {
            (">", "-"): "<-",
            ("o", "-"): "o-",
            (">", "o"): "<o",
            (">", ">"): "<>",
        }
        return marker_map.get((u_marker, v_marker), f"{u_marker}{v_marker}")
