from collections import deque
from collections.abc import Hashable, Iterable
from itertools import pairwise

import networkx as nx

from pgmpy.utils.types import Self


class _GraphAlgorithms:
    """Graph-algorithm methods for ``_CoreGraph``-based classes (inherited, not instantiated alone)."""

    def get_ancestral_graph(self, nodes: Iterable[Hashable]) -> Self:
        """
        Return the ancestral graph induced by `nodes`: the subgraph over `nodes` together with all of
        their ancestors.

        Parameters
        ----------
        nodes : iterable of Hashable
            A collection of nodes to induce the ancestral graph on.

        Returns
        -------
        ancestral_graph: Self
            ancestral graph object

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [
        ...     ("A", "B", "->"),
        ...     ("B", "C", "->"),
        ...     ("C", "D", "<>"),
        ...     ("C", "E", "--"),
        ... ]
        >>> graph = _CoreGraph(edge_list=edges)
        >>> ancestral_graph = graph.get_ancestral_graph(["C", "D"])
        >>> sorted(ancestral_graph.get_edges(data=True))
        [('A', 'B', '->'), ('B', 'C', '->'), ('C', 'D', '<>')]

        """
        if isinstance(nodes, str):
            raise ValueError(f"`nodes` must be a collection of nodes, not a single node. Got {nodes!r}.")
        nodes_set = set(nodes)

        ancestors = self.get_ancestors(nodes_set)

        return self.get_subgraph(ancestors)

    def get_markov_blanket(self, nodes: str | Iterable[str]) -> str | Iterable[str]:
        """
        Return the Markov blanket of the given node or nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            A node name or an iterable of node names whose Markov blanket is to be
            computed.

        Returns
        -------
        markov_blanket: set
            A set containing the nodes in the Markov blanket of the input node or
            nodes.

        Notes
        -----
        Defined only for graphs with directed/bidirected edges (DAGs and ADMGs). For an ADMG the
        Markov blanket of ``v`` is the bidirected-connected district ``dis(v)``, the parents of
        ``dis(v)``, and -- for every child ``c`` of ``v`` -- ``dis(c)`` together with its parents, all
        with ``v`` itself removed. For a DAG (where ``dis(v) = {v}``) this reduces to the parents,
        children, and the children's other parents (co-parents).

        Examples
        --------
        >>> from pgmpy.base import ADMG
        >>> edges = [
        ...     ("A", "B", "->"),
        ...     ("B", "C", "->"),
        ...     ("D", "E", "->"),
        ...     ("A", "D", "<>"),
        ...     ("B", "E", "<>"),
        ... ]
        >>> admg = ADMG(edge_list=edges, exposures={"A"}, outcomes={"C"})
        >>> sorted(admg.get_markov_blanket("B"))
        ['A', 'C', 'D', 'E']

        References
        ----------
        - :cite:p:`richardson_2003`
        """
        if not self.SUPPORTED_EDGE_TYPES <= {"->", "<-", "<>"}:
            raise TypeError(
                "get_markov_blanket is only supported for graphs with directed and bidirected edges "
                f"(DAG, ADMG); {self.__class__.__name__} supports {sorted(self.SUPPORTED_EDGE_TYPES)}."
            )

        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        if not nodes_set.issubset(self.nodes):
            raise ValueError("Input nodes must be subset of graph's nodes.")

        has_bidirected = "<>" in self.SUPPORTED_EDGE_TYPES

        def district_with_parents(node):
            # The node's bidirected-connected district plus the parents of every district member
            # (just the node and its parents when the graph has no bidirected edges).
            district = self.get_reachable_nodes(node, "<>") if has_bidirected else {node}
            members = set(district)
            for member in district:
                members.update(self.get_parents(member))
            return members

        markov_blanket = set()
        for node in nodes_set:
            markov_blanket |= district_with_parents(node)
            for child in self.get_children(node):
                markov_blanket |= district_with_parents(child)

        markov_blanket -= nodes_set

        return markov_blanket

    def has_inducing_path(self, u: Hashable, v: Hashable, w: Iterable[Hashable] | None = None) -> bool:
        """
        Check if there exists an inducing path between `u` and `v` relative to `w`.

        `w` is the pool of variables available for conditioning. An inducing path between `u` and
        `v` relative to `w` is a path such that:

        - every intermediate node in `w` is a collider on the path (an intermediate outside `w`
          can never be conditioned on, so it only blocks the path if it is a collider), and
        - every collider on the path is an ancestor of `u` or `v` (which keeps it open under
          every conditioning set drawn from `w`).

        A direct edge between `u` and `v` has no intermediate nodes, so it is (vacuously) an inducing
        path: adjacent nodes are always joined by one, matching the standard definition.

        Such a path is a dependence between `u` and `v` that no subset of `w` can block: `u` and
        `v` cannot be m-separated using only variables from `w`. With the default
        ``w = self.observed``, an inducing path exists exactly when `u` and `v` must be adjacent
        in the MAG over the observed variables :cite:p:`zhang_2008`.

        Parameters
        ----------
        u, v : Hashable
            The endpoints of the path. Both must be present in the graph.

        w : iterable of Hashable, optional (default: the observed nodes)
            The variables available for conditioning. ``None`` uses all nodes except the
            ``latents`` role.

        Returns
        -------
        bool
            True if an inducing path exists, False otherwise.

        Examples
        --------
        A latent chain cannot be blocked by conditioning on observed variables:

        >>> from pgmpy.base import MAG
        >>> mag = MAG(edge_list=[("X", "L", "->"), ("L", "Y", "->")], latents={"L"})
        >>> mag.has_inducing_path("X", "Y")
        True

        A collider path is inducing only if the colliders are ancestors of the endpoints:

        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [
        ...     ("A", "B", "<>"),
        ...     ("A", "C", "<>"),
        ...     ("B", "D", "<>"),
        ...     ("A", "D", "->"),
        ...     ("B", "C", "->"),
        ... ]
        >>> graph = _CoreGraph(edge_list=edges)
        >>> graph.has_inducing_path("C", "D")
        True

        """
        for node in (u, v):
            if node not in self.nodes():
                raise ValueError(f"Node {node} not in graph.")

        w = self.observed if w is None else set(w)
        ancestors = self.get_ancestors([u, v])

        for path in nx.all_simple_edge_paths(self, u, v):
            for into_edge, out_of_edge in pairwise(path):
                mid = into_edge[1]
                is_collider = self.edges[into_edge][mid] == ">" and self.edges[out_of_edge][mid] == ">"
                if (not is_collider and mid in w) or (is_collider and mid not in ancestors):
                    break
            else:
                return True
        return False

    def get_mconnected_nodes(
        self, nodes: Hashable | Iterable[Hashable], conditioning_set: Iterable[Hashable] | None = None
    ) -> set[Hashable]:
        """
        Return all nodes m-connected to `nodes` given `conditioning_set`.

        A walk m-connects two nodes given a conditioning set `Z` when every collider on the walk
        (a node with an arrowhead on both the entering and the leaving edge) is in `Z`, and every
        non-collider is not in `Z`. This walk criterion is equivalent to the standard path
        criterion with colliders in ``An(Z)`` :cite:p:`richardson_2003`; conditioning on a
        descendant of a collider opens it because the walk can detour down to the descendant and
        back. On a purely directed graph this is exactly d-connection.

        Parameters
        ----------
        nodes : Hashable or iterable of Hashable
            A single source node, or a list/set of sources; the result is the union over the
            collection. A ``str``, ``int`` or ``tuple`` is one node.

        conditioning_set : iterable of Hashable, optional (default: empty)
            The conditioned (observed) variables. Must be disjoint from `nodes`.

        Returns
        -------
        connected : set
            All m-connected nodes, including `nodes` themselves. Conditioned nodes appear when an
            m-connecting walk ends at them.

        Raises
        ------
        ValueError
            If a node is missing from the graph, `nodes` and `conditioning_set` overlap, or the
            graph is not ancestral: circle (``o``) endpoints (PAGs) and nodes carrying both an
            undirected edge and an incoming arrowhead (e.g. PDAGs) leave the criterion undefined.

        See Also
        --------
        is_mseparated : The pairwise separation query.

        Examples
        --------
        >>> from pgmpy.base import ADMG
        >>> admg = ADMG(edge_list=[("X", "C", "->"), ("Y", "C", "->")])
        >>> sorted(admg.get_mconnected_nodes("X"))
        ['C', 'X']
        >>> sorted(admg.get_mconnected_nodes("X", conditioning_set={"C"}))
        ['C', 'X', 'Y']

        """
        nodes = list(nodes) if isinstance(nodes, (list, set, frozenset)) else [nodes]
        z = set() if conditioning_set is None else set(conditioning_set)

        for node in [*nodes, *z]:
            if node not in self.nodes():
                raise ValueError(f"Node {node} not in graph.")
        if overlap := (set(nodes) & z):
            raise ValueError(f"`nodes` and `conditioning_set` must be disjoint. Got {sorted(overlap)} in both.")
        if circle_types := (self.get_unique_edge_types() & {"o-", "o>", "oo"}):
            raise ValueError(
                "m-separation is undefined for graphs with circle endpoints (PAGs), where edge "
                f"orientation is uncertain. Found circle edge types: {sorted(circle_types)}."
            )

        # The walk criterion below matches path-based m-separation only on ancestral graphs, where
        # a node on an undirected edge has no arrowheads into it; reject anything else (e.g. a
        # PDAG, whose `--` edges mean "unoriented", not "selection").
        arrowhead_types = {"<-", "<>"} & self.SUPPORTED_EDGE_TYPES
        for x, y, _ in self.get_edges(edge_types={"--"}):
            for node in (x, y):
                if self.get_neighbors(node, arrowhead_types):
                    raise ValueError(
                        f"m-separation requires an ancestral graph: node {node} has both an "
                        "undirected edge and an incoming arrowhead."
                    )

        # BFS over states (node, mark at the node on the edge used to enter it). The walk may pass
        # through a node w iff w-is-a-collider-here equals w-is-conditioned.
        connected = set(nodes)
        visited = set()
        queue = deque()
        for source in nodes:
            for neighbor in self.neighbors(source):
                for data in self.get_edge_data(source, neighbor).values():
                    queue.append((neighbor, data[neighbor]))

        while queue:
            state = queue.popleft()
            if state in visited:
                continue
            visited.add(state)
            node, in_mark = state
            connected.add(node)
            for neighbor in self.neighbors(node):
                for data in self.get_edge_data(node, neighbor).values():
                    is_collider = in_mark == ">" and data[node] == ">"
                    if is_collider == (node in z):
                        queue.append((neighbor, data[neighbor]))
        return connected

    def is_mseparated(self, u: Hashable, v: Hashable, conditioning_set: Iterable[Hashable] | None = None) -> bool:
        """
        Check whether `u` and `v` are m-separated given `conditioning_set`.

        `u` and `v` are m-separated when no m-connecting walk joins them (see
        :meth:`get_mconnected_nodes` for the criterion). On a purely directed graph this is
        exactly d-separation; on ADMGs/MAGs it additionally accounts for bidirected (latent
        confounding) and undirected (selection) edges.

        Parameters
        ----------
        u, v : Hashable
            The nodes to test. Both must be present in the graph and outside `conditioning_set`.

        conditioning_set : iterable of Hashable, optional (default: empty)
            The conditioned (observed) variables.

        Returns
        -------
        bool
            True if `u` and `v` are m-separated, False if they are m-connected.

        Raises
        ------
        ValueError
            If `u` or `v` is missing from the graph or inside `conditioning_set`, or the graph
            is not ancestral (circle endpoints, or an undirected edge meeting an arrowhead).

        See Also
        --------
        get_mconnected_nodes : All nodes m-connected to a set of sources.

        Examples
        --------
        >>> from pgmpy.base import ADMG
        >>> admg = ADMG(edge_list=[("X", "C", "->"), ("Y", "C", "->")])
        >>> admg.is_mseparated("X", "Y")
        True
        >>> admg.is_mseparated("X", "Y", conditioning_set={"C"})
        False

        """
        z = set() if conditioning_set is None else set(conditioning_set)
        if u in z or v in z:
            raise ValueError(f"u and v must not be in conditioning_set. Got u={u!r}, v={v!r}, z={sorted(z)}.")
        if v not in self.nodes():
            raise ValueError(f"Node {v} not in graph.")
        return v not in self.get_mconnected_nodes(u, conditioning_set=z)

    def has_path(self, source: Hashable, target: Hashable, edge_types: str | Iterable[str] | None = None) -> bool:
        """
        Check whether a path from `source` to `target` exists, following only edges of `edge_types`.

        Each step of the path follows an edge whose type (read from the current node's endpoint) is in
        `edge_types`; e.g. ``edge_types="->"`` checks for a directed path. ``None`` follows any edge
        type. A node always has a (trivial) path to itself.

        Parameters
        ----------
        source, target : Hashable
            The endpoints of the path. Both must be present in the graph.

        edge_types : str or iterable of str, optional (default: None)
            The edge type(s) a step may follow; ``None`` allows any type.

        Returns
        -------
        bool

        See Also
        --------
        get_all_paths : Enumerate the paths themselves.
        get_reachable_nodes : All nodes reachable under the given edge types.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->")])
        >>> graph.has_path("A", "C", edge_types="->")
        True

        """
        if target not in self.nodes():
            raise ValueError(f"Node {target} not in graph.")
        return target in self.get_reachable_nodes(source, edge_types)

    def get_all_paths(
        self, source: Hashable, target: Hashable, edge_types: str | Iterable[str] | None = None
    ) -> list[list[Hashable]]:
        """
        Return all simple paths (no repeated nodes) from `source` to `target` following `edge_types`.

        Each step follows an edge whose type (read from the current node's endpoint) is in
        `edge_types`; e.g. ``edge_types="->"`` enumerates directed paths. ``None`` follows any edge
        type. If ``source == target`` the single trivial path ``[source]`` is returned.

        Parameters
        ----------
        source, target : Hashable
            The endpoints of the paths. Both must be present in the graph.

        edge_types : str or iterable of str, optional (default: None)
            The edge type(s) a step may follow; ``None`` allows any type.

        Returns
        -------
        paths : list of lists
            Each inner list is a path given as a sequence of nodes.

        See Also
        --------
        has_path : Whether any such path exists.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("A", "C", "->")])
        >>> sorted(graph.get_all_paths("A", "C", edge_types="->"))
        [['A', 'B', 'C'], ['A', 'C']]

        """
        for node in (source, target):
            if node not in self.nodes():
                raise ValueError(f"Node {node} not in graph.")

        paths = []
        stack = [(source, [source])]
        while stack:
            current, path = stack.pop()
            if current == target:
                paths.append(path)
                continue
            for neighbor in self.get_neighbors(current, edge_types):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return paths

    def get_topological_order(self) -> list[Hashable]:
        """
        Return a topological ordering of the nodes consistent with the directed edges.

        A topological order is an ordering of the nodes in which every directed edge ``u -> v`` has
        ``u`` before ``v``. Following the standard definition for mixed graphs [1]_, bidirected
        (``"<>"``) and undirected (``"--"``) edges impose *no* ordering constraint: a bidirected
        edge encodes latent confounding (no ancestral relation), and undirected edges form chain
        components that are ordered only through their directed edges. The order is therefore well
        defined for DAGs, ADMGs, and (C)PDAGs/MAGs, and is computed on the directed sub-graph.

        Circle endpoints (``"o"``, as used in PAGs) leave the orientation -- and hence the ancestral
        order -- genuinely uncertain, so a ``ValueError`` is raised if the graph contains any circle
        mark.

        Returns
        -------
        order : list
            The nodes in a topological order.

        Raises
        ------
        ValueError
            If the graph contains any edge with a circle endpoint (i.e. a PAG).

        References
        ----------
        - :cite:p:`richardson_2003`

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->")])
        >>> graph.get_topological_order()
        ['A', 'B', 'C']
        """
        # Circle endpoints (PAGs) survive canonicalization only as "o-", "o>" and "oo".
        circle_types = self.get_unique_edge_types() & {"o-", "o>", "oo"}
        if circle_types:
            raise ValueError(
                "Topological order is undefined for graphs with circle endpoints (PAGs), where edge "
                f"orientation is uncertain. Found circle edge types: {sorted(circle_types)}."
            )
        return list(nx.topological_sort(self.get_directed_graph()))

    def has_directed_cycle(self) -> bool:
        """
        Returns whether the directed (``"->"``) edges of the graph contain a cycle.

        Only the directed sub-graph is considered (see ``get_directed_graph``); undirected,
        bidirected and circle edges are ignored.

        Returns
        -------
        bool
            True if the directed edges form at least one cycle, otherwise False.

        See Also
        --------
        get_directed_graph : The directed projection used for the check.
        get_topological_order : A topological order (only exists when there is no directed cycle).
        """
        return not nx.is_directed_acyclic_graph(self.get_directed_graph())

    def _check_new_unshielded_collider(self, u: Hashable, v: Hashable) -> bool:
        """
        Tests if orienting an undirected edge u - v as u -> v creates new unshielded V-structures in the PDAG.

        Checks whether v has any directed parents other than u that are not adjacent to u.

        Returns
        -------
        True, if the orientation u -> v would lead to creation of a new V-structure.
        False, if no new V-structures are formed.
        """
        for node in self.get_parents(v):
            if (node != u) and (not self.has_edge(u, node)):
                return True
        return False
