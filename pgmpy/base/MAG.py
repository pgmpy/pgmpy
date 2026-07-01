from itertools import combinations

from pgmpy.base._base import _CoreGraph


class MAG(_CoreGraph):
    """
    Class for representing Maximal Ancestral Graphs (MAGs).

    A MAG is a graph used in causal inference to represent conditional independence relations when
    some variables are latent (unobserved) or selection is present. MAGs allow directed (``"->"``),
    bidirected (``"<>"``), and undirected (``"--"``) edges -- bidirected edges encode latent
    confounding and undirected edges encode selection bias. A MAG is *maximal*: no edge can be added
    without changing the implied conditional-independence relations.

    Built on :class:`~pgmpy.base._base._CoreGraph`, restricted to the directed/bidirected/undirected
    edge types via ``SUPPORTED_EDGE_TYPES`` (no circle endpoints -- those are for PAGs).

    Parameters
    ----------
    edge_list : iterable of tuples, optional
        Edges of the form ``(u, v, edge_type)`` with ``edge_type`` one of ``"->"``, ``"<-"``,
        ``"<>"``, ``"--"``.

    latents : set, default=set()
        Set of latent (unobserved) variables.

    exposures, outcomes : set, default=set()
        Treatment / response variables (causal-analysis roles).

    roles : dict, optional (default: None)
        A mapping of role name to node(s); equivalent to calling ``with_role`` for each entry.

    Examples
    --------
    >>> from pgmpy.base import MAG
    >>> mag = MAG(edge_list=[("L", "A", "->"), ("A", "B", "<>")], latents={"L"})
    >>> sorted(mag.get_edges(data=True))
    [('A', 'B', '<>'), ('L', 'A', '->')]
    >>> mag.latents
    {'L'}

    References
    ----------
    - :cite:p:`zhang_2008`
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "<>", "--"])

    def is_maximal(self) -> bool:
        """
        Check whether the graph is maximal.

        An ancestral graph is maximal when no edge can be added without changing the implied
        conditional-independence relations -- equivalently, when every pair of non-adjacent nodes
        can be m-separated by some subset of the remaining nodes. By the characterization of
        :cite:p:`zhang_2008` this holds exactly when no *primitive* inducing path (an inducing
        path relative to the full vertex set: every intermediate node is a collider and an
        ancestor of an endpoint) joins a non-adjacent pair. The ``latents`` role is ignored:
        maximality is a property of the graph itself.

        Returns
        -------
        bool
            True if the graph is maximal, False otherwise.

        See Also
        --------
        has_inducing_path : The underlying inducing-path query.
        is_mseparated : The m-separation query maximality is defined through.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> edges = [("A", "B", "<>"), ("A", "C", "<>"), ("B", "D", "<>"), ("A", "D", "->"), ("B", "C", "->")]
        >>> MAG(edge_list=edges).is_maximal()  # C ... D joined by an inducing path
        False
        >>> MAG(edge_list=[*edges, ("C", "D", "<>")]).is_maximal()
        True

        """
        nodes = set(self.nodes())
        return not any(
            not self.has_edge(u, v) and self.has_inducing_path(u, v, w=nodes) for u, v in combinations(self.nodes(), 2)
        )

    def is_visible_edge(self, u, v):
        """
        Whether the directed edge ``u -> v`` is visible in the MAG.

        A directed edge ``u -> v`` is visible if there is a node ``c`` not adjacent to ``v`` such
        that either ``c *-> u``, or there is a collider path from ``c`` into ``u`` on which every
        intermediate node is a parent of ``v``.

        Parameters
        ----------
        u, v : Hashable
            The tail and head of the directed edge.

        Returns
        -------
        bool

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG(edge_list=[("A", "D", "->"), ("B", "C", "->"), ("X", "A", "->")])
        >>> mag.is_visible_edge("A", "D")
        True
        >>> mag.is_visible_edge("B", "C")
        False
        """
        if v not in self.get_children(u):
            return False

        into_u = self.get_neighbors(u, {"<-", "<>"})
        parents_v = self.get_parents(v)
        neighbors_v = self.get_neighbors(v)
        for c in self.nodes:
            if c in {u, v} or c in neighbors_v:
                continue
            # Condition 1: c *-> u directly.
            if c in into_u:
                return True
            # Condition 2: a collider path from c into u whose intermediates are all parents of v.
            for path in self.get_all_paths(c, u):
                if len(path) < 3 or path[-2] not in into_u:
                    continue
                valid = True
                for i in range(1, len(path) - 1):
                    prev_node, curr_node, next_node = path[i - 1], path[i], path[i + 1]
                    if (not self.is_collider(prev_node, curr_node, next_node)) or (curr_node not in parents_v):
                        valid = False
                        break
                if valid:
                    return True
        return False

    def lower_manipulation(self, X, inplace=False):
        """
        Return the MAG after lower manipulation of `X`.

        Visible directed edges out of `X` are removed; invisible ones are removed and replaced by a
        bidirected edge from the child to each of its other (non-`X`) neighbours, to preserve the
        conditional independencies.

        Parameters
        ----------
        X : set
            The nodes to manipulate.

        inplace : bool (default: False)
            If True, modify and return this graph; otherwise return a modified copy.

        Returns
        -------
        MAG

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG(edge_list=[("A", "B", "->"), ("C", "B", "->")])
        >>> new_mag = mag.lower_manipulation({"A"})
        >>> new_mag.has_edge("B", "C", "<>")
        True
        """
        new_mag = self if inplace else self.copy()

        visible, invisible = [], []
        for u in X:
            for v in self.get_children(u):
                (visible if self.is_visible_edge(u, v) else invisible).append((u, v))

        for u, v in visible + invisible:
            new_mag.remove_edge(u, v, "->")

        for u, v in invisible:
            other = v if u in X else u
            for neighbor in self.get_neighbors(v):
                if neighbor != other and neighbor not in X:
                    # A MAG holds at most one edge per pair, so replace any existing edge with `<>`.
                    if new_mag.has_edge(other, neighbor):
                        for edge_type in new_mag.get_edge_type(other, neighbor):
                            new_mag.remove_edge(other, neighbor, edge_type)
                    new_mag.add_edge(other, neighbor, "<>")
        return new_mag

    def upper_manipulation(self, X, inplace=False):
        """
        Return the MAG after upper manipulation of `X`.

        Every edge with an arrowhead into a node of `X` (a directed edge ``* -> X`` or a bidirected
        edge ``* <> X``) is removed; all other edges are kept.

        Parameters
        ----------
        X : set
            The nodes to manipulate.

        inplace : bool (default: False)
            If True, modify and return this graph; otherwise return a modified copy.

        Returns
        -------
        MAG

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG(edge_list=[("Y", "X", "->"), ("X", "Z", "->"), ("A", "X", "->")])
        >>> new_mag = mag.upper_manipulation({"X"})
        >>> new_mag.has_edge("X", "Z"), new_mag.has_edge("A", "X"), new_mag.has_edge("X", "Y")
        (True, False, False)
        """
        new_mag = self if inplace else self.copy()
        for u in X:
            for edge_type in ("<-", "<>"):
                for v in self.get_neighbors(u, edge_type):
                    new_mag.remove_edge(u, v, edge_type)
        return new_mag
