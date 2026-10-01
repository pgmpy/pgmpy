from itertools import combinations
from os import PathLike

import networkx as nx

from pgmpy.base._base import _CoreGraph
from pgmpy.utils.parser import parse_dagitty


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
    - :footcite:t:`zhang_2008`
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "<>", "--"])

    def is_maximal(self) -> bool:
        """
        Check whether the graph is maximal.

        An ancestral graph is maximal when no edge can be added without changing the implied
        conditional-independence relations -- equivalently, when every pair of non-adjacent nodes
        can be m-separated by some subset of the remaining nodes. By the characterization of
        :footcite:t:`zhang_2008` this holds exactly when no *primitive* inducing path (an inducing
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

    @classmethod
    def from_dagitty(cls, string: str | None = None, filename: str | PathLike | None = None) -> "MAG":
        """
        Initializes a `MAG` instance using DAGitty syntax.

        Creates a `MAG` from the dagitty string. The string should use the ``mag { ... }``
        header; directed edges use ``->``, bidirected edges use ``<->``, and undirected
        edges use ``--``. Variable roles are read from ``[exposure]``, ``[outcome]`` and
        ``[latent]`` annotations on standalone node statements.

        Parameters
        ----------
        string: str (default: None)
            A `DAGitty` style multiline set of statements representing the model.
            Refer https://cran.r-project.org/web/packages/dagitty/dagitty.pdf Page 10.

        filename: str or PathLike (default: None)
            The filename of the file containing the model in DAGitty syntax.

        Returns
        -------
        MAG
            The MAG described by the dagitty string, with variable roles set.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG.from_dagitty("mag { X [exposure] Y [outcome] X -> Y Y <-> Z }")
        >>> sorted(mag.get_edges(data=True))
        [('X', 'Y', '->'), ('Y', 'Z', '<>')]
        >>> mag.exposures, mag.outcomes
        ({'X'}, {'Y'})

        Notes
        -----
        - Coefficient annotations (``[beta=...]``) are ignored because MAGs do not carry parameters.
        - Endpoint marks other than tails and arrowheads (e.g. circle endpoints used by PAGs)
          can not be represented in a MAG and raise a ``ValueError``.

        References
        ----------
        dagitty syntax: https://cran.r-project.org/web/packages/dagitty/dagitty.pdf
        """
        if filename:
            with open(filename) as f:
                dagitty_str = f.readlines()
        elif string:
            dagitty_str = string.split("\n")
        else:
            raise ValueError("Either `filename` or `string` need to be specified")

        ebunch, roles, _, nodes = parse_dagitty(dagitty_str)

        edge_tuples = []
        for edge in ebunch:
            if len(edge) == 4:
                u, v, tail_mark, head_mark = edge
                if (tail_mark, head_mark) == ("-", ">"):
                    edge_tuples.append((u, v, "->"))
                elif (tail_mark, head_mark) == (">", ">"):
                    edge_tuples.append((u, v, "<>"))
                elif (tail_mark, head_mark) == ("-", "-"):
                    edge_tuples.append((u, v, "--"))
                else:
                    raise ValueError(
                        f"Edge ({u}, {v}) with endpoint marks ({tail_mark}, {head_mark}) "
                        "can not be represented in a MAG."
                    )
            else:
                # 2-tuples come from a non-mag header; keep them as directed edges.
                u, v = edge
                edge_tuples.append((u, v, "->"))

        mag = cls(edge_list=edge_tuples)
        mag.add_nodes_from(nodes)
        mag.latents = set(roles.get("latents", []))
        mag.exposures = set(roles.get("exposures", []))
        mag.outcomes = set(roles.get("outcomes", []))
        for role, variables in roles.items():
            if role not in ("latents", "exposures", "outcomes"):
                mag.with_role(role=role, variables=variables, inplace=True)
        return mag

    def to_dagitty(self) -> str:
        """
        Convert the MAG to dagitty syntax representation.

        The dagitty syntax represents the graph using the mag { statements } format with
        ``->`` for directed edges, ``<->`` for bidirected edges and ``--`` for undirected
        edges. Variable roles are written as ``[exposure]``, ``[outcome]`` and ``[latent]``
        annotations on standalone node statements. Isolated nodes (nodes with no edges)
        are included as standalone nodes.

        Returns
        -------
        str
            String representation of the MAG in dagitty syntax format.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG(edge_list=[("X", "Y", "->"), ("Y", "Z", "<>")])
        >>> print(mag.to_dagitty())
        mag {
        X -> Y
        Y <-> Z
        }

        >>> mag2 = MAG(edge_list=[("A", "B", "->")], exposures={"A"}, outcomes={"B"})
        >>> mag2.add_node("C")  # Isolated node
        >>> print(mag2.to_dagitty())
        mag {
        A -> B
        A [exposure]
        B [outcome]
        C
        }

        Notes
        -----
        - Node names are converted to string representations using str().
        - If node names contain spaces or special characters, they will be used as-is.
        - Users should ensure node names are valid in R/dagitty context if needed.

        References
        ----------
        dagitty syntax: https://cran.r-project.org/web/packages/dagitty/dagitty.pdf
        """
        statements = []

        edge_statements = {"->": "->", "<>": "<->", "--": "--"}
        for u, v, edge_type in sorted(self.get_edges(data=True), key=lambda e: (str(e[0]), str(e[1]), e[2])):
            statements.append(f"{u} {edge_statements[edge_type]} {v}")

        role_dict = self.get_role_dict()
        node_roles = {}
        for role, marker in (("exposures", "exposure"), ("outcomes", "outcome"), ("latents", "latent")):
            for node in set(role_dict.get(role, [])) | getattr(self, role):
                node_roles.setdefault(node, []).append(marker)

        for node in sorted(node_roles, key=str):
            for marker in node_roles[node]:
                statements.append(f"{node} [{marker}]")

        for node in sorted(nx.isolates(self), key=str):
            if node not in node_roles:
                statements.append(str(node))

        content = "\n".join(statements)
        if content:
            return f"mag {{\n{content}\n}}"
        else:
            return "mag {\n}"
