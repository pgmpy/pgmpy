from collections import deque
from typing import Hashable, Iterable, Optional

from pgmpy.base import MAG


class PAG(MAG):
    """
    Partial Ancestral Graph (PAG).

    A PAG represents an equivalence class of MAGs. It allows for circle endpoints ('o')
    to represent uncertainty about whether an endpoint is an arrow ('>') or a tail ('-').
    """

    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
        roles=None,
    ):
        """Initialize Partial Ancestral Graphs.

        Unlike MAGs, PAGs allow circle marks ('o') at edge endpoints to represent
        uncertainty about the true edge mark in the underlying MAG.

        Parameters
        ----------
        ebunch : Iterable[tuple], optional
            An iterable of edges of the form (u, v, u_mark, v_mark) used to
            initialize the graph. Each mark must be one of {">", "-", "o"}.
            Default is None, which initializes an empty graph.

        latents : set, optional
            Set of latent (unobserved) variables in the graph. Default is
            an empty set.

        roles : dict, optional (default: None)
            The keys are roles, and the values are role names (strings or iterables of str).
            If provided, this will automatically assign roles to the nodes in the graph.
            Passing a key-value pair via ``roles`` is equivalent to calling
            ``with_role(role, variables)`` for each key-value pair in the dictionary.

        Returns
        -------
        PAG
            A new instance of a Partial Ancestral Graph.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        [('A', 'B', {'marks': {'A': '-', 'B': '>'}}),
         ('B', 'C', {'marks': {'B': '>', 'C': '-'}}),
         ('C', 'D', {'marks': {'C': 'o', 'D': 'o'}})]

        Roles can be assigned to nodes in the graph at construction or using methods.

        At construction:
        >>> g = AncestralBase(
        ...     ebunch=[("L", "A", "-", ">"), ("B", "C", "-", ">")],
        ...     latents={"L"},
        ...     roles={"exposure": "A", "outcome": "B"},
        ... )

        Roles can also be assigned after creation using ``with_role`` method.

        >>> g = g.with_role("adjustment", {"L", "C"})

        Vertices of a specific role can be retrieved using ``get_role`` method.

        >>> g.get_role("exposure")
        ["A"]
        >>> g.get_role("adjustment")
        ["L", "C"]
        """
        super().__init__(ebunch=ebunch, latents=latents, roles=roles)

    def is_definite_non_collider(self, vertex, adj_u, adj_v):
        """
        Determine if a vertex on a path is a definite non-collider.

        A vertex is a definite non-collider if:
        - Either incident edge has a tail at the vertex, or
        - Both incident edges have circle marks at vertex and the two adjacent vertices are not adjacent to each other.

        Parameters
        ----------
        vertex : Hashable
            The vertex to check on the path.

        adj_u : Hashable
            The node preceding `vertex` on the path.

        adj_v : Hashable
            The node following `vertex` on the path.

        Returns
        -------
        bool
            True if the vertex is a definite non-collider, False otherwise.
        """
        u_mark = self.edges[adj_u, vertex]["marks"].get(vertex)
        v_mark = self.edges[vertex, adj_v]["marks"].get(vertex)
        if u_mark == "-" or v_mark == "-":
            return True
        if u_mark == "o" and v_mark == "o" and not self.has_edge(adj_u, adj_v):
            return True
        return False

    def has_possibly_directed_path(self, u, v):
        """
        Check if there exists a possibly directed path from node u to node v.

        A path is possibly directed if no edge points into the preceding node along the path.

        Parameters
        ----------
        u : Hashable
            The starting node of the path.

        v : Hashable
            The target node of the path.

        Returns
        -------
        bool
            True if a possibly directed path exists from u to v, False otherwise.
        """
        visited = set()
        queue = deque([u])
        while queue:
            current = queue.popleft()
            if current == v:
                return True
            visited.add(current)
            for neighbor in self.get_neighbors(current):
                mark = self.edges[current, neighbor]["marks"][current]
                if mark != ">" and neighbor not in visited:
                    queue.append(neighbor)
        return False

    def get_possible_ancestors(self, node):
        """
        Return the set of possible ancestors of a given node.

        A node X is a possible ancestor of Y if X=Y or if there exists a possibly directed path from X to Y.

        Parameters
        ----------
        node : Hashable
            The node whose possible ancestors are being queried.

        Returns
        -------
        set
            Set of possible ancestor nodes including the node itself.
        """
        possible_ancestors = set([node])
        for other in self.nodes:
            if self.has_possibly_directed_path(other, node):
                possible_ancestors.add(other)
        return possible_ancestors

    def is_definitely_visible(self, u, v):
        """
        Determine if an edge u -> v is definitely visible in the PAG.

        An edge is definitely visible if it satisfies the visibility conditions
        for all MAGs represented by the PAG.

        Parameters
        ----------
        u : Hashable
            The source node of the edge.

        v : Hashable
            The target node of the edge.

        Returns
        -------
        bool
            True if the edge is definitely visible, False otherwise.
        """
        if not self.has_edge(u, v):
            return False
        if self.edges[u, v]["marks"][u] != "-" or self.edges[u, v]["marks"][v] != ">":
            return False

        for neighbor in self.get_neighbors(u, v_type=">"):
            if neighbor not in self.get_neighbors(v):
                return True

        stack = [u]
        visited = set()

        while stack:
            current = stack.pop()

            for pred in self.get_neighbors(current, u_type=None, v_type=">"):
                if pred in visited or pred == u:
                    continue
                visited.add(pred)

                if pred not in self.get_neighbors(v, u_type=None, v_type=None):
                    return True

                if pred in self.get_neighbors(v, u_type="-", v_type=">"):
                    stack.append(pred)

        return False

    def find_path(self, start, end):
        """
        Finds a single uncovered, potentially directed path from start to end
        using a non-recursive Depth-First Search (DFS).

        A path is:
        - Potentially directed (p.d.) if for any edge X--Y on the path, the mark
          at X is not '>'.
        - Uncovered if for any triple <X, Y, Z> on the path, X and Z are not
          adjacent.

        Parameters
        ----------
        start : Hashable
            The starting node of the path.
        end : Hashable
            The target node of the path.

        Returns
        -------
        list or None
            A list of nodes representing the path, or None if no such path exists.
        """
        stack = [(start, [start])]
        visited = {start}

        while stack:
            current, path = stack.pop()

            if current == end:
                return path
            for neighbor in sorted(self.get_neighbors(current)):
                if neighbor not in visited:
                    if self.edges[current, neighbor]["marks"][current] != ">":
                        is_uncovered = True
                        if len(path) >= 2:
                            prev_node = path[-1]
                            if self.has_edge(prev_node, neighbor):
                                is_uncovered = False

                        if is_uncovered:
                            new_path = path + [neighbor]
                            stack.append((neighbor, new_path))
                            visited.add(neighbor)

        return None

    def rule_0(self, pag, sep_set):
        changed = False
        for u in pag.nodes:
            for v in pag.get_neighbors(u):
                for w in pag.get_neighbors(v):
                    if u != w and not pag.has_edge(u, w):
                        if v in sep_set.get((u, w), set()) or v in sep_set.get(
                            (w, u), set()
                        ):
                            pass
                        else:
                            if (
                                pag.edges[u, v]["marks"][v] == "o"
                                and pag.edges[v, w]["marks"][v] == "o"
                            ):
                                pag.edges[u, v]["marks"][v] = ">"
                                pag.edges[v, w]["marks"][v] = ">"
                                changed = True
        return changed

    def rule_1(self, pag):
        changed = False
        for u in pag.nodes:
            for v in pag.get_neighbors(u, u_type=None, v_type=">"):
                for w in pag.get_neighbors(u, u_type="o"):
                    if not pag.has_edge(v, w):
                        if pag.edges[u, w]["marks"][u] == "o":
                            pag.edges[u, w]["marks"][u] = ">"
                            pag.edges[u, w]["marks"][w] = "-"
                            changed = True
        return changed

    def rule_2(self, pag):
        changed = False
        for u in pag.nodes:
            for v in pag.get_neighbors(u, u_type="-", v_type=">"):
                for w in pag.get_neighbors(u, u_type=">", v_type="-"):
                    if pag.edges[u, w]["marks"][w] == "-":
                        if pag.edges.get((v, w)) and pag.edges[v, w]["marks"][w] == "o":
                            pag.edges[v, w]["marks"][w] = ">"
                            pag.edges[v, w]["marks"][v] = "-"
                            changed = True
            for v in pag.get_neighbors(u, u_type=None, v_type=">"):
                for w in pag.get_neighbors(u, u_type=">", v_type="-"):
                    if pag.edges[u, w]["marks"][w] == "-":
                        if pag.edges.get((v, w)) and pag.edges[v, w]["marks"][w] == "o":
                            pag.edges[v, w]["marks"][w] = ">"
                            pag.edges[v, w]["marks"][v] = "-"
                            changed = True
        return changed

    def rule_6(self, pag):
        changed = False
        for u in pag.nodes:
            for _ in pag.get_neighbors(u, u_type="-", v_type="-"):
                for w in pag.get_neighbors(u, u_type="o", v_type=None):
                    if pag.edges[u, w]["marks"][u] == "o":
                        pag.edges[u, w]["marks"][u] = "-"
                        changed = True
        return changed

    def rule_7(self, pag):
        changed = False
        for u in pag.nodes:
            for v in pag.get_neighbors(u, u_type=None, v_type="o"):
                if pag.edges[v, u]["marks"][v] == "-":
                    for w in pag.get_neighbors(u, u_type="o"):
                        if not pag.has_edge(v, w):
                            pag.edges[u, w]["marks"][u] = "-"
                            changed = True
        return changed

    def rule_8(self, pag):
        changed = False
        for u in pag.nodes:
            for v in pag.get_neighbors(u, u_type="-", v_type=">"):
                for w in pag.get_neighbors(u, u_type=">", v_type="-"):
                    if (
                        pag.edges.get((v, w))
                        and pag.edges[v, w]["marks"][v] == "o"
                        and pag.edges[v, w]["marks"][w] == ">"
                    ):
                        pag.edges[v, w]["marks"][v] = "-"
                        changed = True

            for v in pag.get_neighbors(u, v_type="o"):
                if pag.edges[v, u]["marks"][v] == "-":
                    for w in pag.get_neighbors(u, u_type=">", v_type="-"):
                        if (
                            pag.edges.get((v, w))
                            and pag.edges[v, w]["marks"][v] == "o"
                            and pag.edges[v, w]["marks"][w] == ">"
                        ):
                            pag.edges[v, w]["marks"][v] = "-"
                            changed = True
        return changed

    def rule_9(self, pag):
        changed = False
        for u, v in pag.edges:
            if (
                pag.edges[u, v]["marks"][u] == "o"
                and pag.edges[u, v]["marks"][v] == ">"
            ):

                queue = deque([(u, [u])])
                visited = {u}

                while queue:
                    current, path = queue.popleft()

                    if current == v:
                        is_uncovered_pd_path = True
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            if pag.edges[u, v]["marks"][u] == ">":
                                is_uncovered_pd_path = False
                                break
                            if i > 0 and i < len(path) - 1:
                                prev_node = path[i - 1]
                                if pag.has_edge(prev_node, v):
                                    is_uncovered_pd_path = False
                                    break

                        if is_uncovered_pd_path:
                            v = path[1]
                            if not pag.has_edge(v, v):
                                pag.edges[u, v]["marks"][u] = "-"
                                changed = True
                                break

                    for neighbor in pag.get_neighbors(current):
                        if (
                            pag.edges[current, neighbor]["marks"][current] != ">"
                            and neighbor not in visited
                        ):
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
        return changed

    def rule_10(self, pag):
        changed = False
        for u in pag.nodes:
            for v in pag.get_neighbors(u, u_type=">", v_type="-"):
                for w in pag.get_neighbors(u, u_type=">", v_type="-"):
                    if v != w:
                        for alpha in pag.get_neighbors(u, u_type="o", v_type=">"):
                            p1 = self.find_path(alpha, v)
                            p2 = self.find_path(alpha, w)

                            if p1 and p2:
                                mu = p1[1] if len(p1) > 1 else v
                                omega = p2[1] if len(p2) > 1 else w

                                if mu != omega and not pag.has_edge(mu, omega):
                                    pag.edges[alpha, u]["marks"][alpha] = "-"
                                    changed = True
        return changed

    def apply_orientation_rules(self, rules, inplace=False, sepsets=None):
        pag = self if inplace else self.copy()

        rules_map = {
            "R0": pag.rule_0,
            "R1": pag.rule_1,
            "R2": pag.rule_2,
            "R3": pag.rule_3,
            "R4": pag.rule_4,
            "R5": pag.rule_5,
            "R6": pag.rule_6,
            "R7": pag.rule_7,
            "R8": pag.rule_8,
            "R9": pag.rule_9,
            "R10": pag.rule_10,
        }

        if not rules:
            rules_to_apply = [
                "R0",
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
                "R7",
                "R8",
                "R9",
                "R10",
            ]

        else:
            rules_to_apply = [r.upper() for r in rules]

        changed = True
        while changed:
            changed = False
            for rule_name in rules_to_apply:
                rule_func = rules_map.get(rule_name)
                if rule_func:
                    if rule_name in ["R0", "R4"]:
                        if sepsets is None:
                            raise ValueError(
                                f"Rule {rule_name} requires sepsets to be provided."
                            )
                        if rule_func(pag, sepsets):
                            changed = True
                    else:
                        if rule_func(pag):
                            changed = True
