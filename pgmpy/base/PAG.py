#!/usr/bin/env python

from pgmpy.base.AncestralBase import AncestralBase


class PAG(AncestralBase):
    """
    Partial Ancestral Graph (PAG) implementation extending AncestralGraph.
    Implements PAG-specific methods from:
        - Definition 15: Lower/Upper manipulations
        - PAG-specific visibility (definitely visible edges)
        - Possible ancestor / m-connection checks
    """

    def is_definitely_visible_edge(self, u, v):
        """
        Checks if the directed edge u -> v is definitely visible in the PAG.

        'Definitely visible' means it is visible in ALL MAGs represented by this PAG.
        In practice we use the same structural conditions as MAG visibility,
        but applied to PAG marks (with circles allowed).
        """
        if not self.is_directed(u, v):
            return False

        for C in self.nodes():
            if C in {u, v}:
                continue
            if self.graph.has_edge(C, v) or self.graph.has_edge(v, C):
                continue  # C must not be adjacent to v

            # Case a: C *-> u
            if self.has_arrowhead_at(C, u):
                return True

            # Case b: collider-path search from C to u
            visited = set()
            stack = [(C, None)]  # {curr_node, prev_node}

            while stack:
                curr, prev = stack.pop()

                if curr == u and prev is not None:
                    if self.has_arrowhead_at(prev, u) and (
                        v in self.get_children(prev)
                    ):
                        return True
                    continue

                visited.add(curr)

                for nbr in self.graph.successors(curr):
                    if nbr == prev:
                        continue
                    # Must be collider at curr
                    if prev is not None and not (
                        self.has_arrowhead_at(prev, curr)
                        and self.has_arrowhead_at(nbr, curr)
                    ):
                        continue
                    if (v in self.get_children(curr)) or curr == C:
                        stack.append((nbr, curr))

        return False

    def lower_manipulation(self, X):
        """
        PAG lower-manipulation:
            - Delete all definitely visible edges out of X.
            - Replace all other edges out of X with bi-directed (<->) edges.
        """
        P_new = self.copy()

        for x in X:
            for y in list(P_new.graph.successors(x)):
                if self.is_definitely_visible_edge(x, y):
                    P_new.remove_edge(x, y)
                    if P_new.graph.has_edge(y, x):
                        P_new.remove_edge(y, x)
                else:
                    # Replace with bi-directed
                    P_new.add_edge(x, y, "arrowhead", "arrowhead")

        return P_new

    def upper_manipulation(self, X):
        """
        PAG upper-manipulation:
            - Delete all edges into X (same logic as MAG).
        """
        P_new = self.graph

        for x in X:
            for y in list(P_new.predecessors(x)):
                P_new.remove_edge(y, x)

        return P_new

    def possible_ancestors(self, node):
        """
        Returns the set of possible ancestors of `node` in the PAG
        (Def. in Section 3 for PAGs).
        """
        visited = set()
        stack = [node]

        while stack:
            curr = stack.pop()
            for nbr in self.graph.predecessors(curr):
                # There must be no arrowhead into nbr from curr
                if not self.has_arrowhead_at(curr, nbr) and nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        return visited

    def possible_descendants(self, node):
        """
        Returns the set of possible descendants of `node` in the PAG.
        """
        visited = set()
        stack = [node]

        while stack:
            curr = stack.pop()
            for nbr in self.graph.successors(curr):
                # There must be no arrowhead into nbr from curr
                if not self.has_arrowhead_at(nbr, curr) and nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        return visited
