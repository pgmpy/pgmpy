#!/usr/bin/env python

from pgmpy.base.AncestralBase import AncestralBase


class MAG(AncestralBase):
    """
    Maximal Ancestral Graph (MAG) implementation extending AncestralGraph.
    Implements MAG-specific methods from:
        - Definition 8: Visibility
        - Definition 11: Lower/Upper manipulations
        - Inducing path checking
    """

    def is_visible_edge(self, u, v):

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

            # Case b: collider path check from C to u
            visited = set()
            stack = [(C, None)]  # (current, previous)

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
        MAG lower-manipulation: For each x in X:
            - Remove all visible edges out of x.
            - Replace all invisible edges out of x with bi-directed edges.

        Returns a new MAG instance (does not modify in place).
        """
        M_new = self.copy()

        for x in X:
            for y in list(M_new.graph.successors(x)):
                if self.is_visible_edge(x, y):
                    M_new.remove_edge(x, y)
                    if M_new.graph.has_edge(y, x):
                        M_new.remove_edge(y, x)
                else:
                    # Replace with bi-directed
                    M_new.add_edge(x, y, "arrowhead", "arrowhead")

        return M_new

    def upper_manipulation(self, X):
        """
        MAG upper-manipulation: For each x in X, delete all edges into x.
        """
        M_new = self.copy()

        for x in X:
            for y in list(M_new.graph.predecessors(x)):
                M_new.remove_edge(y, x)

        return M_new

    def has_inducing_path(self, u, v, L=None):
        """
        Checks if there exists an inducing path between u and v
        relative to L (default empty set) in the MAG.

        Definition: Every non-endpoint vertex not in L is a collider
        on the path and every collider is an ancestor of an endpoint.

        Parameters
        ----------
        u, v : hashable
            Endpoints to check for an inducing path.

        L : set, optional
            Subset of vertices relative to which the path is evaluated.
            Defaults to empty set.

        Returns
        -------
        bool
            True if such a path exists, False otherwise.
        """
        if L is None:
            L = set()

        def dfs(curr, target, visited, path):
            if curr == target:
                n = len(path)
                for i in range(1, n - 1):
                    vertex = path[i]
                    if vertex not in L:
                        # Must be collider
                        if not (
                            self.has_arrowhead_at(path[i - 1], vertex)
                            and self.has_arrowhead_at(path[i + 1], vertex)
                        ):
                            return False
                    # Collider must be ancestor of an endpoint
                    if self.has_arrowhead_at(
                        path[i - 1], vertex
                    ) and self.has_arrowhead_at(path[i + 1], vertex):
                        if path[0] not in self.get_descendants(vertex) and path[
                            -1
                        ] not in self.get_descendants(vertex):
                            return False
                return True
            visited.add(curr)
            neighbors = set(self.graph.successors(curr)) | set(
                self.graph.predecessors(curr)
            )
            for nbr in neighbors:
                if nbr not in visited:
                    if dfs(nbr, target, visited.copy(), path + [nbr]):
                        return True
            return False

        return dfs(u, v, set(), [u])
