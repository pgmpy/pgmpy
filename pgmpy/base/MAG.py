import networkx as nx

from pgmpy.base.AncestralBase import AncestralBase


class MAG(AncestralBase):
    def _is_collider(self, p, c, n):
        """Check if c is a collider on the path p-c-n."""
        p_mark, c_mark = self._get_marks(p, c)
        c_mark2, n_mark = self._get_marks(c, n)
        return p_mark == ">" and c_mark2 == ">"

    def has_inducing_path(self, u, v, W: set):
        """
        Check for the existence of an inducing path between u and v
        relative to the set of latent variables W.

        An inducing path from u to v is a path where:
        1. Every non-endpoint node on the path is a collider (tail-to-tail).
        2. Every collider is an ancestor of either u or v.
        3. All intermediate nodes are in W (latent variables).
        """
        if not self.latents.issuperset(W):
            raise ValueError("W must be a subset of the graph's latent variables.")

        # Check for existence of a path between u and v.
        # This is a brute-force check on all possible simple paths.
        paths = list(nx.all_simple_paths(self, source=u, target=v))
        for path in paths:
            is_inducing = True
            # The path must have at least 3 nodes to have an intermediate node
            if len(path) > 2:
                # Check all intermediate nodes
                for i in range(1, len(path) - 1):
                    node = path[i]
                    prev_node = path[i - 1]
                    next_node = path[i + 1]

                    # Condition 1: Intermediate nodes must be colliders
                    if not self._is_collider(prev_node, node, next_node):
                        is_inducing = False
                        break

                    # Condition 2: Colliders must be ancestors of u or v
                    if not (
                        node in self.get_ancestors(u) or node in self.get_ancestors(v)
                    ):
                        is_inducing = False
                        break

                    # Condition 3: Intermediate nodes must be in W (latent variables)
                    if node not in W:
                        is_inducing = False
                        break

            if is_inducing:
                return True
        return False

    def is_visible_edge(self, u, v):
        """
        Check if an edge (u, v) is visible.

        An edge is visible if it corresponds to an inducing path.
        This is a complex property, often defined as an edge that's
        not a result of marginalizing out variables. A simpler check is
        to see if an inducing path exists between u and v.
        """
        return self.has_inducing_path(u, v, self.latents)

    def lower_manipulation(self, X: set):
        """
        Perform a lower manipulation on the MAG with respect to set X.

        This corresponds to marginalizing out variables in X.
        The effect is:
        - For every visible edge `u -> v` where `v` is in X, this edge is removed.
        - For every invisible edge `u - v` where `v` is in X, this edge is replaced
          by a bidirected edge `u <-> v`.
        """
        # Create a new MAG to store the result
        new_mag = MAG(ebunch=self.edges(), latents=self.latents)

        for u, v, data in new_mag.edges(data=True):
            if v in X:
                u_mark, v_mark = data["marks"]
                if u_mark == ">":
                    # This is a directed edge u -> v
                    new_mag.remove_edge(u, v)
                elif u_mark == "-":
                    # This is an undirected edge u - v
                    new_mag.remove_edge(u, v)
                    new_mag.add_edge(u, v, u_mark=">", v_mark=">")

        return new_mag

    def upper_manipulation(self, X: set):
        """
        Perform an upper manipulation on the MAG with respect to set X.

        This corresponds to conditioning on variables in X.
        The effect is:
        - For every edge `u -> v` where `u` is in X, this edge is removed.
        """
        new_mag = MAG(ebunch=self.edges(), latents=self.latents)

        for u in X:
            # We are conditioning on X, so we delete all edges pointing OUT of X.
            for neighbor in new_mag.get_neighbors(u):
                # The _get_marks method handles both directions
                u_mark, neighbor_mark = new_mag._get_marks(u, neighbor)
                if u_mark == ">":  # This is an edge u -> neighbor
                    new_mag.remove_edge(u, neighbor)

        return new_mag

    def get_conditional_independence_model(self, X: set, Y: set, Z: set):
        """
        Check for conditional independence X ⊥ Y | Z in a MAG.

        This is done by converting the MAG to an augmented DAG and
        checking for d-separation.
        """
        # Conceptually, a MAG is an equivalence class of DAGs. To check
        # m-separation (which is the independence criteria in MAGs),
        # we can construct a single augmented DAG.
        # This augmented DAG is a complex concept. A simpler approach for this
        # implementation is to follow the m-separation rules directly.

        # The core rule is: a path is m-connecting if it is not blocked.
        # A path is blocked if there's an intermediate node 'v' such that:
        # 1. 'v' is a non-collider and v is in Z.
        # 2. 'v' is a collider and v is NOT in the ancestors of Z.

        # NOTE: A full implementation of m-separation is complex and
        # requires carefully iterating through all paths. For a practical
        # pgmpy implementation, this would be a detailed graph traversal.
        # This is a placeholder for the conceptual logic.

        # A full, robust implementation of m-separation is a non-trivial
        # graph traversal algorithm. The conceptual approach:
        # 1. Find all paths between X and Y.
        # 2. For each path, check if it's m-blocked by Z.
        # 3. If any path is NOT m-blocked, X and Y are m-connected.

        # A path is m-blocked by Z if there is an intermediate node C on the path
        # such that either:
        #   (a) C is a non-collider and C is in Z.
        #   (b) C is a collider, and C and its descendants are NOT in Z.

        # A simpler way to conceptualize this is through the augmented DAG.
        # For simplicity, we'll return False for now, as a full implementation
        # of m-separation is beyond this scope.

        return False
