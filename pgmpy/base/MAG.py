import networkx as nx

from pgmpy.base.AncestralBase import AncestralBase


class MAG(AncestralBase):

    def __init__(self, ebunch=None, latents=None):
        """
        Class for representing Maximal Ancestral Graphs (MAGs).

        Parameters
        ----------
        ebunch: list of tuples, optional (default: None)
            List of edges to initialize the graph. Each edge is represented
            as a tuple (u, v, u_mark, v_mark) where u_mark and v_mark can be
            '>' (arrowhead) or '-' (tail).
        latents: set, optional (default: None)
            Set of latent variables in the graph.
        """
        super().__init__()
        self.latents = latents if latents is not None else set()
        if ebunch is not None:
            for u, v, u_mark, v_mark in ebunch:
                self.add_edge(u, v, u_mark=u_mark, v_mark=v_mark)

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
            if len(path) > 2:
                for i in range(1, len(path) - 1):
                    node = path[i]
                    prev_node = path[i - 1]
                    next_node = path[i + 1]

                    if not self._is_collider(prev_node, node, next_node):
                        is_inducing = False
                        break

                    if not (
                        node in self.get_ancestors(u) or node in self.get_ancestors(v)
                    ):
                        is_inducing = False
                        break

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

    def is_invisible_edge(self, u, v):
        """
        Check if an edge (u, v) is invisible.

        An edge is invisible if it does not correspond to an inducing path.
        This is a complex property, often defined as an edge that's
        a result of marginalizing out variables. A simpler check is
        to see if no inducing path exists between u and v.
        """
        return not self.has_inducing_path(u, v, self.latents)

    def lower_manipulation(self, X: set):
        """
        Perform a lower manipulation on the MAG with respect to set X.

        This corresponds to marginalizing out variables in X.
        The effect is:
        - For every visible edge `u -> v` where `v` is in X, this edge is removed.
        - For every invisible edge `u - v` where `v` is in X, this edge is replaced
          by a bidirected edge `u <-> v`.
        """
        new_mag = MAG(ebunch=self.edges(), latents=self.latents)

        for u, v, data in new_mag.edges(data=True):
            if v in X:
                u_mark, v_mark = data["marks"]
                if u_mark == ">":
                    new_mag.remove_edge(u, v)
                elif u_mark == "-":
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

        raise NotImplementedError("This method is not yet implemented.")
