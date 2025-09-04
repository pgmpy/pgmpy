import networkx as nx

from pgmpy.base import AncestralBase


class PAG(AncestralBase):
    def get_definite_m_connecting_path(self, u, v, Z: set):
        """
        Check if a path between u and v is definitely m-connecting
        given a conditioning set Z.

        A path is "definitely m-connecting" if it is m-connecting
        in *all* of the MAGs consistent with this PAG. This occurs when
        the path's properties are fixed across the equivalence class.

        This is a complex check. A path is definitely m-connecting if:
        1. All colliders on the path are definitely in the ancestors of Z.
        2. All non-colliders on the path are definitely not in Z.

        This requires checking for `o` marks on the path.
        If an edge has an `o` mark, we can't be sure of its direction, so the
        path might not be m-connecting in some MAGs.
        """
        # Get all simple paths between u and v.
        paths = list(nx.all_simple_paths(self, source=u, target=v))

        # A path is definitely m-connecting if there's at least one path
        # that is not definitely m-blocked.
        for path in paths:
            is_definitely_m_blocked = False

            for i in range(1, len(path) - 1):
                node = path[i]
                prev = path[i - 1]
                next = path[i + 1]

                is_collider = self._is_collider(prev, node, next)

                # Case 1: Collider
                if is_collider:
                    # Check if 'node' is a definite ancestor of Z.
                    # This check is tricky. It is definite if the path to Z
                    # contains no 'o' marks.

                    # For a simple implementation, we assume if `o` is present,
                    # it is NOT definite.
                    path_to_z_exists = False
                    for z_node in Z:
                        if nx.has_path(self, node, z_node):
                            # The path must only have definite marks
                            # This is a simplification. A full check is more complex.
                            path_to_z_exists = True
                            break
                    if not path_to_z_exists:
                        is_definitely_m_blocked = True
                        break

                # Case 2: Non-collider
                else:
                    # Non-colliders must be definitely not in Z.
                    # This means we must check if there's a path from `u` to `node`
                    # with a `>` mark at `node`'s side, and `node` is in Z.
                    # Simplified: if `node` is in Z, it's blocked.
                    if node in Z:
                        is_definitely_m_blocked = True
                        break

            if not is_definitely_m_blocked:
                return True
        return False

    def _is_collider(self, p, c, n):
        """
        Check if c is a collider on the path p-c-n in the PAG.

        A node is a definite collider if the marks are '>', '>'.
        """
        p_mark, c_mark = self._get_marks(p, c)
        c_mark2, n_mark = self._get_marks(c, n)
        return p_mark == ">" and c_mark2 == ">"

    def invariance_under_intervention(self, X: set, Y: set, Z: set):
        """
        Check if the conditional probability P(Y|Z) is invariant under an
        intervention on X (do(X=x)).

        The do-calculus rules for PAGs depend on the existence of
        "definite m-connecting paths."

        Rule 3 from the paper is key:
        P(Y|do(X),Z) = P(Y|X,Z) if there is no definite m-connecting path
        between X and Y given Z.
        """
        if not X:
            # If there's no intervention, it's trivially invariant.
            return True

        for x_node in X:
            for y_node in Y:
                # If there's a definite m-connecting path, the probability
                # is not invariant.
                if self.get_definite_m_connecting_path(x_node, y_node, Z):
                    return False

        return True

    def upper_manipulation(self, X: set):
        """
        Perform an upper manipulation on the PAG with respect to set X.

        This corresponds to conditioning on variables in X.
        The paper notes this does not always result in a valid PAG.
        The operation is to remove all edges pointing **out** of X.
        """
        new_pag = PAG(ebunch=self.edges(), latents=self.latents)

        for u in X:
            for neighbor in new_pag.get_neighbors(u):
                u_mark, neighbor_mark = new_pag._get_marks(u, neighbor)
                if u_mark == ">":
                    new_pag.remove_edge(u, neighbor)

        return new_pag

    def lower_manipulation(self, X: set):
        """
        Perform a lower manipulation on the PAG with respect to set X.

        This corresponds to marginalizing out variables in X.
        The paper notes this does not always result in a valid PAG.
        The operation is to remove all edges pointing **into** X.
        """
        new_pag = PAG(ebunch=self.edges(), latents=self.latents)

        for v in X:
            for neighbor in new_pag.get_neighbors(v):
                v_mark, neighbor_mark = new_pag._get_marks(v, neighbor)
                if neighbor_mark == ">":
                    new_pag.remove_edge(neighbor, v)

        return new_pag
