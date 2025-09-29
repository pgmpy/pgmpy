import itertools
from typing import Hashable, Iterable

import networkx as nx

from pgmpy.base._mixin_roles import _GraphRolesMixin
from pgmpy.global_vars import logger


class PDAG(_GraphRolesMixin, nx.DiGraph):
    """
    Class for representing PDAGs (also known as CPDAG). PDAGs are the equivalence classes of
    DAGs and contain both directed and undirected edges.

    Note: In this class, undirected edges are represented using two edges in both direction i.e.
    an undirected edge between X - Y is represented using X -> Y and X <- Y.
    """

    def __init__(
        self,
        directed_ebunch: list[tuple[Hashable, Hashable]] = [],
        undirected_ebunch: list[tuple[Hashable, Hashable]] = [],
        latents: Iterable[Hashable] = [],
        roles=None,
    ):
        """
        Initializes a PDAG class.

        Parameters
        ----------
        directed_ebunch: list, array-like of 2-tuples
            List of directed edges in the PDAG.

        undirected_ebunch: list, array-like of 2-tuples
            List of undirected edges in the PDAG.

        latents: list, array-like
            List of nodes which are latent variables.

        roles : dict, optional (default: None)
            A dictionary mapping roles to node names.
            The keys are roles, and the values are role names (strings or iterables of str).
            If provided, this will automatically assign roles to the nodes in the graph.
            Passing a key-value pair via ``roles`` is equivalent to calling
            ``with_role(role, variables)`` for each key-value pair in the dictionary.

        Returns
        -------
        An instance of the PDAG object.

        Examples
        --------
        """
        self.directed_edges = set(directed_ebunch)
        self.undirected_edges = set(undirected_ebunch)

        super(PDAG, self).__init__(
            self.directed_edges.union(self.undirected_edges).union(
                set([(Y, X) for (X, Y) in self.undirected_edges])
            )
        )
        self.latents = set(latents)

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as a dictionary.")

        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    def all_neighbors(self, node):
        """
        Returns a set of all neighbors of a node in the PDAG. This includes both directed and undirected edges.

        Parameters
        ----------
        node: any hashable python object
            The node for which to get the neighboring nodes.

        Returns
        -------
        set: A set of neighboring nodes.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "C"), ("D", "C")],
        ...     undirected_ebunch=[("B", "A"), ("B", "D")],
        ... )
        >>> pdag.all_neighbors("A")
        {'B', 'C'}
        """
        return {x for x in self.successors(node)} | {x for x in self.predecessors(node)}

    def directed_children(self, node):
        """
        Returns a set of children of node such that there is a directed edge from `node` to child.
        """
        return {x for x in self.successors(node) if (node, x) in self.directed_edges}

    def directed_parents(self, node):
        """
        Returns a set of parents of node such that there is a directed edge from the parent to `node`.
        """
        return {x for x in self.predecessors(node) if (x, node) in self.directed_edges}

    def has_directed_edge(self, u, v):
        """
        Returns True if there is a directed edge u -> v in the PDAG.
        """
        if (u, v) in self.directed_edges:
            return True
        else:
            return False

    def has_undirected_edge(self, u, v):
        """
        Returns True if there is an undirected edge u - v in the PDAG.
        """
        if (u, v) in self.undirected_edges or (v, u) in self.undirected_edges:
            return True
        else:
            return False

    def undirected_neighbors(self, node):
        """
        Returns a set of neighboring nodes such that all of them have an undirected edge with `node`.

        Parameters
        ----------
        node: any hashable python object
            The node for which to get the undirected neighboring nodes.

        Returns
        -------
        set: A set of neighboring nodes.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "C"), ("D", "C")],
        ...     undirected_ebunch=[("B", "A"), ("B", "D")],
        ... )
        >>> pdag.undirected_neighbors("A")
        {'B'}
        """
        return {var for var in self.successors(node) if self.has_edge(var, node)}

    def is_adjacent(self, u, v):
        """
        Returns True if there is an edge between u and v. This can be either of u - v, u -> v, or u <- v.
        """
        if (u, v) in self.edges or (v, u) in self.edges:
            return True
        else:
            return False

    def copy(self):
        """
        Returns a copy of the object instance.

        Returns
        -------
        Copy of PDAG: pgmpy.dag.PDAG
            Returns a copy of self.
        """
        pdag = PDAG(
            directed_ebunch=list(self.directed_edges.copy()),
            undirected_ebunch=list(self.undirected_edges.copy()),
            latents=self.latents,
        )
        pdag.add_nodes_from(self.nodes())

        for role, vars in self.get_role_dict().items():
            pdag.with_role(role=role, variables=vars, inplace=True)
        return pdag

    def _directed_graph(self):
        """
        Returns a subgraph containing only directed edges.
        """
        dag = nx.DiGraph(self.directed_edges)
        dag.add_nodes_from(self.nodes())
        return dag

    def orient_undirected_edge(self, u, v, inplace=False):
        """
        Orients an undirected edge u - v as u -> v.

        Parameters
        ----------
        u, v: Any hashable python objects
            The node names.

        inplace: boolean (default=False)
            If True, the PDAG object is modified inplace, otherwise a new modified copy is returned.

        Returns
        -------
        None or pgmpy.base.PDAG: The modified PDAG object.
            If inplace=True, returns None and the object itself is modified.
            If inplace=False, returns a PDAG object.
        """

        if inplace:
            pdag = self
        else:
            pdag = self.copy()

        # Remove the edge for undirected_edges.
        if (u, v) in pdag.undirected_edges:
            pdag.undirected_edges.discard((u, v))
        elif (v, u) in pdag.undirected_edges:
            pdag.undirected_edges.discard((v, u))
        else:
            raise ValueError(f"Undirected Edge {u} - {v} not present in the PDAG.")

        # Remove the inverse edge from the graph
        pdag.remove_edge(v, u)

        # Add the edge to directed_edges.
        pdag.directed_edges.add((u, v))

        if not inplace:
            return pdag

    def _check_new_unshielded_collider(self, u, v):
        """
        Tests if orienting an undirected edge u - v as u -> v creates new unshielded V-structures in the PDAG.

        Checks whether v has any directed parents other than u that are not adjacent to u.

        Returns
        -------
        True, if the orientation u -> v would lead to creation of a new V-structure.
        False, if no new V-structures are formed.
        """
        for node in self.directed_parents(v):
            if (node != u) and (not self.is_adjacent(u, node)):
                return True
        return False

    def apply_meeks_rules(self, apply_r4=False, inplace=False, debug=False):
        """
        Applies the Meek's rules to orient the undirected edges of a PDAG to return a CPDAG.

        Parameters
        ----------
        apply_r4: boolean (default=False)
            If True, applies Rules 1 - 4 of Meek's rules.
            If False, applies only Rules 1 - 3.

        inplace: boolean (default=False)
            If True, the PDAG object is modified inplace, otherwise a new modified copy is returned.

        debug: boolean (default=False)
            If True, prints the rules being applied to the PDAG.

        Returns
        -------
        None or pgmpy.base.PDAG: The modified PDAG object.
            If inplace=True, returns None and the object itself is modified.
            If inplace=False, returns a PDAG object.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "B")], undirected_ebunch=[("B", "C"), ("C", "B")]
        ... )
        >>> pdag.apply_meeks_rules()
        >>> pdag.directed_edges
        {('A', 'B'), ('B', 'C')}
        """
        if inplace:
            pdag = self
        else:
            pdag = self.copy()

        changed = True
        while changed:
            changed = False

            # Rule 1: If X -> Y - Z and
            #            (X not adj Z) and
            #            (adding Y -> Z doesn't create cycle) and
            #            (adding Y -> Z doesn't create an unshielded collider) =>  Y → Z
            for y in pdag.nodes():
                # Select x's such that there are directed edges x -> y.
                for x in pdag.directed_parents(y):
                    for z in pdag.undirected_neighbors(y):
                        if (
                            (not pdag.is_adjacent(x, z))
                            and (not pdag._check_new_unshielded_collider(y, z))
                            and (not nx.has_path(pdag._directed_graph(), z, y))
                        ):
                            pdag.orient_undirected_edge(y, z, inplace=True)
                            changed = True
                            if debug:
                                logger.info(
                                    f"Applying Rule 1: {x} -> {y} - {z} => {x} -> {y} -> {z}"
                                )

            # Rule 2: If X -> Z -> Y  and X - Y =>  X → Y
            for z in pdag.nodes():
                xs = pdag.directed_parents(z)
                ys = pdag.directed_children(z)

                for x in xs:
                    for y in ys:
                        if pdag.has_undirected_edge(x, y):
                            pdag.orient_undirected_edge(x, y, inplace=True)
                            changed = True
                            if debug:
                                logger.info(
                                    f"Applying Rule 2: {x} -> {z} -> {y} and {x} - {y} => {x} -> {y}"
                                )

            # Rule 3: If X - {Y, Z, W} and {Z, Y} -> W => X -> W
            for x in pdag.nodes():
                undirected_nbs = pdag.undirected_neighbors(x)

                if len(undirected_nbs) < 3:
                    continue

                for y, z, w in itertools.permutations(undirected_nbs, 3):
                    if pdag.has_directed_edge(y, w) and pdag.has_directed_edge(z, w):
                        pdag.orient_undirected_edge(x, w, inplace=True)
                        changed = True
                        if debug:
                            logger.info(
                                f"Applying Rule 3: {x} - {y}, {z}, {w} "
                                f"{y}, {z} -> {w} => {x} -> {w}"
                            )
                        break

            # Rule 4: If d -> c -> b & a - {b, c, d} and b not adj d => a -> b
            if apply_r4:
                for c in pdag.nodes():
                    for b in pdag.directed_children(c):
                        for d in pdag.directed_parents(c):
                            if b == d or pdag.is_adjacent(b, d):
                                continue  # b adjacent d => rule not applicable

                            # find nodes a that are undirected neighbor to b, d,
                            #  and directed or undirected neighbor to c
                            cand = set(pdag.undirected_neighbors(b)).intersection(
                                pdag.all_neighbors(c),
                                pdag.undirected_neighbors(d),
                            )
                            for a in cand:
                                pdag.orient_undirected_edge(a, b, inplace=True)
                                changed = True
                                break
        if not inplace:
            return pdag

    def to_dag(self):
        """
        Returns one possible DAG which is represented using the PDAG.

        Returns
        -------
        pgmpy.base.DAG: Returns an instance of DAG.

        Examples
        --------
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "B"), ("C", "B")],
        ...     undirected_ebunch=[("C", "D"), ("D", "A")],
        ... )
        >>> dag = pdag.to_dag()
        >>> print(dag.edges())
        OutEdgeView([('A', 'B'), ('C', 'B'), ('D', 'C'), ('A', 'D')])

        References
        ----------
        [1] Dor, Dorit, and Michael Tarsi.
          "A simple algorithm to construct a consistent extension of a partially oriented graph."
            Technicial Report R-185, Cognitive Systems Laboratory, UCLA (1992): 45.
        """
        # Add required edges if it doesn't form a new v-structure or an opposite edge
        # is already present in the network.
        from pgmpy.base import DAG

        dag = DAG()
        # Add all the nodes and the directed edges
        dag.add_nodes_from(self.nodes())
        dag.add_edges_from(self.directed_edges)
        dag.latents = self.latents

        pdag = self.copy()
        while pdag.number_of_nodes() > 0:
            # find node with (1) no directed outgoing edges and
            #                (2) the set of undirected neighbors is either empty or
            #                    undirected neighbors + parents of X are adjacent
            found = False
            for X in sorted(pdag.nodes()):
                undirected_neighbors = pdag.undirected_neighbors(X)
                neighbors_are_adjacent = all(
                    (
                        pdag.has_edge(Y, Z) or pdag.has_edge(Z, Y)
                        for Z in pdag.all_neighbors(X)
                        for Y in undirected_neighbors
                        if not Y == Z
                    )
                )

                if not pdag.directed_children(X) and (
                    not undirected_neighbors or neighbors_are_adjacent
                ):
                    found = True
                    # add all edges of X as outgoing edges to dag
                    for Y in pdag.undirected_neighbors(X):
                        dag.add_edge(Y, X)
                    pdag.remove_node(X)
                    break

            if not found:
                logger.warning(
                    "PDAG has no faithful extension (= no oriented DAG with the "
                    + "same v-structures as PDAG). Remaining undirected PDAG edges "
                    + "oriented arbitrarily."
                )
                for X, Y in pdag.edges():
                    if not dag.has_edge(Y, X):
                        try:
                            dag.add_edge(X, Y)
                        except ValueError:
                            pass
                break
        return dag

    def to_graphviz(self) -> object:
        """
        Retuns a pygraphviz object for the DAG. pygraphviz is useful for
        visualizing the network structure.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model("alarm")
        >>> model.to_graphviz()
        <AGraph <Swig Object of type 'Agraph_t *' at 0x7fdea4cde040>>
        """
        return nx.nx_agraph.to_agraph(self)

    def __eq__(self, other):
        """
        Checks if two PDAGs are equal. Two PDAGs are considered equal if they
        have the same nodes, edges, latent variables, and variable roles.

        Parameters
        ----------
        other: PDAG object
            The other PDAG to compare with.

        Returns
        -------
        bool
            True if the PDAGs are equal, False otherwise.
        """
        if not isinstance(other, PDAG):
            return False

        return (
            set(self.nodes()) == set(other.nodes())
            and set(self.directed_edges) == set(other.directed_edges)
            and set(self.undirected_edges) == set(other.undirected_edges)
            and self.latents == other.latents
            and self.get_role_dict() == other.get_role_dict()
        )
