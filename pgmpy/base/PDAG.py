import itertools

import networkx as nx

from pgmpy import logger
from pgmpy.base._base import _CoreGraph


class PDAG(_CoreGraph):
    """
    Class for representing PDAGs (also known as CPDAGs). PDAGs are the equivalence classes of DAGs
    and contain both directed (``"->"``) and undirected (``"--"``) edges.

    Built on :class:`~pgmpy.base._base._CoreGraph`, restricted to directed/undirected edges via
    ``SUPPORTED_EDGE_TYPES``.

    Parameters
    ----------
    edge_list : iterable of tuples, optional
        Edges of the form ``(u, v, edge_type)`` with ``edge_type`` one of ``"->"``, ``"<-"``, ``"--"``.

    latents : set, default=set()
        Set of latent (unobserved) variables.

    exposures, outcomes : set, default=set()
        Treatment / response variables (causal-analysis roles).

    roles : dict, optional (default: None)
        A mapping of role name to node(s); equivalent to calling ``with_role`` for each entry.

    Examples
    --------
    >>> from pgmpy.base import PDAG
    >>> pdag = PDAG(edge_list=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--")], latents=["A"])
    >>> sorted(pdag.directed_edges)
    [('A', 'C'), ('D', 'C')]
    >>> sorted(pdag.undirected_edges)
    [('A', 'B')]
    >>> pdag.latents
    {'A'}
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "--"])

    @property
    def directed_edges(self) -> set[tuple]:
        """The set of directed edges ``(u, v)`` (meaning ``u -> v``)."""
        return {(u, v) for u, v, _ in self.get_edges(edge_types={"->"})}

    @property
    def undirected_edges(self) -> set[tuple]:
        """The set of undirected edges, each as a sorted ``(u, v)`` tuple."""
        return {tuple(sorted((u, v))) for u, v, _ in self.get_edges(edge_types={"--"})}

    def chain_component(self, node) -> set:
        """Returns the chain component of `node`: all nodes reachable through undirected (``"--"``) edges."""
        return self.get_reachable_nodes(node, "--")

    def is_clique(self, nodes) -> bool:
        """Whether every pair of `nodes` is joined by an undirected (``"--"``) edge."""
        return all(self.has_edge(u, v, "--") for u, v in itertools.combinations(nodes, 2))

    def has_semidirected_path(self, source, target, blocked_nodes=None, ignore_direct_edge=False) -> bool:
        """
        Returns True if there is a semi-directed path from `source` to `target`, i.e. a path that
        follows directed edges forward and undirected edges in either direction.

        Parameters
        ----------
        source, target : Hashable
            The endpoints of the path.

        blocked_nodes : iterable, optional
            Nodes not allowed on the path.

        ignore_direct_edge : bool
            If True, ignore the direct ``source -> target`` step.
        """
        blocked_nodes = set() if blocked_nodes is None else set(blocked_nodes)
        if (source in blocked_nodes) or (target in blocked_nodes):
            return False
        if source == target:
            return True
        if not (self.has_node(source) and self.has_node(target)):
            return False

        # Forward directed (`->`) edges and undirected (`--`) edges, both read from `node`'s endpoint.
        visited = {source}
        stack = [source]
        while stack:
            node = stack.pop()
            for child in self.get_neighbors(node, {"->", "--"}):
                if child in visited or child in blocked_nodes:
                    continue
                if ignore_direct_edge and node == source and child == target:
                    continue
                if child == target:
                    return True
                visited.add(child)
                stack.append(child)
        return False

    def orient_undirected_edge(self, u, v, inplace=False):
        """
        Orients an undirected edge ``u - v`` as ``u -> v``.

        Parameters
        ----------
        u, v : Hashable
            The node names.

        inplace : bool (default: False)
            If True, modify this PDAG and return None; otherwise return a modified copy.
        """
        pdag = self if inplace else self.copy()
        if not pdag.has_edge(u, v, "--"):
            raise ValueError(f"Undirected Edge {u} - {v} not present in the PDAG.")
        pdag.replace_edge(u, v, "--", "->")
        if not inplace:
            return pdag

    def has_acyclic_extension(self) -> bool:
        """Returns True if the PDAG admits an acyclic DAG extension."""
        if not self.undirected_edges:
            return not self.has_directed_cycle()
        return nx.is_directed_acyclic_graph(self.to_dag())

    def apply_meeks_rules(self, apply_r4=False, inplace=False, debug=False):
        """
        Applies Meek's rules to orient the undirected edges of the PDAG, returning a (maximally
        oriented) CPDAG.

        Parameters
        ----------
        apply_r4 : bool (default: False)
            If True, applies Rules 1-4; otherwise only Rules 1-3.

        inplace : bool (default: False)
            If True, modify this PDAG and return None; otherwise return a modified copy.

        debug : bool (default: False)
            If True, log the rules being applied.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(edge_list=[("A", "B", "->"), ("B", "C", "--")])
        >>> pdag = pdag.apply_meeks_rules()
        >>> sorted(pdag.directed_edges)
        [('A', 'B'), ('B', 'C')]
        """
        pdag = self if inplace else self.copy()

        changed = True
        while changed:
            changed = False

            # Rule 1: X -> Y - Z, X not adj Z, no cycle, no new unshielded collider  =>  Y -> Z
            for y in pdag.nodes():
                for x in pdag.get_parents(y):
                    for z in pdag.get_neighbors(y, "--"):
                        if (
                            (not pdag.has_edge(x, z))
                            and (not pdag._check_new_unshielded_collider(y, z))
                            and (not nx.has_path(pdag.get_directed_graph(), z, y))
                        ):
                            pdag.orient_undirected_edge(y, z, inplace=True)
                            changed = True
                            if debug:
                                logger.info(f"Applying Rule 1: {x} -> {y} - {z} => {x} -> {y} -> {z}")

            # Rule 2: X -> Z -> Y and X - Y  =>  X -> Y
            for z in pdag.nodes():
                for x in pdag.get_parents(z):
                    for y in pdag.get_children(z):
                        if pdag.has_edge(x, y, "--"):
                            pdag.orient_undirected_edge(x, y, inplace=True)
                            changed = True
                            if debug:
                                logger.info(f"Applying Rule 2: {x} -> {z} -> {y} and {x} - {y} => {x} -> {y}")

            # Rule 3: X - {Y, Z, W} and {Z, Y} -> W  =>  X -> W
            for x in pdag.nodes():
                undirected_nbs = pdag.get_neighbors(x, "--")
                if len(undirected_nbs) < 3:
                    continue
                for y, z, w in itertools.permutations(undirected_nbs, 3):
                    if (
                        pdag.has_edge(y, w, "->")
                        and pdag.has_edge(z, w, "->")
                        and not pdag.has_edge(y, z)  # the two parents of w must be non-adjacent
                        and not nx.has_path(pdag.get_directed_graph(), w, x)  # x->w must not close a cycle
                    ):
                        pdag.orient_undirected_edge(x, w, inplace=True)
                        changed = True
                        if debug:
                            logger.info(f"Applying Rule 3: {x} - {y}, {z}, {w} and {y}, {z} -> {w} => {x} -> {w}")
                        break

            # Rule 4: d -> c -> b, a - {b, c, d}, b not adj d  =>  a -> b
            if apply_r4:
                for c in pdag.nodes():
                    for b in pdag.get_children(c):
                        for d in pdag.get_parents(c):
                            if b == d or pdag.has_edge(b, d):
                                continue
                            cand = pdag.get_neighbors(b, "--").intersection(
                                pdag.get_neighbors(c),
                                pdag.get_neighbors(d, "--"),
                            )
                            for a in cand:
                                if nx.has_path(pdag.get_directed_graph(), b, a):
                                    continue  # orienting a -> b would close a directed cycle
                                pdag.orient_undirected_edge(a, b, inplace=True)
                                changed = True
                                break
        if not inplace:
            return pdag

    def to_cpdag(self):
        """Returns the CPDAG corresponding to one DAG extension of the PDAG."""
        from pgmpy.base import DAG

        if self.undirected_edges:
            dag = self.to_dag()
        else:
            dag = DAG()
            dag.add_nodes_from(self.nodes())
            dag.add_edges_from(self.directed_edges)
            dag.latents = self.latents

        cpdag = dag.to_pdag()
        for role, variables in self.get_role_dict().items():
            cpdag.with_role(role=role, variables=variables, inplace=True)
        return cpdag

    def to_dag(self):
        """
        Returns one possible DAG represented by this PDAG.

        Returns
        -------
        pgmpy.base.DAG

        References
        ----------
        - :cite:p:`dor_tarsi_1992`
        """
        from pgmpy.base import DAG

        dag = DAG()
        dag.add_nodes_from(self.nodes())
        dag.add_edges_from(self.directed_edges)
        dag.latents = self.latents

        pdag = self.copy()
        while pdag.number_of_nodes() > 0:
            # Find a node with no directed outgoing edge whose undirected neighbours are either empty
            # or whose undirected neighbours + neighbours are mutually adjacent.
            found = False
            for x in sorted(pdag.nodes()):
                undirected_neighbors = pdag.get_neighbors(x, "--")
                neighbors_are_adjacent = all(
                    pdag.has_edge(y, z) for z in pdag.get_neighbors(x) for y in undirected_neighbors if not y == z
                )

                if not pdag.get_children(x) and (not undirected_neighbors or neighbors_are_adjacent):
                    found = True
                    for y in pdag.get_neighbors(x, "--"):
                        dag.add_edge(y, x)
                    pdag.remove_node(x)
                    break

            if not found:
                logger.warning(
                    "PDAG has no faithful extension (= no oriented DAG with the same v-structures as "
                    "PDAG). Remaining undirected PDAG edges oriented arbitrarily."
                )
                for x, y in pdag.get_edges(data=False):
                    if dag.has_edge(x, y) or dag.has_edge(y, x):
                        continue
                    try:
                        dag.add_edge(x, y)
                    except ValueError:
                        # x -> y would close a cycle; the reverse is acyclic, so keep the edge as y -> x
                        dag.add_edge(y, x)
                break
        return dag
