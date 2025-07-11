import collections
from typing import Hashable, Optional, Sequence
import networkx as nx
from networkx import MultiDiGraph

from pgmpy.base.DAG import DAG as pgmpy_DAG


class ADMG(MultiDiGraph):
    """
    A class representing an Acyclic Directed Mixed Graph (ADMG).

    An ADMG is a directed graph that allows for both directed and bidirected edges.
    This class extends the `networkx.MultiDiGraph` and provides additional functionality
    for operations involving directed and bidirected edges.

    Parameters
    ----------
    directed_ebunch : list of tuple, optional
        List of directed edges to initialize the graph, where each tuple is (u, v).
    bidirected_ebunch : list of tuple, optional
        List of bidirected edges to initialize the graph, where each tuple is (u, v).
    latents : set of str, optional
        Set of latent variables in the graph. These are not directly represented as nodes
        but are used to indicate the presence of bidirected edges.
    """

    def __init__(self, directed_ebunch=None, bidirected_ebunch=None, latents=None):
        super().__init__()
        # Using edge attributes to distinguish bidirected edges
        self.latents = set(latents) if latents else set()

        if directed_ebunch:
            self.add_directed_edges(directed_ebunch)
        if bidirected_ebunch:
            self.add_bidirected_edges(bidirected_ebunch)

    def add_node(self, node):
        """
        Adds a node to the ADMG from the MultiDiGraph class.
        """
        super().add_node(node)

    def add_nodes_from(self, nodes, **attr):
        """
        Adds multiple nodes to the graph.

        Parameters
        ----------
        nodes : iterable
            An iterable of nodes to add.
        """
        return super().add_nodes_from(nodes, **attr)

    def add_directed_edges(self, ebunch):
        """
        Adds directed edges (u -> v) to the ADMG.

        Parameters
        ----------
        ebunch : list of tuple
            List of directed edges, where each tuple is (u, v).
        """
        for u, v in ebunch:
            if u is None or v is None:
                raise ValueError("Can't add since one of nodes is None")

            key = super().add_edge(u, v, type="directed")
            if not nx.is_directed_acyclic_graph(self):
                super().remove_edge(u, v, key=key)
                raise ValueError("Adding this edge would create a cycle in the graph.")

    def add_bidirected_edges(self, ebunch):
        """
        Adds bidirected edges (u <-> v) to the ADMG.

        Parameters
        ----------
        ebunch : list of tuple
            List of bidirected edges, where each tuple is (u, v).
        """
        for u, v in ebunch:
            if u is None or v is None:
                raise ValueError("Can't add since one of the nodes is None")
            if u == v:
                raise ValueError("Cannot add a bidirected edge from a node to itself.")

            # Add two directed edges with a 'type' attribute indicating bidirected
            super().add_edge(u, v, type="bidirected")
            super().add_edge(v, u, type="bidirected")

    def add_edge(self, u, v, **attr):
        """
        Raises an error if trying to add a regular edge.
        """
        raise NotImplementedError(
            "Use add_directed_edge or add_bidirected_edge to add edges."
        )

    def get_directed_parents(self, nodes):
        """
        Get directed parents of given nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes to query.

        Returns
        -------
        set
            Set of directed parents.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        directed_parents = set()

        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            for pred in self.predecessors(node):
                data = self.get_edge_data(pred, node)
                for key in data:
                    if data[key].get("type") == "directed":
                        directed_parents.add(pred)
        return directed_parents

    def get_bidirected_parents(self, nodes):
        """
        Get bidirected parents (nodes connected via bidirected edge) of the given nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes to query.

        Returns
        -------
        set
            Set of bidirected parents.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        bidirected_parents = set()

        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            # Get neighbors and check for bidirected edges
            for neighbor in super().neighbors(node):
                if (
                    self.has_edge(node, neighbor)
                    and self.get_edge_data(node, neighbor, 0).get("type")
                    == "bidirected"
                ) or (
                    self.has_edge(neighbor, node)
                    and self.get_edge_data(neighbor, node, 0).get("type")
                    == "bidirected"
                ):
                    bidirected_parents.add(neighbor)

        return bidirected_parents

    def get_children(self, nodes):
        """
        Get children of given nodes (i.e., targets of outgoing directed edges).

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of children nodes.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        children = set()
        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            for successor in super().successors(node):
                # Only consider truly directed edges
                if self.get_edge_data(node, successor, 0)["type"] == "directed":
                    children.add(successor)
        return children

    def get_spouses(self, nodes):
        """
        Get spouses of given nodes (i.e., nodes connected via bidirected edges).

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of spouses.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        spouses = set()
        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            for neighbor in super().neighbors(node):
                # Check if the edge to/from the neighbor is bidirected
                if (
                    self.has_edge(node, neighbor)
                    and self.get_edge_data(node, neighbor, 0).get("type")
                    == "bidirected"
                ) or (
                    self.has_edge(neighbor, node)
                    and self.get_edge_data(neighbor, node, 0).get("type")
                    == "bidirected"
                ):
                    spouses.add(neighbor)
        return spouses

    def get_ancestors(self, nodes):
        """
        Get ancestors of given nodes via directed paths.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of ancestor nodes including the input nodes.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        ancestors = set()
        for node in nodes_set:
            if node in self.nodes:
                # Use a temporary graph containing only directed edges for ancestry
                temp_dag = nx.DiGraph()
                for u, v, key, data in self.edges(keys=True, data=True):
                    if data.get("type") == "directed":
                        temp_dag.add_edge(u, v)
                if node in temp_dag:  # Ensure node exists in the temp_dag
                    ancestors.update(nx.ancestors(temp_dag, node).union({node}))
        return ancestors

    def get_descendants(self, nodes):
        """
        Get descendants of given nodes via directed paths.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of descendant nodes including the input nodes.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        descendants = set()
        for node in nodes_set:
            if node in self.nodes:
                # Use a temporary graph containing only directed edges for descendants
                temp_dag = nx.DiGraph()
                for u, v, key, data in self.edges(keys=True, data=True):
                    if data.get("type") == "directed":
                        temp_dag.add_edge(u, v)
                if node in temp_dag:  # Ensure node exists in the temp_dag
                    descendants.update(nx.descendants(temp_dag, node).union({node}))
        return descendants

    def get_district(self, nodes):
        """
        Return district of a node: maximal set connected via bidirected edges.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Nodes in the same bidirected-connected component.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        all_districts = set()

        for start_node in nodes_set:
            if start_node not in self.nodes:
                continue

            district_components = set()
            queue = collections.deque([start_node])
            visited = {start_node}

            while queue:
                currentNode = queue.popleft()
                district_components.add(currentNode)
                # Iterate through all neighbors and check for bidirected edges
                for neighbor in super().neighbors(currentNode):
                    if (
                        self.has_edge(currentNode, neighbor)
                        and self.get_edge_data(currentNode, neighbor, 0).get("type")
                        == "bidirected"
                    ) or (
                        self.has_edge(neighbor, currentNode)
                        and self.get_edge_data(neighbor, currentNode, 0).get("type")
                        == "bidirected"
                    ):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                for predecessor in super().predecessors(currentNode):
                    if (
                        self.has_edge(currentNode, predecessor)
                        and self.get_edge_data(currentNode, predecessor, 0).get("type")
                        == "bidirected"
                    ) or (
                        self.has_edge(predecessor, currentNode)
                        and self.get_edge_data(predecessor, currentNode, 0).get("type")
                        == "bidirected"
                    ):
                        if predecessor not in visited:
                            visited.add(predecessor)
                            queue.append(predecessor)

            all_districts.update(district_components)
        return all_districts

    def get_ancestral_graph(self, nodes):
        """
        Return the ancestral graph induced by the input nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes to induce subgraph on.

        Returns
        -------
        ADMG
            Subgraph induced by ancestors of the given nodes.

        Raises
        ------
        ValueError
            If any input node is not in the graph.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        if not nodes_set.issubset(self.nodes):
            raise ValueError("Input nodes must be subset of graph's nodes.")

        # Create a new ADMG instance for the ancestral graph
        new_admg = ADMG()
        new_admg.add_nodes_from(list(nodes_set))  # Add all nodes in nodes_set

        # Add directed edges from the original graph that have both endpoints in nodes_set
        for u, v, key, data in self.edges(keys=True, data=True):
            if data.get("type") == "directed" and u in nodes_set and v in nodes_set:
                new_admg.add_directed_edges(
                    [(u, v)]
                )  # Use add_directed)edges to maintain cycle check

        # Add bidirected edges from the original graph that have both endpoints in nodes_set
        processed_bidirected_pairs = set()
        for u, v, key, data in self.edges(keys=True, data=True):
            if data.get("type") == "bidirected":
                if u in nodes_set and v in nodes_set:
                    # Ensure we add each bidirected pair only once in the new graph
                    if (u, v) not in processed_bidirected_pairs and (
                        v,
                        u,
                    ) not in processed_bidirected_pairs:
                        new_admg.add_bidirected_edges([(u, v)])
                        processed_bidirected_pairs.add((u, v))
                        processed_bidirected_pairs.add(
                            (v, u)
                        )  # Mark both directions as processed

        return new_admg

    def get_markov_blanket(self, nodes):
        """
        Compute the Markov blanket for the given node(s).

        Includes:
        - Parents
        - Children
        - Spouses (nodes sharing a child)
        - Parents of nodes in the district

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of nodes in the Markov blanket.
        """
        nodes_set = {nodes} if isinstance(nodes, set) else set(nodes)
        if not nodes_set.issubset(self.nodes):
            raise ValueError("Input nodes must be subset of graph's nodes.")
        markov_blanket = set()
        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            # Get parents
            parents = self.get_directed_parents(node)
            district_parents = self.get_bidirected_parents(node)
            markov_blanket.update(parents)
            markov_blanket.update(district_parents)
            # Get children
            children = self.get_children(node)
            markov_blanket.update(children)
            # Get spouses
            spouses = self.get_spouses(node)
            markov_blanket.update(spouses)
        return markov_blanket

    def to_dag(self):
        """
        Project ADMG into a DAG by introducing latent variables for bidirected edges.

        Returns
        -------
        pgmpy.base.DAG.DAG
            DAG with latent variables replacing bidirected edges.
        """
        dag_edges = []

        # Add directed edges
        for u, v, data in self.edges(data=True):
            if data.get("type") == "directed":
                dag_edges.append((u, v))

        # add latent nodes and edges for bidirected edges
        latent_nodes_map = {}
        for u, v, data in self.edges(data=True):
            if data.get("type") == "bidirected":
                sorted_pair = tuple(sorted((u, v)))
                if sorted_pair not in latent_nodes_map:
                    latent_var = f"L_{sorted_pair[0]}_{sorted_pair[1]}"
                    latent_nodes_map[sorted_pair] = latent_var
                    dag_edges.append((latent_var, sorted_pair[0]))
                    dag_edges.append((latent_var, sorted_pair[1]))

        dag_nodes = set(self.nodes()) | set(latent_nodes_map.values())

        # Create a new DAG instance
        dag_instance = pgmpy_DAG()
        dag_instance.add_nodes_from(dag_nodes)
        dag_instance.add_edges_from(dag_edges)

        return dag_instance

    def is_mseparated(
        self,
        nodes_u,
        nodes_v,
        conditional_set=None,
    ):
        """
        Test m-separation between two sets of nodes given a conditioning set.

        Parameters
        ----------
        nodes_u : str or iterable of str
            First set of nodes.

        nodes_v : str or iterable of str
            Second set of nodes.

        conditional_set : set of str, optional
            Conditioning set (default is empty set).

        Returns
        -------
        bool
            True if nodes_u and nodes_v are m-separated; False otherwise.
        """
        if conditional_set is None:
            conditional_set = set()

        # Convert nodes_u and nodes_v to sets
        nodes_u_set = {nodes_u} if isinstance(nodes_u, str) else set(nodes_u)
        nodes_v_set = {nodes_v} if isinstance(nodes_v, str) else set(nodes_v)

        new_dag = self.to_dag()
        for u in nodes_u_set:
            for v in nodes_v_set:
                # if they are dconnected, they are not mseparated
                if new_dag.is_dconnected(u, v, observed=conditional_set):
                    return False
        return True

    def is_mconnected(
        self,
        nodes_u,
        nodes_v,
        conditional_set=None,
    ):
        """
        Test m-connectedness between two node sets given a conditioning set.

        Parameters
        ----------
        nodes_u : str or iterable of str
            First set of nodes.

        nodes_v : str or iterable of str
            Second set of nodes.

        conditional_set : set of str, optional
            Conditioning set.

        Returns
        -------
        bool
            True if m-connected; False if m-separated.
        """
        return not self.is_mseparated(nodes_u, nodes_v, conditional_set)

    def mconnected_nodes(self, nodes_u, nodes_v=None, conditional_set=None):
        """
        Find all nodes m-connected to nodes in `nodes_u` given `conditional_set`.

        Parameters
        ----------
        nodes_u : str or iterable of str
            Set of source nodes.

        nodes_v : str or iterable of str, optional
            If provided, filters the result to this set.

        conditional_set : set of str, optional
            Conditioning set (default is empty set).

        Returns
        -------
        set
            Nodes m-connected to `nodes_u` (or their intersection with `nodes_v` if provided).
        """
        if conditional_set is None:
            conditional_set = set()

        dag = self.to_dag()
        if isinstance(nodes_u, str):
            nodes_u = [nodes_u]

        m_connected_set = set()

        for node in nodes_u:
            active_nodes = dag.active_trail_nodes(node, observed=conditional_set)
            active_nodes = {n for n in active_nodes if not str(n).startswith("L_")}
            m_connected_set.update(active_nodes)

        if nodes_v is not None:
            nodes_v_set = {nodes_v} if isinstance(nodes_v, str) else set(nodes_v)
            return m_connected_set & nodes_v_set

        return m_connected_set
