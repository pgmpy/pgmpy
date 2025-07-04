import collections
from typing import Hashable, Optional, Sequence
import networkx as nx
from networkx import MultiDiGraph

from pgmpy.base.DAG import DAG as pgmpy_DAG


class ADMG(MultiDiGraph):
    """
    Abstract class for an ADMG (Acyclic Directed Mixed Graph).
    An ADMG is a directed graph that may contain both directed and undirected edges.
    It is used to represent causal relationships in a system where some relationships are known to be directed,
    while others are not specified as directed or undirected.
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
        super().add_node(node)

    def add_nodes_from(self, nodes, **attr):
        return super().add_nodes_from(nodes, **attr)

    def _add_directed_edge(self, u, v):
        if u is None or v is None:
            raise ValueError("Can't add since one of nodes is None")

        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)

        key = super().add_edge(u, v, type="directed")

        if not nx.is_directed_acyclic_graph(self):
            super().remove_edge(u, v, key=key)
            raise ValueError("Adding this edge would create a cycle in the graph.")

    def _add_bidirected_edge(self, u, v):
        if u is None or v is None:
            raise ValueError("Can't add since one of~ the nodes is None")
        if u == v:
            raise ValueError("Cannot add a bidirected edge from a node to itself.")

        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)

        # Add two directed edges with a 'type' attribute indicating bidirected
        key_uv = super().add_edge(u, v, type="bidirected")
        key_vu = super().add_edge(v, u, type="bidirected")

        # To ensure consistency, you might want to store the keys or manage them.
        # For simplicity in this example, we'll assume the user won't directly
        # manipulate these keys without going through the ADMG methods.

    def add_directed_edges(self, ebunch):
        for u, v in ebunch:
            self._add_directed_edge(u, v)

    def add_bidirected_edges(self, ebunch):
        for u, v in ebunch:
            self._add_bidirected_edge(u, v)

    def add_edge(self, u, v, **attr):
        raise NotImplementedError(
            "Use add_directed_edge or add_bidirected_edge to add edges."
        )

    def _get_parent(self, node):
        """
        Internal method to get the parent of a node.
        This is used to ensure that the parent is always a directed edge.
        """
        if node not in self.nodes:
            raise ValueError(f"Node {node} is not in the graph.")

        parents = set()
        for pred in self.predecessors(node):
            data = self.get_edge_data(pred, node)
            for key in data:
                if data[key].get("type") == "directed":
                    parents.add(pred)
        return parents

    def get_parents(self, nodes):
        """
        Get the parents of the given nodes in the ADMG.
        Returns a tuple of two sets: (parents, district_parents).
        - parents: Direct parents (directed edges).
        - district_parents: Parents connected by bidirected edges.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        parents = set()
        district_parents = set()

        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            # Get direct parents
            direct_parents = self._get_parent(node)
            parents.update(direct_parents)

            # district parents are those which are district as well as parents
            for parent in direct_parents:
                # Check if the parent is connected by a bidirected edge
                for neighbor in super().neighbors(parent):
                    if (
                        self.has_edge(parent, neighbor)
                        and self.get_edge_data(parent, neighbor, 0).get("type")
                        == "bidirected"
                    ) or (
                        self.has_edge(neighbor, parent)
                        and self.get_edge_data(neighbor, parent, 0).get("type")
                        == "bidirected"
                    ):
                        district_parents.add(parent)

        return parents, district_parents

    def get_children(self, nodes):
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
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        if not nodes_set.issubset(self.nodes):
            raise ValueError("Input nodes must be subset of graph's nodes.")

        # Create a new ADMG instance for the ancestral graph
        new_admg = ADMG()
        new_admg.add_nodes_from(list(nodes_set))  # Add all nodes in nodes_set

        # Add directed edges from the original graph that have both endpoints in nodes_set
        for u, v, key, data in self.edges(keys=True, data=True):
            if data.get("type") == "directed" and u in nodes_set and v in nodes_set:
                new_admg._add_directed_edge(
                    u, v
                )  # Use _add_directed_edge to maintain cycle check

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
                        new_admg._add_bidirected_edge(u, v)
                        processed_bidirected_pairs.add((u, v))
                        processed_bidirected_pairs.add(
                            (v, u)
                        )  # Mark both directions as processed

        return new_admg

    def get_markov_blanket(self, nodes):
        nodes_set = {nodes} if isinstance(nodes, set) else set(nodes)
        if not nodes_set.issubset(self.nodes):
            raise ValueError("Input nodes must be subset of graph's nodes.")
        markov_blanket = set()
        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            # Get parents
            parents, district_parents = self.get_parents(node)
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
        Converts the ADMG to a directed acyclic graph (DAG).
        For each bidirected x<->y in the ADMG, add a latent node {x}_{y}
        and add two edgws x<----{x}_{y}-->y
        """
        dag_edges = []
        dag_nodes = set()

        # Add directed edges
        for u, v, data in self.edges(data=True):
            if data.get("type") == "directed":
                dag_edges.append((u, v))
                dag_nodes.update([u, v])

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
                    dag_nodes.add(latent_var)
                    dag_nodes.update(sorted_pair)

        # Create a new DAG instance
        dag_instance = pgmpy_DAG()
        dag_instance.add_nodes_from(dag_nodes)
        dag_instance.add_edges_from(dag_edges)

        return dag_instance

    def _is_d_connected_internal(
        self,
        start: "Hashable",
        end: "Hashable",
        observed: Optional[Sequence["Hashable"]] = None,
        include_latents=False,
    ):
        """
        Internal helper to check d-connection on the **transformed** DAG.
        This method will operate on the DAG obtained by converting the ADMG
        to a directed acyclic graph (DAG) using the `to_dag` method.
        """
        new_dag = self.to_dag()

        if new_dag.is_dconnected(start, end, observed=observed):
            return True
        else:
            return False

    def is_m_separated(
        self,
        nodes_u,
        nodes_v,
        conditional_set=None,
    ):
        """
        Check if nodes_u and nodes_v are m-separated given conditional_set in the ADMG.
        This method uses the d-connection check on the transformed DAG.
        """
        if conditional_set is None:
            conditional_set = set()

        # Convert nodes_u and nodes_v to sets
        nodes_u_set = {nodes_u} if isinstance(nodes_u, str) else set(nodes_u)
        nodes_v_set = {nodes_v} if isinstance(nodes_v, str) else set(nodes_v)

        for u in nodes_u_set:
            for v in nodes_v_set:
                # if they are d_connected, they must also be m_connected
                if self._is_d_connected_internal(u, v, observed=conditional_set):
                    return False

        return True

    def is_m_connected(
        self,
        nodes_u,
        nodes_v,
        conditional_set=None,
    ):
        """
        Checks if two sets of nodes are m_connected given a conditional set.
        """
        return not self.is_m_separated(nodes_u, nodes_v, conditional_set)

    def m_connected_nodes(self, nodes_u, nodes_v, conditional_set=None):
        """
        Finds all nodes that are m-connected to any node in `nodes_u`
        given the `conditional_set`.
        """
        if conditional_set is None:
            conditional_set = set()

        if not isinstance(nodes_u, list):
            nodes_u = [nodes_u]

        m_connected_set = set()

        new_dag = self.to_dag()

        # Iterate over all the original ADMG nodes to find their connections
        for node in self.nodes:
            # Node always connected to itself
            if node in nodes_u:
                m_connected_set.add(node)
                continue

            for node in nodes_u:
                if new_dag.is_dconnected(node, node, observed=conditional_set):
                    m_connected_set.add(node)
                    break

        return m_connected_set
