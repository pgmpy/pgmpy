import collections
import networkx as nx
from networkx import MultiDiGraph

class ADMG(MultiDiGraph):
    """
    Abstract class for an ADMG (Acyclic Directed Mixed Graph).
    An ADMG is a directed graph that may contain both directed and undirected edges.
    It is used to represent causal relationships in a system where some relationships are known to be directed,
    while others are not specified as directed or undirected.
    """
    def __init__(self, directed_ebunch, bidirected_ebunch, latents=None):
        """
        Initialize an ADMG with directed and bidirected edges.
        
        Parameters
        ----------
        directed_ebunch : list of tuples
            List of directed edges in the form (source, target).
        bidirected_ebunch : list of tuples
            List of bidirected edges in the form (node1, node2).
        latents : list, optional
            List of latent variables (nodes) in the graph.
        """
        super().__init__()
        self.bi_directed_edges = collections.defaultdict(set)
        self.latents = set(latents) if latents else set()

        if directed_ebunch:
            self.add_directed_edges(directed_ebunch)
        if bidirected_ebunch:
            self.add_bidirected_edges(bidirected_ebunch)
    
    def add_node(self, node):
        super().add_node(node)
    
    def add_nodes_from(self, nodes_for_adding, **attr):
        """add multiple nodes to the graph."""
        return super().add_nodes_from(nodes_for_adding, **attr)

    def add_directed_edge(self, u, v):
        """
        Adds a directed edge from node u to node v.
        Error raised if the edge additions would create a cycle in the graph.
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError("Both nodes must be present in the graph.")

        # Temporarily add edge using networkx's add_edge method
        key = self.add_edge(u, v)

        # Check for cycles using networkx's builtin function
        if not nx.is_directed_acyclic_graph(self):
            # If a cycle is detected, remove the edge and raise an error
            self.remove_edge(u, v, key=key)
            raise ValueError("Adding this edge would create a cycle in the graph.")

        # If no cycle is detected, then the edge has been added
    
    def add_directed_edges(self, u, v):
        """
        Adds a bidirected edge between nodes u and v.
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError("Both nodes must be present in the graph.")
        
        if u == v:
            raise ValueError("Cannot add a bidirected edge from a node to itself.")
        
        # add the bidirected edge in symmetry
        self.bi_directed_edges[u].add(v)
        self.bi_directed_edges[v].add(u)

    def add_directed_edges(self, ebunch):
        for u, v in ebunch:
            self.add_directed_edge(u, v)
    
    def add_bidirected_edges(self, ebunch):
        for u, v in ebunch:
            self.add_bidirected_edge(u, v)
    
    def add_edge(self, u, v, **attr):
        """
        Overrides the networkx add_edge method.
        Users should use the add_directed_edge method to add directed edges.
        """
        raise NotImplementedError("Use add_directed_edge to add directed edges.")

    def get_parents(self, nodes):
        """
        Returns the parents of a given node or a set of nodes.
        Uses the networkx method to get predecessors for directed edges,
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        parents = set()
        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            parents.update(super().predecessors(node))
        return parents
    
    def get_children(self, nodes):
        """
        Returns the children of a given node or a set of nodes.
        Uses the networkx method to get successors for directed edges.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        children = set()
        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            children.update(super().successors(node))
        return children

    def get_spouses(self, nodes):
        """
        Returns the spouses of a given node or a set of nodes.
        Spouses are defined as the union of parents and children.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        spouses = set()
        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            spouses.update(self.bi_directed_edges.get(node, set()))
        return spouses
    
    def get_ancestors(self, nodes):
        """
        Returns the ancestors of a given node or a set of nodes.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        ancestors = set()
        for node in nodes_set:
            if node in self.nodes:
                ancestors.update(nx.ancestors(self, node).union({node}))
        return ancestors

    def get_descendants(self, nodes):
        """
        Returns the descendants of a given node or a set of nodes.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        descendants = set()
        for node in nodes_set:
            if node in self.nodes:
                descendants.update(nx.descendants(self, node).union({node}))
        return descendants

    def get_district(self, nodes):
        """
        district(x) = {v | v <-> ... <-> x in ADMG or v = x}
        Returns the district of a given node or a set of nodes.
        If nodes is a set, returns union of districts for each node in the set.
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
                for spouse in self.get_spouses(currentNode):
                    if spouse not in visited:
                        visited.add(spouse)
                        queue.append(spouse)
            all_districts.update(district_components)
        return all_districts

# We will define the m-separation algorithm later, skipping it for now

    def to_dag(self):
        """
        Converts the ADMG to a Directed Acyclic Graph (DAG) by replacing
        each bidirected edge x <-> y with a latent node "L_{x}_{y}" and
        two directed edges x <- L_{x}_{y} -> y.
        """
        new_dag = nx.DiGraph()
        # Add all the observed nodes
        new_dag.add_nodes_from(self.nodes)

        # Add all the directed edges
        for u, v, _ in self.edges(key=True):
            if not self.has_edge(u, v):
                new_dag.add_edge(u, v)
        
        # Replace the bidirected edges with latent nodes
        processed_bidirected_pairs = set()
        for u, neighbors in self.bi_directed_edges.items():
            for v in neighbors:
                if (u, v) not in processed_bidirected_pairs and (v, u) not in processed_bidirected_pairs:
                    latent_node = f"L_{u}_{v}"
                    new_dag.add_node(latent_node)
                    new_dag.add_edge(u, latent_node)
                    new_dag.add_edge(latent_node, v)
                    processed_bidirected_pairs.add((u, v))
        
        return new_dag
    
    def get_ancestral_graph(self, nodes):
        """
        Returns an ADMG graph which would represent the ancestral structure of the given nodes.
        The induces subgraph H_A consists of all the nodes in A and all edges in H with both
        endpoints in A.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        if not nodes_set.issubset(self.nodes):
            raise ValueError("Input nodes must be subset of graph's nodes.")
        
        # An ancestral graph usually means it contains all the ancestors of the given nodes_set
        # But, the paper's definition of H_A is simply the graph induced by the nodes in A
        # So we take A -> nodes_set
        new_admg= ADMG(nodes = list(nodes_set))

        # add directed edges from the original graph
        for u, v in self.edges():
            if u in nodes_set and v in nodes_set:
                new_admg.add_directed_edge(u, v)
                # If the originagit l graph is ADMG, its subgraph will also be ADMG
        
        # add bidirected edges from the original graph
        for u in nodes_set:
            for v in self.bi_directed_edges.get(u, set()):
                if v in nodes_set and (u,v) not in new_admg.bi_directed_edges:
                    new_admg.add_bidirected_edges([(u, v)])
        
        return new_admg

        

            

        


