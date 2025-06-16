import collections
import networkx as nx
from networkx import DiGraph

class ADMG(DiGraph):
    """
    Abstract class for an ADMG (Acyclic Directed Mixed Graph).
    An ADMG is a directed graph that may contain both directed and undirected edges.
    It is used to represent causal relationships in a system where some relationships are known to be directed,
    while others are not specified as directed or undirected.
    """

    def __init__(self, nodes=None):
        """Initializes a set of nodes"""
        super().__init__(self.nodes)

        if nodes is None:
            self.add_nodes_from(nodes)
        
        self.bidirected_edges = collections.defaultdict(set)
    
    ### Graph Representation and Management Methods ###
    def add_directed_edges(self, u, v):
        """
        Returns a directed edge from u to v it it exists.
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError("Both nodes must be part of the graph.")
    
        # Add the edge between u and v
        super().add_edge(u, v)

        if not nx.is_directed_acyclic_graph(self):
            super().remove_edge(u, v)
            raise ValueError("The graph must remain acyclic after adding the edge.")

    def add_bidirected_edges(self, u, v):
        """
        Adds a bidirected edge between nodes u and v.
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError("Both nodes must be part of the graph.")
        
        if u == v:
            raise ValueError("Cannot add a bidirected edge from a node to itself.")
        
        # Add the bidirected edge
        self.bidirected_edges[u].add(v)
        self.bidirected_edges[v].add(u)
    

    ### Relationship Query and Graph traversal ###

    # Direct Relations
    def get_parents(self, node):
        """Returns the set of direct parents of a node (predecessors in networkx)."""
        if node not in self.nodes:
            return set()
        return set(super().predecessors(node))

    def get_children(self, node):
        """Returns the set of direct children of a node (successors in networkx)."""
        if node not in self.nodes:
            return set()
        return set(super().successors(node))
    
    def get_spouses(self, node):
        """Returns the set of direct spouses of a node (via bi-directed edges)."""
        if node not in self.nodes:
            return set()
        return self.bi_directed_edges.get(node, set())
    
    # Indirect Relations
    def get_ancestors(self, node):
        """
        Finds all nodes u such that there is a directed path u -> ... -> node, including node itself.
        Leverages networkx's ancestor function for directed part.
        """
        if node not in self.nodes:
            return set()
        return nx.ancestors(self, node).union({node})

    def get_descendants(self, node):
        """
        Finds all nodes u such that there is a directed path node -> ... -> u, including node itself.
        Leverages networkx's descendant function for directed part.
        """
        if node not in self.nodes:
            return set()
        return nx.descendants(self, node).union({node})

    ### m - separation logic ###
    def _is_m_connecting(self, path_nodes, path_edges, Z_set, Z_ancestors):
        """
        Checks if a specified path is m-connecting given Z_set.
        """
        if(len(path_nodes) < 3):
            return True # A path with 1 or 2 cant be blocked by colliders
        
        for i in range(1, len(path_nodes) -1):
            current_node = path_nodes[i]
            prev_edge = path_edges[i - 1]
            next_edge = path_edges[i]
            
            is_collider = False

            if prev_edge[2] == 'directed' and prev_edge[1] == current_node and \
               next_edge[2] == 'directed' and next_edge[0] == current_node:
                is_collider = True
            
            elif prev_edge[2] == 'bidirected' and next_edge[2] == 'bidirected':
                is_collider = True
            
            elif prev_edge[2] == 'directed' and prev_edge[1] == current_node and next_edge[2] == 'bidirected':
                is_collider = True

            elif prev_edge[2] == 'bidirected' and next_edge[2] == 'directed' and next_edge[0] == current_node:
                is_collider = True
            
            if is_collider:
                if current_node not in Z_ancestors:
                    return False

            else: # Non-collider
                if current_node in Z_set:
                    return False
        
        return True
    
    def is_m_connected(self,X_set, Y_set, Z_set):
        """
        Checks if u and v are m-connected given Z_set.
        Uses networkx to find all paths and checks each path for m-connection.
        """

        u = set(u)
        v = set(v)
        Z_set = set(Z_set)

        Z_ancestors = set()
        for z in Z_set:
            Z_ancestors.update(self.get_ancestors(z))

        for x in X_set:
            for y in Y_set:
                if x==y:
                    continue

                if x not in self.nodes or y not in self.nodes:
                    raise ValueError("Both nodes must be part of the graph.")
                paths = nx.all_simple_paths(self, source=x, target=y)

                for path in paths:
                    path_nodes = path
                    path_edges = []

                    for i in range(len(path_nodes) - 1):
                        u = path_nodes[i]
                        v = path_nodes[i + 1]
                        edge_type = 'directed' if self.has_edge(u, v) else 'bidirected' if v in self.bidirected_edges[u] else None
                        if edge_type is not None:
                            path_edges.append((u, v, edge_type))

                    if not self._is_m_connecting(path_nodes, path_edges, Z_set, Z_ancestors):
                        return False
        return True