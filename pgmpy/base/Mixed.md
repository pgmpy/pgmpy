## Mixed Graph Reademe Section

### Ancestral Edge
This class represents a single edge at either MAG or PAG. It stores the two connected nodes
along with the edge mark type at each end.

#### Utility Functions
1) `is_directed_from_u_to_v`
2) `is_directed_from_u_to_v`
3) `is_bidirected`
4) `has_circle_at`


### AncestralBase Graph
This is an abstract base class that provides a common functionalities for both **MAGs** and **PAGs** classes.
It handles the edges, the AncestralEdge types and also an internal **DAG** structure for efficient query handling.

#### Utility Functions
1) `add_node`
2) `add_nodes_from`
3) `add_edge` : Adds an Ancestral Edge to the graph and also updates the adjacency list
4) `get_edge` : Retrieves the ancestral object between 2 specified nodes

#### Directed and Undirected Relationsips
1) `get_neighbors` : Returns the set of all the nodes directly connected to the specified node.
2) `get_parents` : Returns the set of all the nodes that have a directed edge pointing towards the specified node
3) `get_children` : Returns the set of all the nodes into which the specified node is pointing
4) `get_spouses` : Returns the set of all the nodes connected to the specified node by a bidirected edge
5) `get_ancestors` : Returns the ancestors of the specified node. We can leverage the internal DAG method of pgmpy
6) `get_descendents` : Returns the descendants of the specified node. We can leverage the internal DAG method of pgmpy

#### Quering the types of edges from the graph
1) `_get_all_simple_paths(self, start, end)`:
A private helper method that finds all the simple paths between the start and end nodes using networkx. Returns a list of all the paths
2) `is_collider(self, path_nodes, node)`:
Check if a node is a collider on a given path sequence.



