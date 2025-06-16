## ADMG readme section

Paper Link : https://sci-hub.se/https://doi.org/10.1111/1467-9469.00323

### Methods defined for this class

#### Graph Representation and Management
These features will help to represent the graph, structure its construction

1) Edge Types:
* Directed Edges:
- Method to add directed edge from source node to destination node (u ----> v)
- Internal Storage for directed edges (maybe an adjacency list)

* Bidirected Edges
- Method to add a bidirected edge between 2 nodes (u <-------> v)
- Internal Storage for bi-directed edges (an adjacency list or mapping)

#### Graph Traversal and Relationship Discovery
These methods help us query the graph for its structural relationships
Assuming a node **x**, we define the following functions!
1) Directed Relationships
- get_parents(x)
- get_children(x)
- get_spouses(x)

2) Indirect Relationships
- get_ancestors(x)
- get_descendents(x)

#### Conditional Independence (m-separation)
This will help us determine the conditional independence relationships.
1) Path Analysis and Collider / Non-collider identification
A function , given a path, will return whether an intermediate node is a collider or a non-collider.
2) m-separation algorithm
- Main public method
- Determines if a set of vertices (X_set) is **m-separated** from another set of vertices (Y_set)

