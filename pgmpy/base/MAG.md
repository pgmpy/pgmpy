## MAG Readme Section
This class extends the AncestralGraphBase and implements the functionalities specific to Maximal Ancestral Graphs.
Including their defining properties and manipulation rules.

### Methods
1) is_inducing_path(self, path_nodes, L_set)
Checks if a given path_nodes sequence is an inducing path relative to a set of latent variables L_set

2) `is_maximal` : Checks if the MAG satisfies the "maximal" property:
There are no inducing paths between the non-adjacent vertices

3) `is_m_connecting_path`(self, path_nodes, Z_set)
Checks if a path_nodes sequence is m-connecting (active) relative to
conditioning set Z_set (Definition 2)

4) `is_m_separated`(self, X_set, Y_set, Z_set)
Checks if the 2 sets of nodes are m-separated given set Z_set

5) `is_visible_edge`(self, u, v)
Checks if a directed edge u---->v in the MAG is "visible" (Definition 8). This involves checking for a third node **C** that is not adjacent to **V** and has specific connections to **U**.


