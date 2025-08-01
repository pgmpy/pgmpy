## FCI Algorithm

FCI algorithm for causal discovery in the presence of latent confounders and selection bias
It outputs a PAG (Partial Ancestral Graph) which represents the causal relationships consistent with the observed conditional independence tests.

#### Methods to include in the FCI class
1) `init`: Which will initialize the algorithm
2) `build_skeleton` : Which will build the initial skeleton (we can invoke this method from PC algorithm)
3) `orient_colliders ` : The nodes which from a v-shaped structure would be identified and hence removed from the initial causal structure

#### Applying the orientation rules

- There are 10 rules which are iteratively applied to the initial causak structure 
(according to the zhang 2008) to the current PAG until no more edges can be oriented.
These rules involve complex logic to handle definite and possible causal relationships, especially in the presence of latent variables and selection bias,
using information from discriminating paths.

4) `is_unshielded_triple` : Checks for unshielded triples in the graph. An unshielded triple looks something like X-Y-Z where Y is adjacent to both X and Z but X and Z are not adjacent to each other.

5) `d-seperation` and related methods

6) `update_edge_marks` : This is the function which needs to be called after we form the final PAG, after applying all the orientation rules.

7) `estimate` : This executes the FCI algorithm to learn about the causal structure from the data and outputs the PAG (Partial Ancestral Graph)