import networkx as nx
import numpy as np

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import SHD


class SelfCompatibilityScore:
    """
    Computes the graphical self compatibility score of `estimator` on `data`.

    The graphical self-compatibility measure [1] defines how consistent the
    learned subgraphs are with the learned model structure. This can be used
    as a proxy measure for the SHD between the learned and true graph.

    Parameters
    ----------
    estimator: class
        The causal discovery algorithm to use. This can be any class
        inheriting pgmpy.estimators.StructureEstimator.

    num_subsets : int (default=50)
        The number of random subsets of variables to draw.

    subset_fraction : float (default=0.8)
        The fraction of variables to sample in each sampled subset.

    seed: int or None (default: None)
        The seed value for the random number generator.

    estimator_kwargs: kwargs
        Additional arguments for the `estimator.estimate` method.

    Returns
    -------
    float: Self compatibility score.
        Mean SHD between latent-projected joint and marginal DAGs.

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.estimators import HillClimbSearch
    >>> from pgmpy.metrics import self_compatibility_graphical
    >>> model = get_example_model("cancer")
    >>> df = model.simulate(1000)
    >>> score = self_compatibility_graphical(
    ...     HillClimbSearch,
    ...     df,
    ...     num_subsets=3,
    ...     subset_fraction=0.5,
    ...     scoring_method="bic-d",
    ... )
    >>> print(score)

    References
    ----------
    [1] Faller, P. M., et al. (2024). “Self-compatibility: Evaluating Causal
        Discovery without Ground Truth.” In AISTATS. arXiv:2307.09552
    """

    _tags = {
        "name": "self_compatibility_score",
        "requires_true_graph": False,
        "requires_data": True,
        "lower_is_better": False,
        "is_default": False,
    }

    def __init__(
        self,
        num_subsets: int = 50,
        subset_fraction: float = 0.8,
        seed: int | None = None,
        **estimator_kwargs,
    ):
        self.num_subsets = num_subsets
        self.subset_fraction = subset_fraction
        self.seed = seed
        self.estimator_kwargs = estimator_kwargs

    def _latent_admg(self, dag: DAG, observed: list) -> nx.DiGraph:
        """
        Compute the latent-projection ADMG L(G, observed_set) of a DAG G onto a subset observed_set, V.
        (Helper function for self_compatibility_graphical)

        Implements Definition 5 (latent ADMG) from [1] (Faller et al., AISTATS 2024)

        Parameters
        ----------
        dag : DAG
            The full DAG G on variables V (may include latent nodes).

        observed : list
            Subset observed_set ⊂ V to project onto (observed variables).

        Returns
        -------
        nx.DiGraph:
            An ADMG over the observed nodes, where:
                - X -> Y encodes a latent-only directed chain X ->…-> Y
                - X <-> Y is encoded by having both X -> Y and Y -> X in the graph.

        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.metrics.metrics import _latent_admg
        >>> # Example 1: A -> H -> B, observe only A,B
        >>> dag1 = DiscreteBayesianNetwork([("A", "H"), ("H", "B")])
        >>> admg1 = _latent_admg(dag1, observed=["A", "B"])
        >>> set(admg1.edges()) == {("A", "B")}
        True

        >>> # Example 2: Latent confounder H -> A and H -> B, observe only A,B
        >>> dag2 = DiscreteBayesianNetwork([("H", "A"), ("H", "B")])
        >>> admg2 = _latent_admg(dag2, observed=["A", "B"])
        >>> set(admg2.edges()) == {("A", "B"), ("B", "A")}
        True

        >>> # Example 3: Unshielded collider with extension A -> B <- C -> D, observe only A,C
        >>> dag3 = DiscreteBayesianNetwork([("A", "B"), ("C", "B"), ("C", "D")])
        >>> admg3 = _latent_admg(dag3, observed=["A", "C"])
        >>> set(admg3.edges()) == set()
        True

        >>> # Example 4: Pure confounding A -> B and A -> C, observe only B,C
        >>> dag4 = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
        >>> admg4 = _latent_admg(dag4, observed=["B", "C"])
        >>> set(admg4.edges()) == {("B", "C"), ("C", "B")}
        True

        References
        ----------
        [1] Faller, P. M., et al. (2024).
        “Self-compatibility: Evaluating Causal Discovery without Ground Truth.”
        In AISTATS. arXiv:2307.09552
        """
        # pdb.set_trace()
        observed_set = set(observed)
        latent_set = set(dag.nodes()) - observed_set
        full_directed = dag

        directed_edges = set()
        bidirected_edges = set()

        # Building a latent only subgraph H
        H = nx.DiGraph()
        for u, v in dag.edges():
            if u in latent_set or v in latent_set:
                H.add_edge(u, v)
        # now explicitly add all observed and latent nodes so
        # that calls to has_path do not crash on missing nodes.
        H.add_nodes_from(observed_set)  # (add only nodes, no edges)
        H.add_nodes_from(latent_set)

        # 1) Directed‐chain detection
        for u in observed_set:
            # find all observed v reachable via any active trail
            reachable = dag.active_trail_nodes([u], observed=[], include_latents=True)[u]
            for v in (reachable & observed_set) - {u}:
                # check if we already added a u->v?
                if (u, v) in directed_edges:
                    continue

                # a) look for u->…->v via latents
                if nx.has_path(H, u, v):
                    # ensure that v does not have any other observed parents
                    parents_v = {p for p in dag.predecessors(v) if p in observed_set}
                    if parents_v.issubset({u}):
                        directed_edges.add((u, v))
                        continue

                # b) similarly, now we look for u->…->v via latents
                if nx.has_path(H, v, u):
                    parents_u = {p for p in dag.predecessors(u) if p in observed_set}
                    if parents_u.issubset({v}):
                        directed_edges.add((v, u))
                        continue

                # c) if neither direction gives a directed chain then try bidirected
                #  (i) latent common‐parent?
                for latent in latent_set:
                    if dag.has_edge(latent, u) and dag.has_edge(latent, v):
                        bidirected_edges.add((u, v))
                        bidirected_edges.add((v, u))
                        break
                else:
                    # (ii) shared observed child w via latent‐only chains?
                    for w in observed_set - {u, v}:
                        sub_uw = full_directed.subgraph(latent_set | {u, w})
                        sub_vw = full_directed.subgraph(latent_set | {v, w})
                        if nx.has_path(sub_uw, u, w) and nx.has_path(sub_vw, v, w):
                            bidirected_edges.add((u, v))
                            bidirected_edges.add((v, u))
                            break

        # 2) Preserve any original observed -> observed edges
        for x, y in dag.edges():
            if x in observed_set and y in observed_set:
                directed_edges.add((x, y))

        # 3) Assemble final ADMG
        admg = PDAG()
        admg.add_nodes_from(observed_set)
        admg.add_edges_from(directed_edges)
        admg.add_edges_from(bidirected_edges)
        return admg

    def evaluate(self, X, estimator):
        # Step 0: Initialize variables
        rng = np.random.RandomState(self.seed)
        subset_size = max(3, int(self.subset_fraction * len(X.columns)))
        shd_values = []

        # Step 1: Learn the model structure on all the variables.
        full_structure = estimator(**self.estimator_kwargs).fit(X).causal_graph_
        if isinstance(full_structure, PDAG):
            full_structure = full_structure.to_dag()

        # Step 2: Iterate over subset of variables and learn structure on them.
        for i in range(self.num_subsets):
            # Step 2.1: Select a subset of variables and data.
            observed_vars = rng.choice(X.columns, size=subset_size, replace=False)
            sub_data = X.loc[:, observed_vars]

            # Step 2.2: Learn the structure on subset of variables and compute the
            #           marginal graph from the full graph.
            structure_proj = self._latent_admg(full_structure, observed_vars)
            marginal_structure = estimator(**self.estimator_kwargs).fit(sub_data).causal_graph_

            # Step 2.3: Compute the SHD between learned and marginal graph.
            shd_values.append(SHD().evaluate(structure_proj, marginal_structure))

        # Step 3: Return the average SHD.
        return float(np.mean(shd_values)) if shd_values else 0.0
