__authors__ = ["srhmm", "Nimish-4", "ankurankan"]

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.structure_score import BaseStructureScore, get_scoring_method


class TOPIC(_BaseCausalDiscovery):
    """The TOPIC algorithm for causal discovery / structure learning.

    This class implements the TOPIC algorithm [1] for causal discovery. Given a tabular dataset, TOPIC estimates the
    causal structure among the variables in the data in a Directed Acyclic Graph (DAG). The algorithm works by
    establishing a topological ordering among the variables using a local scoring criterion, and in the process adds and
    respectively prunes directed edges among the variables that are consistent with the topological ordering.

    Parameters
    ----------

    scoring_method : str or BaseStructureScore instance
        The score to be optimized during structure estimation.  Supported structure scores: k2, bdeu, bds, bic-d, aic-d,
        ll-g, aic-g, bic-g, ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should be an instance of
        `BaseStructureScore`.

    return_type : str (default: "dag")
        The type of structure to return. Can be one of: `dag`, `pdag`. TOPIC by default orients all edges and returns a
        fully directed structure.

        - If `return_type=dag`, a fully directed structure (i.e., DAG) is returned.
        - If `return_type=pdag`: the (fully) directed structure is converted to a PDAG instance.

    min_improvement : float, default=1e-6
        The minimal score improvement used for edge addition and removal.
        If the structure_score specified in `scoring_method` is not an MDL score but BIC, AIC, etc., this
        `min_improvement` will be used to check whether score differences are large enough to be considered sufficient
        for edge  addition and  removal.

    show_progress : bool, default=False
        If True, shows a progress bar while learning the causal structure.

    Attributes
    ----------
    causal_graph_ : :class:`~pgmpy.base.DAG` or :class: `~pgmpy.base.PDAG`
        The learned causal graph.

        - If `return_type="dag"`, this will be a DAG instance.
        - If `return_type="pdag"`, this will be a PDAG instance.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/ecoli70")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the TOPIC algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery.TOPIC import TOPIC
    >>> topic = TOPIC()
    >>> _ = topic.fit(df)
    >>> edges = sorted(topic.causal_graph_.edges())
    >>> len(edges) > 0
    True

    References
    ----------
    .. [1] Xu, S., Mameche, S. and Vreeken, J. Information-Theoretic Causal Discovery in Topological Order.
           International Conference on Artificial Intelligence and Statistics (AISTATS), 2025.

    """

    def __init__(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        return_type: str = "dag",
        min_improvement: float = 1e-6,
        show_progress: bool = False,
    ):
        self.return_type = return_type
        self.scoring_method = scoring_method
        self.min_improvement = min_improvement
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame):
        """The fitting procedure for the TOPIC algorithm.

        Parameters
        ----------
        X: pd.DataFrame
            The input dataset.
        """
        # Step 0: Initialize scoring method and data structures
        score = get_scoring_method(scoring_method=self.scoring_method, data=X)

        dag_current = DAG()
        dag_current.add_nodes_from(list(X.columns))
        candidates = list(dag_current.nodes)
        topological_order_ = []

        # Step 1: Find the topological order for the variables.
        pbar = tqdm(
            range(self.n_features_in_),
            desc="Topological order",
            unit="node",
            disable=not (self.show_progress and config.SHOW_PROGRESS),
        )

        for _ in pbar:
            # Step 1.1: Find the next source node in topological order.
            # Step 1.1.1: Builds a pair-wise improvement matrix where entry [i, j] is the score gain of adding
            # candidates[i] -> candidates[j] to dag_current.

            n = len(candidates)
            score_improvement = np.zeros((n, n))
            for j, effect in enumerate(candidates):
                base_parents = tuple(dag_current.get_parents(effect))
                old_score = score.local_score(effect, base_parents)
                for i, cause in enumerate(candidates):
                    if i == j:
                        continue
                    score_improvement[i, j] = score.local_score(effect, base_parents + (cause,)) - old_score

            # Step 1.1.2: Source is the candidate with the smallest maximum incoming improvement (i.e. the least
            # preferred sink).
            delta = score_improvement - score_improvement.T
            np.fill_diagonal(delta, -np.inf)
            source = candidates[np.argmin(delta.max(axis=0))]
            candidates.remove(source)
            topological_order_.append(source)

            pbar.set_description(f"Processing: {source}")
            pbar.set_postfix_str(f"remaining={len(candidates)}")

            # Step 1.2: Add edges from the source to remaining candidates if they improve the score sufficiently.
            for node in candidates:
                current_parents = tuple(dag_current.get_parents(node))
                gain = score.local_score(node, current_parents + (source,)) - score.local_score(node, current_parents)
                if gain > self.min_improvement:
                    dag_current.add_edge(source, node)

            # Step 1.3: Prune edges into the source. Repeatedly remove the parent whose removal least decreases the
            # score, using min_improvement as the tolerance (symmetric with the edge-addition threshold above).
            # The last remaining parent is never removed.
            current_parents = list(dag_current.get_parents(source))
            while len(current_parents) > 1:
                old_score = score.local_score(source, tuple(current_parents))
                best_parent = None
                best_harm = float("-inf")
                for parent in current_parents:
                    new_parents = tuple(p for p in current_parents if p != parent)
                    harm = score.local_score(source, new_parents) - old_score
                    if harm >= -self.min_improvement and harm > best_harm:
                        best_harm = harm
                        best_parent = parent
                if best_parent is None:
                    break
                dag_current.remove_edge(best_parent, source)
                current_parents.remove(best_parent)

        # Step 2: Store the learned causal graph and related attributes.
        if self.return_type == "dag":
            self.causal_graph_ = dag_current
        elif self.return_type == "pdag":
            self.causal_graph_ = dag_current.to_pdag()
        else:
            raise ValueError(f"return_type must be one of: dag, pdag, got {self.return_type}")

        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_)
        self.topological_order_ = topological_order_

        return self
