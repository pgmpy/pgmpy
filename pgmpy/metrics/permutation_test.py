#!/usr/bin/env python
import math
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators.CITests import ci_registry
from pgmpy.metrics import _BaseUnsupervisedMetric


class PermutationTest(_BaseUnsupervisedMetric):
    """
    Permutation-based test for falsifying causal graphs using observational data.

    For a given DAG, the test checks whether the DAG has fewer Local Markov Condition (LMC) violations than a baseline.
    The baseline has the same causal structure as the DAG. This baseline is chosen to be the node permutations of the
    given DAG. Fewer LMC violations mean that the DAG is more consistent/robust than a 'random' guess.

    The test performs two evaluations:
    1. Falsifiability: Whether the DAG is informative enough to be falsifiable. This is judged using the fraction of
    DAGs in the node perumations that are Markov equivalent to given DAG
    2. Falsification: Whether the given DAG performs significantly better than the node perumations, in terms of fewer
    LMC violations.

    Parameters
    ----------
    dag: pgmpy.base.DAG
        The causal graph to test.

    data : pandas.DataFrame
        Data to test the graph against. The column names of the DataFrame must match the variable names in the `model`.

    significance_level : float, default=0.05
        Significance level for conditional independence tests. Lower values make
        the test more conservative for accepting the null hypothesis.

    n_permutations : int, optional
        Number of random node permutations to generate for the baseline.
        If None, uses max(20, int(1/significance_level)).
        If -1, uses all possible permutations (factorial of number of nodes)

    ci_test : str, Instance of BaseCITest, optional
        link them here
        The statistical conditional independence test to use for evaluating the Local Markov Conditions in data.
        See :class:`pgmpy.estimators.CITests` for more details.

    return_summary : bool, default=False
        If True, returns detailed information about the test including
        individual LMC violations and permutation results.

    show_progress : bool, default=True
        Whether to show progress bar during permutation testing.

    Returns
    -------
    result : dict
        Dictionary containing test results with the following keys:

        - 'falsifiable' : bool
            Whether the graph is informative enough to be falsifiable.
            True if < significance_level fraction of permutations lie in same MEC.

        - 'falsified' : bool
            Whether the graph is falsified by the test.
            True if the graph is falsifiable AND performs significantly better than random.

        - 'p_value_falsifiable' : float
            P-value for the falsifiability test (fraction of permutations in same MEC).

        - 'p_value_falsified' : float
            P-value for the falsification test (fraction of permutations performing worse).

        - 'lmc_violations' : int
            Number of Local Markov Condition violations in the given graph.

        - 'summary' : dict (if return_summary=True)
            Detailed results including permutation violations and test statistics.

    References
    ----------
    .. [1] Eulig, E., Mastakouri, A. A., Blöbaum, P., Hardt, M., & Janzing, D. (2025). Toward falsifying causal graphs
    using a permutation-based test. Proceedings of the AAAI Conference on Artificial Intelligence, 39(25), 26778-26786.

    Examples
    --------
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.metrics import PermutationTest
    >>> from pgmpy.example_models import load_model

    >>> model = load_model("bnlearn/earthquake")
    >>> data = model.simulate(2000)
    >>> permutation_test = PermutationTest()
    >>> result_true = permutation_test.evaluate(
    ...     data, model, ci_test="pillai", n_permutations=-1
    ... )
    >>> print(
    ...     f"Falsifiable: {result_true['falsifiable']}, Falsified: {result_true['falsified']}"
    ... )

    >>> # Test with falsifiable wrong model
    >>> model.remove_edge("Earthquake", "Alarm")
    >>> model.add_edge("Burglary", "Earthquake")
    >>> result = permutation_test.evaluate(
    ...     data, model, ci_test="pillai", n_permutations=-1
    ... )
    >>> print(f"Falsifiable: {result['falsifiable']}, Falsified: {result['falsified']}")

    >>> # Test with not informative model not falsiable
    >>> wrong_model = DiscreteBayesianNetwork(
    ...     [("JohnCalls", "Earthquake"), ("Earthquake", "Burglary")]
    ... )
    >>> result_wrong = permutation_test.evaluate(
    ...     data, wrong_model, ci_test="pillai", n_permutations=-1
    ... )
    >>> print(f"Wrong model falsifiable: {result_wrong['falsifiable']}")
    """

    _tags = {
        "name": "permutation_test",
        "requires_true_graph": False,
        "requires_data": True,
        "lower_is_better": True,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
    }

    def __init__(
        self,
        n_permutations: int | None = None,
        significance_level: float = 0.05,
        ci_test: str | None = None,
        return_summary: bool = False,
        show_progress: bool = True,
        seed: int | None = None,
    ):
        self.n_permutations = n_permutations
        self.significance_level = significance_level
        self.ci_test = ci_test
        self.return_summary = return_summary
        self.show_progress = show_progress
        self.seed = seed
        super().__init__()

    def _get_parental_triples(self, dag):
        """
        Returns a list of (node, non_descendant, parents) triples for LMC/TPA validation.
        """
        triples = []
        for node in dag.nodes():
            parents = list(dag.predecessors(node))
            non_descendants = dag._get_non_descendants(node, exclude_parents=True)
            for nd in non_descendants:
                triples.append((node, nd, parents))
        return triples

    def _violations(self, permuted_dag, data, ci_test, significance_level, original_dag=None):
        """Calculate LMC and TPA violations."""
        triples = self._get_parental_triples(permuted_dag)
        n_lmc_violations = 0
        n_tpa_violations = 0

        # lmc violations
        for node, nd, parents in triples:
            res = ci_test(
                X=node,
                Y=nd,
                Z=parents,
                data=data,
                boolean=False,
                significance_level=significance_level,
            )
            pval = res[1]
            if pval <= significance_level:
                n_lmc_violations += 1

        # tpa violations
        if original_dag is not None:
            for node, nd, parents in triples:
                # Check d-separation in original DAG
                if original_dag.is_dconnected(node, nd, observed=parents):
                    n_tpa_violations += 1
        return n_lmc_violations, n_tpa_violations

    def _get_permutation_list(self, nodes, n_permutations, exclude_original_order=False):
        if self.seed is not None:
            np.random.seed(self.seed)
        if n_permutations == -1 or n_permutations >= math.factorial(len(nodes)):
            perms = list(permutations(nodes))
            if exclude_original_order:
                perms = [perm for perm in perms if list(perm) != list(nodes)]
            return perms
        else:
            perms = set()
            while len(perms) < n_permutations:
                perm = tuple(np.random.permutation(nodes))
                if exclude_original_order and perm == tuple(nodes):
                    continue
                perms.add(perm)
            return list(perms)

    def _evaluate(
        self,
        X: pd.DataFrame,
        causal_graph: DAG,
    ):

        # Step 0: Initialize variables and validate inputs.

        nodes = list(causal_graph.nodes())
        data_columns = set(X.columns)

        if not set(nodes).issubset(data_columns):
            missing_vars = set(nodes) - data_columns
            raise ValueError(f"Data missing variables present in model: {missing_vars}")

        if self.n_permutations is None:
            n_permutations = max(20, int(1 / self.significance_level))
        elif self.n_permutations == -1:
            n_permutations = math.factorial(len(nodes))

        ci_test = ci_registry.get_test(self.ci_test, data=X)
        permutation_violations = []
        tpa_violations = []
        n_within_mec = 0

        # Step 1: Compute LMC violations for the given DAG.

        n_lmc_violations, _ = self._violations(causal_graph, X, ci_test, self.significance_level)

        # Step 2: Generate permutations and compute LMC violations for each to construct null distribution.

        perm_list = self._get_permutation_list(nodes, n_permutations)
        if self.show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(perm_list, desc="Constructing Null Distribution")
        else:
            pbar = perm_list

        for permuted_nodes in pbar:
            perm_mapping = dict(zip(nodes, permuted_nodes))
            # Relabel nodes in the original DAG
            nx_permuted_dag = nx.relabel_nodes(causal_graph, perm_mapping, copy=True)
            permuted_dag = DAG()
            permuted_dag.add_nodes_from(nx_permuted_dag.nodes())
            permuted_dag.add_edges_from(nx_permuted_dag.edges())

            n_perm_lmc_violations, n_perm_tpa_violations = self._violations(
                permuted_dag, X, ci_test, self.significance_level, causal_graph
            )
            permutation_violations.append(n_perm_lmc_violations)
            tpa_violations.append(n_perm_tpa_violations)
            if n_perm_tpa_violations == 0:
                n_within_mec += 1

        # Step 3: Compute test statistics and p-values.

        # Step 3.1: Falsifiability test
        p_value_falsifiable = n_within_mec / n_permutations

        # Step 3.2: Falsification test
        count_less_violations = sum(1 for v in permutation_violations if v <= n_lmc_violations)
        p_value_falsified = count_less_violations / n_permutations

        # Step 3.3: Confidence intervals for the falsification p-value
        ci_lower, ci_upper = proportion_confint(
            count_less_violations,
            n_permutations,
            alpha=self.significance_level,
            method="wilson",
        )

        # Step 4: Compile results and return
        result = {
            "p_value_falsifiable": p_value_falsifiable,
            "p_value_falsified": p_value_falsified,
            "n_lmc_violations": n_lmc_violations,
            "n_within_mec": n_within_mec,
            "ci_lower_falsified": ci_lower,
            "ci_upper_falsified": ci_upper,
            "falsifiable": p_value_falsifiable <= self.significance_level,
            "falsified": (p_value_falsifiable <= self.significance_level)
            and (p_value_falsified >= self.significance_level),
        }

        if self.return_summary:
            result["summary"] = {
                "lmc_permutation_violations": permutation_violations,
                "tpa_permutation_violations": tpa_violations,
                "significance_level": self.significance_level,
                "ci_test": ci_test,
            }

        return result
