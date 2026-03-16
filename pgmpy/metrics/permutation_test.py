#!/usr/bin/env python
import math
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators.CITests import ci_registry
from pgmpy.models import DynamicBayesianNetwork


def _get_parental_triples(dag):
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


def _lmc_violations(dag, data, ci_test, significance_level):
    """Validate the local markov condition for a given directed graph. Return number of violations."""
    triples = _get_parental_triples(dag)
    n_violations = 0
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
            n_violations += 1
    return n_violations, len(triples)


def _tpa_violations(permuted_dag, original_dag):
    """
    Evaluate which pairwise parental d-separations (parental triples) in `permuted_dag` are
    violated assuming `original_dag` is the ground truth DAG.
    """
    triples = _get_parental_triples(permuted_dag)
    n_violations = 0
    for node, nd, parents in triples:
        # Check d-separation in original DAG
        if original_dag.is_dconnected(node, nd, observed=parents):
            n_violations += 1
    return n_violations, len(triples)


def permutation_test(
    dag: DAG,
    data: pd.DataFrame,
    significance_level: float = 0.05,
    n_permutations: Optional[int] = None,
    ci_test: Optional[str] = None,
    return_summary: bool = True,
    show_progress: bool = True,
):
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

    ci_test : {"pillai_trace", "chi_square", "pearsonr", "gcm", "g_sq", "log_likelihood", "freeman_tuckey",
    "modified_log_likelihood", "neyman", "cressie_read"}

        The statistical conditional independence test to use for evaluating the Local Markov Conditions in data.
        See :class:`pgmpy.estimators.CITests` for more details.

    return_summary : bool, default=True
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
    >>> from pgmpy.metrics import permutation_test
    >>> from pgmpy.utils import get_example_model

    >>> # Test with a known model
    >>> model = get_example_model("cancer")
    >>> data = model.simulate(1000)
    >>> result = permutation_test(model, data)
    >>> print(
    ...     f"Falsifiable: {result['p_value_falsifiable']}, Falsified: {result['p_value_falsified']}"
    ... )

    >>> # Test with wrong model (should be falsified)
    >>> wrong_model = DiscreteBayesianNetwork(
    ...     [("Cancer", "Smoker"), ("Smoker", "Pollution")]
    ... )
    >>> result_wrong = permutation_test(wrong_model, data)
    >>> print(f"Wrong model falsified: {result_wrong['p_value_falsified']}")
    """
    # Step 0: Initialize variables and validate inputs.
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Data should be a pandas DataFrame. Got: {type(data)}")

    if not isinstance(dag, DAG) or isinstance(dag, DynamicBayesianNetwork):
        raise TypeError(f"DAG must be a `pgmpy.base.DAG` object. Got: {type(dag)}")

    nodes = list(dag.nodes())
    data_columns = set(data.columns)

    if not set(nodes).issubset(data_columns):
        missing_vars = set(nodes) - data_columns
        raise ValueError(f"Data missing variables present in model: {missing_vars}")

    if n_permutations is None:
        n_permutations = max(20, int(1 / significance_level))
    elif n_permutations == -1:
        n_permutations = math.factorial(len(nodes))

    ci_test = ci_registry.get_test(ci_test, data=data)
    permutation_violations = []
    n_within_mec = 0

    # Step 1: Compute LMC violations for the given DAG.

    n_lmc_violations, _ = _lmc_violations(dag, data, ci_test, significance_level)

    # Step 2: Generate permutations and compute LMC violations for each to construct null distribution.
    if show_progress and config.SHOW_PROGRESS:
        pbar = tqdm(range(n_permutations), desc="Constructing Null Distribution")
    else:
        pbar = range(n_permutations)

    for _ in pbar:
        # Permute node labels to create a new DAG
        permuted_nodes = np.random.permutation(nodes)
        perm_mapping = dict(zip(nodes, permuted_nodes))
        # Relabel nodes in the original DAG
        nx_permuted_dag = nx.relabel_nodes(dag, perm_mapping, copy=True)
        permuted_dag = DAG()
        permuted_dag.add_nodes_from(nx_permuted_dag.nodes())
        permuted_dag.add_edges_from(nx_permuted_dag.edges())

        n_perm_lmc_violations, _ = _lmc_violations(
            permuted_dag, data, ci_test, significance_level
        )
        permutation_violations.append(n_perm_lmc_violations)
        n_tpa_violations, _ = _tpa_violations(permuted_dag, dag)
        if n_tpa_violations == 0:
            n_within_mec += 1

    # Step 3: Compute test statistics and p-values.

    # Step 3.1: Falsifiability test
    p_value_falsifiable = n_within_mec / n_permutations

    # Step 3.2: Falsification test
    count_less_violations = sum(
        1 for v in permutation_violations if v <= n_lmc_violations
    )
    p_value_falsified = count_less_violations / n_permutations

    # Step 3.3: Confidence intervals for the falsification p-value
    ci_lower, ci_upper = proportion_confint(
        count_less_violations,
        n_permutations,
        alpha=significance_level,
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
    }

    if return_summary:
        result["summary"] = {
            "permutation_violations": permutation_violations,
            "significance_level": significance_level,
            "ci_test": ci_test,
        }

    return result
