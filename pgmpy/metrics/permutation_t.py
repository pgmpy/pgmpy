#!/usr/bin/env python

import math
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators.CITests import chi_square, pearsonr
from pgmpy.global_vars import logger


def permutation_t(
    model,
    data,
    ci_test="chi_square",
    significance_level=0.05,
    n_permutations=None,
    return_summary=False,
    show_progress=True,
):
    """
    Permutation-based test for falsifying causal graphs using observational data.

    This method implements the permutation-based falsification test from Eulig et al. (2025).
    It tests whether a given DAG is significantly better than random node permutations
    by comparing Local Markov Condition (LMC) violations.

    The test performs two evaluations:
    1. Falsifiability: Whether the graph is informative enough to be falsifiable
    2. Falsification: Whether the graph performs significantly better than random

    Parameters
    ----------
    model : pgmpy.base.DAG or pgmpy.models.DiscreteDiscreteDiscreteBayesianNetwork
        The causal graph to test for falsification.

    data : pandas.DataFrame
        Observational data to test the graph against. Should contain all variables
        present in the model.

    ci_test : str, default='chi_square'
        The conditional independence test to use for testing Local Markov Conditions.
        Options: 'chi_square' (for discrete data), 'pearsonr' (for continuous data).

    significance_level : float, default=0.05
        Significance level for conditional independence tests. Lower values make
        the test more conservative.

    n_permutations : int, optional
        Number of random node permutations to generate for the baseline.
        If None, uses max(20, int(1/significance_level)).

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
    Eulig, E., Mastakouri, A. A., Blöbaum, P., Hardt, M., & Janzing, D. (2025).
    Toward falsifying causal graphs using a permutation-based test.
    Proceedings of the AAAI Conference on Artificial Intelligence, 39(25), 26778-26786.

    Examples
    --------
    >>> from pgmpy.models import DiscreteDiscreteBayesianNetwork
    >>> from pgmpy.metrics import permutation_based_falsification_test
    >>> from pgmpy.utils import get_example_model

    >>> # Test with a known model
    >>> model = get_example_model('cancer')
    >>> data = model.simulate(1000)
    >>> result = permutation_based_falsification_test(model, data)
    >>> print(f"Falsifiable: {result['falsifiable']}, Falsified: {result['falsified']}")

    >>> # Test with wrong model (should be falsified)
    >>> wrong_model = DiscreteDiscreteBayesianNetwork([('Cancer', 'Smoker'), ('Smoker', 'Pollution')])
    >>> result_wrong = permutation_based_falsification_test(wrong_model, data)
    >>> print(f"Wrong model falsified: {result_wrong['falsified']}")
    """

    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")

    model_nodes = set(model.nodes())
    data_columns = set(data.columns)

    if not model_nodes.issubset(data_columns):
        missing_vars = model_nodes - data_columns
        raise ValueError(f"Data missing variables present in model: {missing_vars}")

    # Set default number of permutations
    if n_permutations is None:
        n_permutations = max(20, int(1 / significance_level))

    # Initialize CI test function
    if ci_test == "chi_square":
        ci_test_func = chi_square
    elif ci_test == "pearsonr":
        ci_test_func = pearsonr
    else:
        raise ValueError(
            f"Unsupported CI test: {ci_test}. Use 'chi_square' or 'pearsonr'"
        )

    logger.info(
        f"Starting permutation-based falsification test with {n_permutations} permutations"
    )
    # Step 1: Count LMC violations in the given graph
    lmc_violations_given = _count_lmc_violations(
        model, data, ci_test_func, significance_level
    )

    # Step 2: Generate permutations and test them
    nodes = list(model.nodes())
    permutation_violations = []
    same_mec_count = 0

    # Set up progress bar
    if show_progress and config.SHOW_PROGRESS:
        pbar = tqdm(total=n_permutations, desc="Testing permutations")

    for i in range(n_permutations):
        # Generate random permutation
        perm = np.random.permutation(nodes)
        perm_mapping = dict(zip(nodes, perm))

        # Create permuted graph
        permuted_model = _create_permuted_graph(model, perm_mapping)

        # Count LMC violations in permuted graph
        lmc_violations_perm = _count_lmc_violations(
            permuted_model, data, ci_test_func, significance_level
        )

        permutation_violations.append(lmc_violations_perm)

        # Check if in same Markov equivalence class (0 violations = same structure)
        if lmc_violations_perm == 0:
            same_mec_count += 1

        if show_progress and config.SHOW_PROGRESS:
            pbar.update(1)

    if show_progress and config.SHOW_PROGRESS:
        pbar.close()

    # Step 3: Compute test results

    # Falsifiability test: fraction of permutations in same MEC
    p_value_falsifiable = same_mec_count / n_permutations
    falsifiable = p_value_falsifiable < significance_level

    # Falsification test: fraction of permutations with more violations
    better_than_count = sum(
        1 for v in permutation_violations if v > lmc_violations_given
    )
    # Use conservative estimate with +1 for both numerator and denominator
    p_value_falsified = (better_than_count + 1) / (n_permutations + 1)
    falsified = falsifiable and (p_value_falsified < significance_level)

    # Prepare results
    result = {
        "falsifiable": falsifiable,
        "falsified": falsified,
        "p_value_falsifiable": p_value_falsifiable,
        "p_value_falsified": p_value_falsified,
        "lmc_violations": lmc_violations_given,
        "n_permutations": n_permutations,
        "same_mec_count": same_mec_count,
    }

    if return_summary:
        result["summary"] = {
            "permutation_violations": permutation_violations,
            "significance_level": significance_level,
            "ci_test": ci_test,
            "mean_permutation_violations": np.mean(permutation_violations),
            "std_permutation_violations": np.std(permutation_violations),
            "min_permutation_violations": np.min(permutation_violations),
            "max_permutation_violations": np.max(permutation_violations),
        }

    logger.info(f"Test completed. Falsifiable: {falsifiable}, Falsified: {falsified}")

    # Restore original random state
    np.random.set_state(original_random_state)

    return result


def _count_lmc_violations(model, data, ci_test_func, significance_level):
    """
    Count violations of Local Markov Conditions in the given model.

    For each node X with parents Pa(X), tests if X ⊥ NonDesc(X) \ Pa(X) | Pa(X)
    where NonDesc(X) are all non-descendants of X.
    """
    violations = 0
    nodes = list(model.nodes())

    for node in nodes:
        # Get parents of the current node
        parents = list(model.predecessors(node))

        # Get non-descendants of the current node
        non_descendants = _get_non_descendants(model, node)

        # Test independence with each non-descendant that is not a parent
        test_nodes = [nd for nd in non_descendants if nd not in parents and nd != node]

        for test_node in test_nodes:
            try:
                # Test: node ⊥ test_node | parents
                _, p_value = ci_test_func(node, test_node, parents, data)

                # If p_value < significance_level, we reject independence (violation)
                if p_value < significance_level:
                    violations += 1

            except Exception as e:
                # Handle edge cases (e.g., insufficient data, constant columns)
                logger.debug(
                    f"CI test failed for {node} ⊥ {test_node} | {parents}: {e}"
                )
                continue

    return violations


def _get_non_descendants(model, node):
    """
    Get all non-descendants of a node in the DAG.

    Non-descendants are all nodes that are not reachable from the given node
    by following directed edges.
    """
    descendants = set()

    # Use BFS to find all descendants
    queue = list(model.successors(node))
    visited = set()

    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            descendants.add(current)
            queue.extend(model.successors(current))

    # Non-descendants are all nodes except descendants and the node itself
    all_nodes = set(model.nodes())
    non_descendants = all_nodes - descendants - {node}

    return list(non_descendants)


def _create_permuted_graph(model, perm_mapping):
    """
    Create a new graph with permuted node labels.

    Parameters
    ----------
    model : DAG or DiscreteDiscreteBayesianNetwork
        Original graph
    perm_mapping : dict
        Mapping from original node names to permuted names

    Returns
    -------
    permuted_model : DAG
        New DAG with permuted node labels
    """
    # Create new edges with permuted labels
    new_edges = []
    for edge in model.edges():
        new_source = perm_mapping[edge[0]]
        new_target = perm_mapping[edge[1]]
        new_edges.append((new_source, new_target))

    # Create new DAG
    permuted_model = DAG()
    permuted_model.add_edges_from(new_edges)

    return permuted_model


# Alias for shorter function name
falsify_graph = permutation_t
