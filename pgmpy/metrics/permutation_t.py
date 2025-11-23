#!/usr/bin/env python

from itertools import permutations
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators.CITests import get_callable_ci_test
from pgmpy.global_vars import logger
from pgmpy.metrics import implied_cis
from pgmpy.models import (
    DiscreteBayesianNetwork,
    DynamicBayesianNetwork,
    FunctionalBayesianNetwork,
    LinearGaussianBayesianNetwork,
)


def _count_lmc_violations(
    model, data, implied_CIs, ci_test_func, significance_level=0.05
):
    """
    Count violations of Local Markov Conditions in the given model.

    For each node X with parents Pa(X), tests if X ⊥ NonDesc(X) \ Pa(X) | Pa(X)
    where NonDesc(X) are all non-descendants of X.
    """
    violations = 0
    nodes = list(model.nodes())

    for node in nodes:
        parents = list(model.predecessors(node))

        non_descendants = model._get_non_descendants(node)

        # Test independence with each non-descendant that is not a parent
        test_nodes = [nd for nd in non_descendants if nd not in parents and nd != node]

        for test_node in test_nodes:
            try:
                _, p_value = ci_test_func(node, test_node, parents, data)

                if p_value <= significance_level:
                    violations += 1

            except Exception as e:
                # Handle edge cases (e.g., insufficient data, constant columns)
                logger.debug(
                    f"CI test failed for {node} ⊥ {test_node} | {parents}: {e}"
                )
                continue

    return violations


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


def _create_permuted_CIs(valid_CIs, perm_mapping):
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
    def apply_mapping(node):

        if type(node) == list:
            return [perm_mapping[n] for n in node]
        else:
            return perm_mapping[node]

    new_CIs = valid_CIs.copy()
    new_CIs["u"] = valid_CIs["u"].apply(lambda x: apply_mapping(x))
    new_CIs["v"] = valid_CIs["v"].apply(lambda x: apply_mapping(x))
    new_CIs["cond_vars"] = valid_CIs["cond_vars"].apply(lambda x: apply_mapping(x))

    return new_CIs


def d_separated_triples(model, ci_test_func: Callable, data, significance_level=0.05):

    CIs_dif = implied_cis(model, data, ci_test_func)
    valid_CIs = CIs_dif[CIs_dif["p-value"] >= significance_level]
    dsep_triples = set()
    for _, row in valid_CIs.iterrows():
        dsep_triples.add((row["u"], row["v"], frozenset(row["cond_vars"])))

    return dsep_triples


def permutation_t(
    dag,
    data,
    significance_level: float = 0.05,
    n_permutations: int = None,
    ci_test: str = "chi_square",
    return_summary: bool = True,
    show_progress: bool = True,
):
    """
    Permutation-based test for falsifying causal graphs using observational data.

    For a given DAG, the test checks whether the DAG has fewer Local Markov Condition
    (LMC) violations than a baseline. The baseline has the same causal structure as the DAG.
    This baseline is chosen to be the node permutations of the given DAG. Fewer LMC
    violations mean that the DAG is more consistent/robust than a 'random' guess.

    The test performs two evaluations:
    1. Falsifiability: Whether the DAG is informative enough to be falsifiable. This is
    judged using the fraction of DAGs in the node perumations that are Markov equivalent to
    given DAG
    2. Falsification: Whether the given DAG performs significantly better than the node perumations,
    in terms of fewer LMC violations.

    Parameters
    ----------
    model : pgmpy.base.DAG or any BayesianNetwork from pgmpy.models
        The causal graph to test for falsification.

    data : pandas.DataFrame
        Observational data to test the graph against. Should contain all variables
        present in the model.

    significance_level : float, default=0.05
        Significance level for conditional independence tests. Lower values make
        the test more conservative for accepting the null hypothesis.

    n_permutations : int, optional
        Number of random node permutations to generate for the baseline.
        If None, uses max(20, int(1/significance_level)).

    ci_test : str or fun
        The statistical test to use for testing conditional independence in
        the dataset. If `str` values should be one of:
            "independence_match": If using this option, an additional parameter
                    `independencies` must be specified.
            "chi_square": Uses the Chi-Square independence test. This works
                    only for discrete datasets.
            "pearsonr": Uses the partial correlation based on pearson
                    correlation coefficient to test independence. This works
                    only for continuous datasets.
            "g_sq": G-test. Works only for discrete datasets.
            "log_likelihood": Log-likelihood test. Works only for discrete dataset.
            "freeman_tuckey": Freeman Tuckey test. Works only for discrete dataset.
            "modified_log_likelihood": Modified Log Likelihood test. Works only for discrete variables.
            "neyman": Neyman test. Works only for discrete variables.
            "cressie_read": Cressie Read test. Works only for discrete variables.

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
    Eulig, E., Mastakouri, A. A., Blöbaum, P., Hardt, M., & Janzing, D. (2025).
    Toward falsifying causal graphs using a permutation-based test.
    Proceedings of the AAAI Conference on Artificial Intelligence, 39(25), 26778-26786.

    Examples
    --------
    >>> from pgmpy.models import DiscreteDiscreteBayesianNetwork
    >>> from pgmpy.metrics import permutation_based_falsification_test
    >>> from pgmpy.utils import get_example_model

    >>> # Test with a known model
    >>> model = get_example_model("cancer")
    >>> data = model.simulate(1000)
    >>> result = permutation_based_falsification_test(model, data)
    >>> print(f"Falsifiable: {result['falsifiable']}, Falsified: {result['falsified']}")

    >>> # Test with wrong model (should be falsified)
    >>> wrong_model = DiscreteDiscreteBayesianNetwork(
    ...     [("Cancer", "Smoker"), ("Smoker", "Pollution")]
    ... )
    >>> result_wrong = permutation_based_falsification_test(wrong_model, data)
    >>> print(f"Wrong model falsified: {result_wrong['falsified']}")
    """

    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Data should be a pandas DataFrame. Got: {type(data)}")

    if isinstance(
        dag,
        (
            DAG,
            DiscreteBayesianNetwork,
            DynamicBayesianNetwork,
            FunctionalBayesianNetwork,
            LinearGaussianBayesianNetwork,
        ),
    ):
        if len(dag.latents) > 0:
            raise ValueError(
                f"Found latent variables: {dag.latents}. "
                "permutation_t does not support latent variables."
            )
    else:
        raise TypeError(
            f"DAG should be a pgmpy.base.DAG or Bayesian Network from pgmpy.models. Got: {type(dag)}"
        )

    model_nodes = set(dag.nodes())
    data_columns = set(data.columns)

    if not model_nodes.issubset(data_columns):
        missing_vars = model_nodes - data_columns
        raise ValueError(f"Data missing variables present in model: {missing_vars}")

    # Set default number of permutations
    if n_permutations is None:
        n_permutations = max(20, int(1 / significance_level))

    # Initialize CI test function
    ci_test_func = get_callable_ci_test(ci_test, data=data)

    logger.info(
        f"Starting permutation-based falsification test with {n_permutations} permutations"
    )

    # Step 2: Generate permutations and test them
    nodes = list(dag.nodes())
    permutation_violations = []
    same_mec_count = 0

    original_dsep_triples = d_separated_triples(
        dag, ci_test_func, data, significance_level
    )
    orginal_CIs = implied_cis(dag, data, ci_test_func)
    valid_CIs = orginal_CIs[orginal_CIs["p-value"] >= significance_level]

    # Step 1: Count Local Markov Condition violations in the given graph
    lmc_violations_given = _count_lmc_violations(
        dag, data, valid_CIs, ci_test_func, significance_level
    )

    # Set up progress bar
    if show_progress and config.SHOW_PROGRESS:
        pbar = tqdm(range(n_permutations), desc="Constructing Null Distribution")
    else:
        pbar = range(n_permutations)

    for _ in pbar:
        # Generate random permutation
        perm = np.random.permutation(nodes)
        perm_mapping = dict(zip(nodes, perm))

        # Create permuted graph
        permuted_model = _create_permuted_graph(dag, perm_mapping)
        permuted_CIs = _create_permuted_CIs(valid_CIs, perm_mapping)

        # Count LMC violations in permuted graph
        lmc_violations_perm = _count_lmc_violations(
            permuted_model, data, permuted_CIs, ci_test_func, significance_level
        )

        permutation_violations.append(lmc_violations_perm)

        # Check if in same Markov equivalence class (d separations identical = same structure)
        permuted_dsep_triples = d_separated_triples(
            permuted_model, ci_test_func, data, significance_level
        )

        if original_dsep_triples == permuted_dsep_triples:
            same_mec_count += 1

    # Step 3: Compute test results

    # Falsifiability test: fraction of permutations in same MEC
    p_value_falsifiable = same_mec_count / n_permutations
    falsifiable = p_value_falsifiable <= significance_level

    # Falsification test: fraction of permutations with lesser violations
    count_less_violations = sum(
        1 for v in permutation_violations if v <= lmc_violations_given
    )

    p_value_falsified = count_less_violations / n_permutations
    falsified = falsifiable and (p_value_falsified < significance_level)

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

    return result
