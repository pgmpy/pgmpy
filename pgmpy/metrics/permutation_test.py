#!/usr/bin/env python
from typing import Callable, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators.CITests import ci_registry
from pgmpy.metrics import implied_cis
from pgmpy.models import DynamicBayesianNetwork


def _count_lmc_violations(
    data: pd.DataFrame,
    implied_CIs: pd.DataFrame,
    ci_test: Callable,
    significance_level: float = 0.05,
):
    """
    Given `implied_CIs` and `data`, counts the number of CIs that fail when tested on `data`.
    """
    n_violations = 0
    for _, row in implied_CIs.iterrows():
        n_violations += not ci_test(
            X=row["u"],
            Y=row["v"],
            Z=row["cond_vars"],
            data=data,
            boolean=True,
            significance_level=significance_level,
        )
    return n_violations


def _create_permuted_CIs(ci_df: pd.DataFrame, nodes: list):
    """
    Given the `ci_df`, creates a new dataframe of CIs with the nodes permuted randomly.

    Parameters
    ----------
    ci_df: pd.DataFrame
        A DataFrame containing the Conditional Independences with columns 'u', 'v', and 'cond_vars'.

    nodes : list
        List of all the nodes/variables in the graph/data.

    Returns
    -------
    permuted_CIs : pd.DataFrame with columns 'u', 'v', and 'cond_vars'.
        The implied Conditional Independences, changed according to node permutation.
    """
    permuted_nodes = np.random.permutation(nodes)
    perm_mapping = dict(zip(nodes, permuted_nodes))

    new_cis = pd.DataFrame(columns=["u", "v", "cond_vars"])
    new_cis["u"] = ci_df["u"].apply(perm_mapping.get)
    new_cis["v"] = ci_df["v"].apply(perm_mapping.get)
    new_cis["cond_vars"] = ci_df["cond_vars"].apply(
        lambda t: [perm_mapping[x] for x in t]
    )

    return new_cis


def _compare_CIs(cis1: pd.DataFrame, cis2: pd.DataFrame):
    """Compares two DataFrames of Conditional Independences for equality.

    Parameters
    ----------
    cis1, cis2 : pd.DataFrame
        First DataFrame of Conditional Independences with columns 'u', 'v', and 'cond_vars'.

    Returns
    -------
    bool
        True if both DataFrames represent the same set of Conditional Independences, False otherwise.
    """

    if len(cis1) != len(cis2):
        return False

    set1 = set()
    for _, row in cis1.iterrows():
        set1.add((row["u"], row["v"], frozenset(row["cond_vars"])))

    set2 = set()
    for _, row in cis2.iterrows():
        set2.add((row["u"], row["v"], frozenset(row["cond_vars"])))

    return set1 == set2


def permutation_test(
    dag: DAG,
    data: pd.DataFrame,
    significance_level: float = 0.05,
    n_permutations: int = 100,
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
    >>> from pgmpy.models import DiscreteDiscreteBayesianNetwork
    >>> from pgmpy.metrics import permutation_test
    >>> from pgmpy.utils import get_example_model

    >>> # Test with a known model
    >>> model = get_example_model("cancer")
    >>> data = model.simulate(1000)
    >>> result = permutation_test(model, data)
    >>> print(f"Falsifiable: {result['falsifiable']}, Falsified: {result['falsified']}")

    >>> # Test with wrong model (should be falsified)
    >>> wrong_model = DiscreteDiscreteBayesianNetwork(
    ...     [("Cancer", "Smoker"), ("Smoker", "Pollution")]
    ... )
    >>> result_wrong = permutation_test(wrong_model, data)
    >>> print(f"Wrong model falsified: {result_wrong['falsified']}")
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

    ci_test = ci_registry.get_test(ci_test, data=data)
    permutation_violations = []
    n_within_mec = 0

    # Step 1: Compute LMC violations for the given DAG.
    original_CIs = implied_cis(
        model=dag, data=data, ci_test=ci_test, show_progress=False
    )
    valid_CIs = original_CIs[original_CIs["p-value"] > significance_level]
    n_lmc_violations = original_CIs.shape[0] - valid_CIs.shape[0]

    # Step 2: Generate permutations and compute LMC violations for each to construct null distribution.
    if show_progress and config.SHOW_PROGRESS:
        pbar = tqdm(range(n_permutations), desc="Constructing Null Distribution")
    else:
        pbar = range(n_permutations)

    for _ in pbar:
        # TODO: Check if this is correct - should the LMC violations be computed on all implied CIs or only valid ones?
        permuted_CIs = _create_permuted_CIs(valid_CIs, nodes)
        n_violations_perm = _count_lmc_violations(
            data, permuted_CIs, ci_test, significance_level
        )
        permutation_violations.append(n_violations_perm)

        # TODO: This is wrong - need to compare all implied CIs, not just valid ones. Maybe use is_iequivalent method
        # from pgmpy.base.DAG?
        if _compare_CIs(valid_CIs, permuted_CIs):
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
