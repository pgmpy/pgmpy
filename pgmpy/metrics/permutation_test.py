#!/usr/bin/env python
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.stats.proportion as proportion
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators.CITests import ci_registry
from pgmpy.metrics import implied_cis
from pgmpy.models import DynamicBayesianNetwork


def _count_lmc_violations(
    data: pd.DataFrame,
    implied_CIs: pd.DataFrame,
    ci_test_func: Callable,
    significance_level: float = 0.05,
):
    """
    Given `implied_CIs` and `data`, counts the number of CIs that fail when tested on `data`.
    """
    n_violations = 0
    for _, row in implied_CIs.iterrows():
        n_violations += not ci_test_func(
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


def _d_separated_triples(
    implied_CIs: pd.DataFrame,
):
    # CIs_dif = implied_cis(model, data, ci_test_func)
    # valid_CIs = CIs_dif[CIs_dif["p-value"] > significance_level]
    dsep_triples = set()
    for _, row in implied_CIs.iterrows():
        dsep_triples.add((row["u"], row["v"], frozenset(row["cond_vars"])))

    return dsep_triples


def permutation_test(
    dag: DAG,
    data: pd.DataFrame,
    significance_level: float = 0.05,
    n_permutations: int = 100,
    ci_test: str = "chi_square",
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
    model : pgmpy.base.DAG
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

    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Data should be a pandas DataFrame. Got: {type(data)}")

    if not isinstance(dag, DAG):
        if isinstance(dag, DynamicBayesianNetwork):
            raise TypeError(
                "DAG cannot be an instance of pgmpy.models.DynamicBayesianNetwork."
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

    # Initialize CI test function
    ci_test_func = ci_registry.get_test(ci_test, data=data)

    # Step 1: Perform calculations for original DAG and generate permutations
    nodes = list(dag.nodes())
    permutation_violations = []
    same_mec_count = 0

    orginal_CIs = implied_cis(dag, data, ci_test_func)
    valid_CIs = orginal_CIs[orginal_CIs["p-value"] > significance_level]

    original_dsep_triples = _d_separated_triples(valid_CIs)

    # Step 2: Count Local Markov Condition violations in the given graph
    lmc_violations_given = _count_lmc_violations(
        data, valid_CIs, ci_test_func, significance_level
    )

    # Set up progress bar
    if show_progress and config.SHOW_PROGRESS:
        pbar = tqdm(range(n_permutations), desc="Constructing Null Distribution")
    else:
        pbar = range(n_permutations)

    for _ in pbar:
        # Generate random permutation

        # Create permuted CIs
        permuted_CIs = _create_permuted_CIs(valid_CIs, nodes)

        lmc_violations_perm = _count_lmc_violations(
            data, permuted_CIs, ci_test_func, significance_level
        )
        permutation_violations.append(lmc_violations_perm)

        # Check if in same Markov equivalence class (d separations identical = same structure)
        permuted_dsep_triples = _d_separated_triples(permuted_CIs)

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
    falsified = falsifiable and (p_value_falsified > significance_level)

    result = {
        "falsifiable": falsifiable,
        "falsified": falsified,
        "p_value_falsifiable": p_value_falsifiable,
        "p_value_falsified": p_value_falsified,
        "lmc_violations": lmc_violations_given,
        "n_permutations": n_permutations,
        "same_mec_count": same_mec_count,
    }

    count = int(p_value_falsified * n_permutations)

    ci_lower, ci_upper = proportion.proportion_confint(
        count,
        n_permutations,
        alpha=significance_level,
        method="wilson",
    )

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

    return result
