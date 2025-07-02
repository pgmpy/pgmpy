from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators.CITests import get_callable_ci_test
from pgmpy.global_vars import logger
from pgmpy.models import DiscreteBayesianNetwork


def evaluate(model, ci_test_func: Callable, data, significance_level=0.05):
    violations = 0
    dsep_triples = set()
    nodes = list(model.nodes())
    for node in nodes:
        parents = list(model.get_parents(node))
        non_descendants = model._get_non_descendants(node)
        test_nodes = [
            nd for nd in non_descendants if (nd not in parents and nd != node)
        ]

        for nd in test_nodes:
            try:
                p_value = ci_test_func(node, nd, parents, data, significance_level)
                if not p_value:
                    violations += 1
                    dsep_triples.add((node, nd, parents))
            except Exception as e:
                logger.debug(f"CI test failed for {node} ⊥ {nd} | {parents}: {e}")
                continue

    return (violations, dsep_triples)


def permuted_graph(model, perm_mapping):
    new_edges = []
    for edge in model.edges():
        new_source = perm_mapping[edge[0]]
        new_target = perm_mapping[edge[1]]
        new_edges.append((new_source, new_target))

    permuted_model = DAG()
    permuted_model.add_edges_from(new_edges)
    return permuted_model


def permutation_test(
    model,
    data,
    ci_test,
    n_permutations=None,
    significance_level=0.05,
    show_progress=True,
):

    if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
        raise ValueError(
            f"model must be an instance of DAG or DiscreteBayesianNetwork. Got {type(model)}"
        )
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas.DataFrame instance. Got {type(data)}")

    if len(model.latents) > 0:
        raise ValueError(
            "This test can not be performed on models with latent variables."
        )

    model_nodes = set(model.nodes())
    cols = set(data.columns)

    if not model_nodes.issubset(cols):
        missing_variables = model_nodes - cols
        raise ValueError(f"Data missing variables in model: {missing_variables}")

    if n_permutations is None:
        n_permutations = 1000

    ci_test_func = get_callable_ci_test(test=ci_test, data=data)

    # Constructing the Null Hypothesis (LMC = Local Markov Condition)
    lmc_violations_given, original_dsep_triples = evaluate(
        model, ci_test_func, data, significance_level
    )
    nodes = list(model.nodes())

    permutation_violations = []
    same_mec_count = 0

    if show_progress and config.SHOW_PROGRESS:
        pbar = tqdm(total=n_permutations, desc="Permutation Testing")
    else:
        pbar = range(n_permutations)

    # Running permutations
    for i in range(pbar):
        perm = np.random.permutation(nodes)
        perm_mapping = dict(zip(nodes, perm))

        # Creating the permuted model while maintaining causal structure
        permuted_model = model.create_permuted_graph(perm_mapping)

        # Counting lmc violations in permuted model and Creating a set of d-separated triples in the permuted_model
        lmc_violations_perm, permuted_dsep_triples = evaluate(
            permuted_model, ci_test_func, data, significance_level
        )
        permutation_violations.append(lmc_violations_perm)

        if original_dsep_triples == permuted_dsep_triples:
            same_mec_count += 1

    p_value_falsifiable = same_mec_count / n_permutations
    p_value_falsified = (
        sum(1 for v in permutation_violations if v <= lmc_violations_given)
        / n_permutations
    )
    return (p_value_falsifiable, p_value_falsified)
