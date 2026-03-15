from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG


def _resolve_edge_weights_fn(edge_weights_fn):
    if edge_weights_fn == "mutual_info":
        return mutual_info_score
    if edge_weights_fn == "adjusted_mutual_info":
        return adjusted_mutual_info_score
    if edge_weights_fn == "normalized_mutual_info":
        return normalized_mutual_info_score
    if not callable(edge_weights_fn):
        raise ValueError(
            "edge_weights_fn should either be 'mutual_info', 'adjusted_mutual_info', "
            "'normalized_mutual_info'or a function of form fun(array, array). Got: "
            f"f{edge_weights_fn}"
        )
    return edge_weights_fn


def _get_weights(data, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True):
    edge_weights_fn = _resolve_edge_weights_fn(edge_weights_fn)

    n_vars = len(data.columns)
    pbar = combinations(data.columns, 2)
    if show_progress and config.SHOW_PROGRESS:
        pbar = tqdm(pbar, total=(n_vars * (n_vars - 1) / 2), desc="Building tree")

    vals = Parallel(n_jobs=n_jobs)(
        delayed(edge_weights_fn)(data.loc[:, u], data.loc[:, v]) for u, v in pbar
    )
    weights = np.zeros((n_vars, n_vars))
    indices = np.triu_indices(n_vars, k=1)
    weights[indices] = vals
    weights.T[indices] = vals
    return weights


def _get_conditional_weights(
    data, class_node, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True
):
    edge_weights_fn = _resolve_edge_weights_fn(edge_weights_fn)

    n_vars = len(data.columns)
    pbar = combinations(data.columns, 2)
    if show_progress and config.SHOW_PROGRESS:
        pbar = tqdm(pbar, total=(n_vars * (n_vars - 1) / 2), desc="Building tree")

    def _conditional_edge_weights_fn(u, v):
        cond_marginal = data.loc[:, class_node].value_counts() / data.shape[0]
        cond_edge_weight = 0
        for index, marg_prob in cond_marginal.items():
            df_cond_subset = data[data.loc[:, class_node] == index]
            cond_edge_weight += marg_prob * edge_weights_fn(
                df_cond_subset.loc[:, u], df_cond_subset.loc[:, v]
            )
        return cond_edge_weight

    vals = Parallel(n_jobs=n_jobs)(
        delayed(_conditional_edge_weights_fn)(u, v) for u, v in pbar
    )
    weights = np.zeros((n_vars, n_vars))
    indices = np.triu_indices(n_vars, k=1)
    weights[indices] = vals
    weights.T[indices] = vals
    return weights


def _create_tree_and_dag(weights, columns, root_node):
    T = nx.maximum_spanning_tree(
        nx.from_pandas_adjacency(
            pd.DataFrame(weights, index=columns, columns=columns),
            create_using=nx.Graph,
        )
    )

    D = nx.bfs_tree(T, root_node)
    return DAG(D)
