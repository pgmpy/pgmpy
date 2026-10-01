import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import SID, get_metrics


@pytest.fixture
def sid_scorer():
    return SID()


def test_sid_simple_dag(sid_scorer):
    # Asymmetric test: dag1 -> dag2 vs dag2 -> dag1
    dag1 = DAG([(1, 2), (1, 3), (2, 3)])
    dag2 = DAG([(1, 3), (2, 3)])

    assert sid_scorer(dag1, dag2) == 2
    assert sid_scorer(dag2, dag1) == 0


def test_sid_edge_reversal(sid_scorer):
    dag1 = DAG([(1, 2)])
    dag2 = DAG([(2, 1)])
    assert sid_scorer(dag1, dag2) == 2
    assert sid_scorer(dag2, dag1) == 2


def test_sid_empty_and_single_edge(sid_scorer):
    no_edge = DAG()
    no_edge.add_nodes_from([0, 1])
    one_edge = DAG([(0, 1)])
    opposite_edge = DAG([(1, 0)])

    assert sid_scorer(no_edge, one_edge) == 0
    assert sid_scorer(one_edge, no_edge) == 1
    assert sid_scorer(one_edge, opposite_edge) == 2


def test_sid_identical_dags(sid_scorer):
    dag = DAG([(1, 2), (2, 3), (3, 4), (1, 4)])
    assert sid_scorer(dag, dag) == 0


def test_sid_return_matrix():
    dag1 = DAG([(1, 2), (1, 3), (2, 3)])
    dag2 = DAG([(1, 3), (2, 3)])

    scorer_mat = SID(return_matrix=True)
    mat = scorer_mat(dag1, dag2)

    assert isinstance(mat, pd.DataFrame)
    assert mat.shape == (3, 3)
    expected = np.array(
        [
            [False, False, False],
            [True, False, True],
            [False, False, False],
        ]
    )
    assert np.array_equal(mat.to_numpy(), expected)


def test_sid_benchmark_matrices():
    # 11-node benchmark DAG adjacency matrices validated against R SID package
    adj_mat = [
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            ]
        ),
    ]

    def _to_dag(adj):
        p = len(adj)
        dag = DAG()
        dag.add_nodes_from(range(p))
        for i in range(p):
            for j in range(p):
                if adj[i, j] > 0:
                    dag.add_edge(i, j)
        return dag

    dags = [_to_dag(a) for a in adj_mat]

    expected_0_1 = np.array(
        [
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
        ]
    )

    expected_1_0 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ]
    )

    expected_1_2 = np.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1],
            [0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
    )

    sid_mat_scorer = SID(return_matrix=True)

    assert np.array_equal(sid_mat_scorer(dags[0], dags[1]).to_numpy(), expected_0_1)
    assert np.array_equal(sid_mat_scorer(dags[1], dags[0]).to_numpy(), expected_1_0)
    assert np.array_equal(sid_mat_scorer(dags[1], dags[2]).to_numpy(), expected_1_2)

    sid_scorer = SID()
    assert sid_scorer(dags[0], dags[1]) == int(np.sum(expected_0_1))
    assert sid_scorer(dags[1], dags[0]) == int(np.sum(expected_1_0))
    assert sid_scorer(dags[1], dags[2]) == int(np.sum(expected_1_2))


def test_sid_pdag_evaluation():
    true_dag = DAG([(1, 2), (2, 3)])
    pdag = PDAG(edge_list=[(1, 2, "--"), (2, 3, "->")])

    sid_lower = SID(variant="lower")
    sid_upper = SID(variant="upper")
    sid_mean = SID(variant="mean")

    assert sid_lower(true_dag, pdag) == 0
    assert sid_upper(true_dag, pdag) == 3
    assert sid_mean(true_dag, pdag) == 1.5

    sid_mat_lower = SID(return_matrix=True, variant="lower")
    sid_mat_upper = SID(return_matrix=True, variant="upper")
    sid_mat_mean = SID(return_matrix=True, variant="mean")

    res_lower = sid_mat_lower(true_dag, pdag)
    res_upper = sid_mat_upper(true_dag, pdag)
    res_mean = sid_mat_mean(true_dag, pdag)

    assert isinstance(res_lower, pd.DataFrame)
    assert isinstance(res_upper, pd.DataFrame)
    assert isinstance(res_mean, pd.DataFrame)


def test_sid_invalid_inputs():
    with pytest.raises(ValueError, match="variant must be one of"):
        SID(variant="invalid")

    dag1 = DAG([(1, 2)])
    dag2 = DAG([(1, 3)])
    with pytest.raises(ValueError, match="must be on the same nodes"):
        SID()(dag1, dag2)

    pdag_true = PDAG(edge_list=[(1, 2, "--")])
    with pytest.raises(ValueError, match="true_causal_graph must be a fully directed DAG"):
        SID()(pdag_true, dag1)


def test_sid_get_metrics_lookup():
    metrics = get_metrics(name="SID")
    assert len(metrics) == 1
    assert metrics[0] == SID
