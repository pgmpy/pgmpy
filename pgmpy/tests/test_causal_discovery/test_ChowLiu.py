import networkx as nx
import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery import ChowLiu
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling


@pytest.fixture
def data12():
    np.random.seed(0)
    return pd.DataFrame(
        np.random.randint(low=0, high=2, size=(100, 5)),
        columns=["A", "B", "C", "D", "E"],
    )


@pytest.fixture
def data13():
    np.random.seed(0)
    model = DiscreteBayesianNetwork([("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")])
    cpd_a = TabularCPD("A", 2, [[0.4], [0.6]])
    cpd_b = TabularCPD(
        "B",
        3,
        [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_c = TabularCPD("C", 2, [[0.3, 0.4], [0.7, 0.6]], evidence=["A"], evidence_card=[2])
    cpd_d = TabularCPD(
        "D",
        3,
        [[0.5, 0.3, 0.1], [0.4, 0.4, 0.8], [0.1, 0.3, 0.1]],
        evidence=["B"],
        evidence_card=[3],
    )
    cpd_e = TabularCPD(
        "E",
        2,
        [[0.3, 0.5, 0.2], [0.7, 0.5, 0.8]],
        evidence=["B"],
        evidence_card=[3],
    )
    cpd_f = TabularCPD(
        "F",
        3,
        [[0.3, 0.6], [0.5, 0.2], [0.2, 0.2]],
        evidence=["C"],
        evidence_card=[2],
    )
    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d, cpd_e, cpd_f)
    inference = BayesianModelSampling(model)
    return inference.forward_sample(size=10000)


@pytest.fixture
def data22():
    np.random.seed(0)
    model = DiscreteBayesianNetwork(
        [
            ("A", "R"),
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("A", "E"),
            ("R", "B"),
            ("R", "C"),
            ("R", "D"),
            ("R", "E"),
        ]
    )
    cpd_a = TabularCPD("A", 2, [[0.7], [0.3]])
    cpd_r = TabularCPD(
        "R",
        3,
        [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_b = TabularCPD(
        "B",
        3,
        [
            [0.1, 0.1, 0.2, 0.2, 0.7, 0.1],
            [0.1, 0.3, 0.1, 0.2, 0.1, 0.2],
            [0.8, 0.6, 0.7, 0.6, 0.2, 0.7],
        ],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    cpd_c = TabularCPD(
        "C",
        2,
        [[0.7, 0.2, 0.2, 0.5, 0.1, 0.3], [0.3, 0.8, 0.8, 0.5, 0.9, 0.7]],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    cpd_d = TabularCPD(
        "D",
        3,
        [
            [0.3, 0.8, 0.2, 0.8, 0.4, 0.7],
            [0.4, 0.1, 0.4, 0.1, 0.1, 0.1],
            [0.3, 0.1, 0.4, 0.1, 0.5, 0.2],
        ],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    cpd_e = TabularCPD(
        "E",
        2,
        [[0.5, 0.6, 0.6, 0.5, 0.5, 0.4], [0.5, 0.4, 0.4, 0.5, 0.5, 0.6]],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    model.add_cpds(cpd_a, cpd_r, cpd_b, cpd_c, cpd_d, cpd_e)
    inference = BayesianModelSampling(model)
    return inference.forward_sample(size=10000)


@pytest.mark.parametrize("weight_fn", ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"])
@pytest.mark.parametrize("n_jobs", [2, 1])
@pytest.mark.parametrize("return_type", ["dag", "pdag"])  # checks for return_type
def test_chow_liu(data12, data13, weight_fn, n_jobs, return_type):
    est = ChowLiu(
        root_node="A",
        edge_weights_fn=weight_fn,
        return_type=return_type,
        n_jobs=n_jobs,
        show_progress=False,
    ).fit(data12)
    graph = est.causal_graph_

    assert set(graph.nodes()) == {"A", "B", "C", "D", "E"}
    assert nx.is_tree(graph)

    est = ChowLiu(
        root_node="A",
        edge_weights_fn=weight_fn,
        return_type=return_type,
        n_jobs=n_jobs,
        show_progress=False,
    ).fit(data13)
    graph = est.causal_graph_

    expected_edges = {
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("B", "E"),
        ("C", "F"),
    }
    assert set(graph.nodes()) == {"A", "B", "C", "D", "E", "F"}
    # adjacency matrix, check if the edges are as expected
    if return_type == "dag":
        assert isinstance(graph, DAG)
        assert set(graph.edges()) == expected_edges
        for u, v in expected_edges:
            assert graph.has_edge(u, v)
            assert est.adjacency_matrix_.loc[u, v] == 1
            assert est.adjacency_matrix_.loc[v, u] == 0
    else:
        assert isinstance(graph, PDAG)
        assert not graph.directed_edges
        assert {tuple(sorted(e)) for e in graph.undirected_edges} == {tuple(sorted(e)) for e in expected_edges}
        for u, v in expected_edges:
            assert est.adjacency_matrix_.loc[u, v] == 1
            assert est.adjacency_matrix_.loc[v, u] == 1

    with pytest.raises(ValueError, match="return_type must be one of: dag, pdag, or cpdag"):
        ChowLiu(return_type="invalid", show_progress=False).fit(data12)


def test_chow_liu_auto_root_node(data12):
    weights = ChowLiu._get_weights(data12, show_progress=False)
    sum_weights = weights.sum(axis=0)
    maxw_idx = np.argsort(sum_weights)[::-1]
    root_node = data12.columns[maxw_idx[0]]

    est = ChowLiu(show_progress=False).fit(data12)
    dag = est.causal_graph_
    nodes = list(dag.nodes())

    assert nodes[0] == root_node
    assert nodes == ["D", "A", "C", "B", "E"]


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)
