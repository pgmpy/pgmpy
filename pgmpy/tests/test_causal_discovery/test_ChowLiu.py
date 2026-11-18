import networkx as nx
import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score

from pgmpy.causal_discovery import ChowLiu
from pgmpy.causal_discovery._base import gaussian_mutual_info
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


@pytest.fixture
def data12():
    np.random.seed(0)
    return pd.DataFrame(
        np.random.randint(low=0, high=2, size=(100, 5)),
        columns=["A", "B", "C", "D", "E"],
    ).astype("category")


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
    return model.simulate(n_samples=10000, rng=0)


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
    return model.simulate(n_samples=10000, rng=0)


@pytest.mark.parametrize("weight_fn", ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"])
@pytest.mark.parametrize("n_jobs", [2, 1])
def test_chow_liu(data12, data13, weight_fn, n_jobs):
    est = ChowLiu(
        root_node="A",
        edge_weights_fn=weight_fn,
        n_jobs=n_jobs,
        show_progress=False,
    ).fit(data12)
    dag = est.causal_graph_

    assert set(dag.nodes()) == {"A", "B", "C", "D", "E"}
    assert nx.is_tree(dag)

    est = ChowLiu(
        root_node="A",
        edge_weights_fn=weight_fn,
        n_jobs=n_jobs,
        show_progress=False,
    ).fit(data13)
    dag = est.causal_graph_

    assert set(dag.nodes()) == {"A", "B", "C", "D", "E", "F"}
    assert set(dag.edges()) == {
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("B", "E"),
        ("C", "F"),
    }
    assert dag.has_edge("A", "B")
    assert dag.has_edge("A", "C")
    assert dag.has_edge("B", "D")
    assert dag.has_edge("B", "E")
    assert dag.has_edge("C", "F")


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


@pytest.fixture
def gaussian_chain():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = x + rng.normal(size=500)
    z = y + rng.normal(size=500)
    return pd.DataFrame({"x": x, "y": y, "z": z})


@pytest.mark.parametrize("weight_fn", [None, "gaussian_mutual_info", "mutual_info_regression", mutual_info_regression])
def test_chow_liu_continuous(gaussian_chain, weight_fn):
    est = ChowLiu(root_node="x", edge_weights_fn=weight_fn, n_jobs=1, show_progress=False).fit(gaussian_chain)

    assert set(est.causal_graph_.edges()) == {("x", "y"), ("y", "z")}


def test_chow_liu_default_edge_weights_fn(data12, gaussian_chain):
    discrete = data12.astype("category")
    est = ChowLiu(n_jobs=1, show_progress=False)
    assert est.fit(discrete).edge_weights_fn_ is mutual_info_score
    assert est.fit(gaussian_chain).edge_weights_fn_ is gaussian_mutual_info


def test_gaussian_mutual_info(gaussian_chain):
    x, y = gaussian_chain["x"], gaussian_chain["y"]
    rho = np.corrcoef(x, y)[0, 1]
    assert gaussian_mutual_info(x, y) == pytest.approx(-0.5 * np.log(1 - rho**2))
    assert gaussian_mutual_info(x, y) == pytest.approx(gaussian_mutual_info(y, x))

    # Perfectly collinear and constant variables give finite scores.
    assert np.isfinite(gaussian_mutual_info(x, 2 * x + 1))
    assert np.isfinite(gaussian_mutual_info(x, -x))
    assert gaussian_mutual_info(x, np.ones(len(x))) == 0.0

    collinear = gaussian_chain.assign(w=2 * gaussian_chain["x"] + 1)
    est = ChowLiu(n_jobs=1, show_progress=False).fit(collinear)
    assert nx.is_tree(est.causal_graph_)
    assert set(est.causal_graph_.nodes()) == set(collinear.columns)


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)
