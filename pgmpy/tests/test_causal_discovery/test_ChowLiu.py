import networkx as nx
import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor

from pgmpy.causal_discovery import ChowLiu
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
def gaussian_chain_data():
    rng = np.random.default_rng(0)
    n_samples = 2000
    a = rng.normal(size=n_samples)
    b = 0.9 * a + rng.normal(scale=0.4, size=n_samples)
    c = 0.9 * b + rng.normal(scale=0.4, size=n_samples)
    d = 0.9 * c + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d})


def test_gaussian_mutual_info():
    from pgmpy.causal_discovery._base import gaussian_mutual_info

    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    # Independent variables share no information.
    y = rng.normal(size=2000)
    assert gaussian_mutual_info(x, y) == pytest.approx(0.0, abs=0.01)
    # Correlated Gaussians: I(X; Y) = -0.5 * ln(1 - rho^2), checked against
    # the sample correlation so the test verifies the formula, not the sample.
    rho = 0.6
    z = rng.normal(size=2000)
    y = rho * x + np.sqrt(1 - rho**2) * z
    sample_rho = np.corrcoef(x, y)[0, 1]
    assert gaussian_mutual_info(x, y) == pytest.approx(-0.5 * np.log(1 - sample_rho**2))
    # Cross-check against sklearn's kNN estimator (scikit-learn 1.9.1):
    # mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0].
    from sklearn.feature_selection import mutual_info_regression

    assert gaussian_mutual_info(x, y) == pytest.approx(
        mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0], abs=0.05
    )
    assert gaussian_mutual_info(x, y) == gaussian_mutual_info(y, x)
    # A constant variable shares no information; perfect linear dependence
    # has infinite mutual information.
    assert gaussian_mutual_info(np.ones(10), rng.normal(size=10)) == 0.0
    assert gaussian_mutual_info(x, 2 * x + 1) == np.inf


@pytest.mark.parametrize("weight_fn", ["gaussian_mutual_info", "mutual_info_regression"])
def test_chow_liu_continuous(gaussian_chain_data, weight_fn):
    est = ChowLiu(
        root_node="A",
        edge_weights_fn=weight_fn,
        n_jobs=1,
        show_progress=False,
    ).fit(gaussian_chain_data)
    dag = est.causal_graph_

    assert set(dag.nodes()) == {"A", "B", "C", "D"}
    assert nx.is_tree(dag)
    assert set(dag.edges()) == {("A", "B"), ("B", "C"), ("C", "D")}


def test_chow_liu_continuous_tags():
    assert "continuous" in ChowLiu.get_class_tag("data_types")
    assert ChowLiu(edge_weights_fn="gaussian_mutual_info").get_tags()["data_types"] == ["continuous"]
    assert ChowLiu(edge_weights_fn="mutual_info").get_tags()["data_types"] == ["discrete"]


def test_chow_liu_weight_fn_data_mismatch(data12, gaussian_chain_data):
    with pytest.raises(ValueError, match="requires discrete data"):
        ChowLiu(edge_weights_fn="mutual_info", show_progress=False).fit(gaussian_chain_data)
    with pytest.raises(ValueError, match="requires continuous data"):
        ChowLiu(edge_weights_fn="gaussian_mutual_info", show_progress=False).fit(data12)


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)
