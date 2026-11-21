from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from sklearn.metrics import mutual_info_score

from pgmpy.causal_discovery import TAN
from pgmpy.causal_discovery._base import gaussian_mutual_info
from pgmpy.example_models import load_model
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


@pytest.fixture
def alarm_df():
    return load_model("bnlearn/alarm").simulate(int(1e4), rng=42)


@pytest.mark.parametrize("weight_fn", ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"])
@pytest.mark.parametrize("n_jobs", [2, 1])
def test_tan(data22, weight_fn, n_jobs):
    est = TAN(
        root_node="R",
        class_node="A",
        edge_weights_fn=weight_fn,
        n_jobs=n_jobs,
        show_progress=False,
    ).fit(data22)
    dag = est.causal_graph_

    assert set(dag.nodes()) == {"A", "B", "C", "D", "E", "R"}
    assert set(dag.edges()) == {
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("A", "E"),
        ("A", "R"),
        ("R", "B"),
        ("R", "C"),
        ("R", "D"),
        ("R", "E"),
    }
    assert dag.has_edge("A", "B")
    assert dag.has_edge("A", "C")
    assert dag.has_edge("A", "D")
    assert dag.has_edge("A", "E")
    assert dag.has_edge("R", "B")
    assert dag.has_edge("R", "C")
    assert dag.has_edge("R", "D")
    assert dag.has_edge("R", "E")


def test_tan_auto_class_node(data22):
    # _get_weights is a static method — call it on the class, not an instance.
    weights = TAN._get_weights(data22, show_progress=False)
    sum_weights = weights.sum(axis=0)
    maxw_idx = np.argsort(sum_weights)[::-1]
    root_node = data22.columns[maxw_idx[0]]
    class_node = data22.columns[maxw_idx[1]]

    est = TAN(
        class_node=class_node,
        show_progress=False,
    ).fit(data22)
    dag = est.causal_graph_
    nodes = list(dag.nodes())

    assert nodes[0] == root_node
    assert nodes[-1] == class_node
    assert sorted(nodes) == sorted(["C", "R", "A", "D", "E", "B"])


def test_get_conditional_weights_custom_callable(data22):
    calls = []

    def edge_weights_fn(x, y):
        calls.append((x, y))
        return mutual_info_score(x, y)

    data = data22.astype(str).astype(pd.CategoricalDtype(["unobserved", "2", "1", "0"]))
    weights = TAN._get_conditional_weights(
        data, class_node="A", edge_weights_fn=edge_weights_fn, n_jobs=1, show_progress=False
    )

    # Each pair of feature variables is scored once per observed state of A, on the stored state codes of its samples.
    columns = data.columns
    features = [col for col in columns if col != "A"]
    subsets = [subset for _, subset in data.groupby("A", observed=True)]
    assert all(isinstance(codes, np.ndarray) and codes.dtype.kind == "i" for call in calls for codes in call)
    for (x, y), ((u, v), subset) in zip(calls, product(combinations(features, 2), subsets), strict=True):
        np.testing.assert_array_equal(x, subset[u].cat.codes)
        np.testing.assert_array_equal(y, subset[v].cat.codes)

    # I(u; v | A) = sum_a P(A=a) * I(u; v | A=a), with each term computed on the samples where A=a.
    expected = np.zeros((len(columns), len(columns)))
    for subset in subsets:
        expected += (len(subset) / len(data)) * np.array(
            [[0 if u == v else mutual_info_score(subset[u], subset[v]) for v in columns] for u in columns]
        )
    np.testing.assert_allclose(weights, expected, atol=1e-12)


def test_tan_real_dataset(alarm_df):
    expected_edges = [
        ("CVP", "LVFAILURE"),
        ("CVP", "INTUBATION"),
        ("CVP", "TPR"),
        ("CVP", "DISCONNECT"),
        ("CVP", "VENTMACH"),
        ("CVP", "HR"),
        ("CVP", "FIO2"),
        ("CVP", "HRBP"),
        ("CVP", "VENTLUNG"),
        ("CVP", "PAP"),
        ("CVP", "HISTORY"),
        ("CVP", "PCWP"),
        ("CVP", "INSUFFANESTH"),
        ("CVP", "SAO2"),
        ("CVP", "EXPCO2"),
        ("CVP", "PRESS"),
        ("CVP", "PULMEMBOLUS"),
        ("CVP", "ARTCO2"),
        ("CVP", "MINVOLSET"),
        ("LVFAILURE", "HISTORY"),
        ("LVFAILURE", "PCWP"),
        ("INTUBATION", "INSUFFANESTH"),
        ("EXPCO2", "INTUBATION"),
        ("HR", "TPR"),
        ("PRESS", "DISCONNECT"),
        ("VENTLUNG", "VENTMACH"),
        ("VENTMACH", "PRESS"),
        ("VENTMACH", "MINVOLSET"),
        ("HR", "HRBP"),
        ("ARTCO2", "HR"),
        ("SAO2", "FIO2"),
        ("PCWP", "VENTLUNG"),
        ("VENTLUNG", "EXPCO2"),
        ("VENTLUNG", "ARTCO2"),
        ("ARTCO2", "SAO2"),
        ("PULMEMBOLUS", "PAP"),
        ("SAO2", "PULMEMBOLUS"),
    ]
    features = [
        "LVFAILURE",
        "INTUBATION",
        "TPR",
        "DISCONNECT",
        "VENTMACH",
        "HR",
        "FIO2",
        "HRBP",
        "VENTLUNG",
        "PAP",
        "HISTORY",
        "PCWP",
        "INSUFFANESTH",
        "SAO2",
        "EXPCO2",
        "PRESS",
        "PULMEMBOLUS",
        "ARTCO2",
        "MINVOLSET",
    ]
    target = "CVP"

    est = TAN(
        root_node=features[0],
        class_node=target,
        show_progress=False,
    ).fit(alarm_df[features + [target]])
    edges = est.causal_graph_.edges()

    assert set(expected_edges) == set(edges)


@pytest.fixture
def gaussian_tan_data():
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, size=1000)
    x = label + rng.normal(size=1000)
    y = x + rng.normal(size=1000)
    z = y + rng.normal(size=1000)
    return pd.DataFrame({"label": pd.Categorical(label), "x": x, "y": y, "z": z})


@pytest.mark.parametrize("weight_fn", [None, "gaussian_mutual_info", "mutual_info_regression"])
def test_tan_continuous(gaussian_tan_data, weight_fn):
    est = TAN(class_node="label", root_node="x", edge_weights_fn=weight_fn, n_jobs=1, show_progress=False).fit(
        gaussian_tan_data
    )

    assert set(est.causal_graph_.edges()) == {
        ("label", "x"),
        ("label", "y"),
        ("label", "z"),
        ("x", "y"),
        ("y", "z"),
    }


def test_tan_continuous_features_string_class_labels(gaussian_tan_data):
    data = gaussian_tan_data.assign(label=gaussian_tan_data["label"].map({0: "a", 1: "b"}).astype("category"))

    weights = TAN._get_conditional_weights(
        data, class_node="label", edge_weights_fn="gaussian_mutual_info", n_jobs=1, show_progress=False
    )
    class_idx = data.columns.get_loc("label")
    assert weights.shape == (4, 4)
    assert not weights[class_idx].any() and not weights[:, class_idx].any()
    assert (weights[1:, 1:][np.triu_indices(3, k=1)] > 0).all()

    est = TAN(class_node="label", root_node="x", n_jobs=1, show_progress=False).fit(data)
    assert est.edge_weights_fn_ is gaussian_mutual_info
    assert set(est.causal_graph_.edges()) == {
        ("label", "x"),
        ("label", "y"),
        ("label", "z"),
        ("x", "y"),
        ("y", "z"),
    }


def test_tan_default_edge_weights_fn(data22, gaussian_tan_data):
    est = TAN(class_node="A", n_jobs=1, show_progress=False)
    assert est.fit(data22).edge_weights_fn_ is mutual_info_score

    est = TAN(class_node="label", n_jobs=1, show_progress=False)
    assert est.fit(gaussian_tan_data).edge_weights_fn_ is gaussian_mutual_info

    mixed = gaussian_tan_data.assign(w=pd.Categorical(gaussian_tan_data["label"]))
    with pytest.raises(ValueError, match="mixed"):
        est.fit(mixed)


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)
