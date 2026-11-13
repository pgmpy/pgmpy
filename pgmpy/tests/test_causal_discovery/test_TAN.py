import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor

from pgmpy.causal_discovery import TAN
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
    n_samples = 2000
    y = rng.integers(0, 2, size=n_samples)
    a = np.where(y == 1, 1.0, -1.0) + rng.normal(scale=0.5, size=n_samples)
    b = 0.9 * a + rng.normal(scale=0.4, size=n_samples)
    c = 0.9 * b + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"A": a, "B": b, "C": c, "Y": y})


@pytest.mark.parametrize("weight_fn", ["gaussian_mutual_info", "mutual_info_regression"])
def test_tan_continuous(gaussian_tan_data, weight_fn):
    est = TAN(
        class_node="Y",
        root_node="A",
        edge_weights_fn=weight_fn,
        n_jobs=1,
        show_progress=False,
    ).fit(gaussian_tan_data)
    dag = est.causal_graph_

    assert set(dag.nodes()) == {"A", "B", "C", "Y"}
    assert set(dag.edges()) == {
        ("Y", "A"),
        ("Y", "B"),
        ("Y", "C"),
        ("A", "B"),
        ("B", "C"),
    }


def test_tan_weight_fn_data_mismatch(data12, gaussian_tan_data):
    with pytest.raises(ValueError, match="requires discrete data"):
        TAN(class_node="A", edge_weights_fn="mutual_info", show_progress=False).fit(gaussian_tan_data)
    with pytest.raises(ValueError, match="requires continuous data"):
        TAN(class_node="A", edge_weights_fn="gaussian_mutual_info", show_progress=False).fit(data12)


def test_tan_rejects_continuous_class_node(gaussian_tan_data):
    rng = np.random.default_rng(0)
    continuous_class_data = gaussian_tan_data.assign(Y=rng.normal(size=len(gaussian_tan_data)))
    for weight_fn in ["gaussian_mutual_info", "mutual_info_regression"]:
        with pytest.raises(ValueError, match="class_node 'Y' must be discrete"):
            TAN(class_node="Y", edge_weights_fn=weight_fn, show_progress=False).fit(continuous_class_data)


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)
