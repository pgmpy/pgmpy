import networkx as nx
import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor

from pgmpy.estimators import TreeSearch
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


@pytest.fixture
def data12():
    # set random seed
    np.random.seed(0)

    # test data for chow-liu
    return pd.DataFrame(
        np.random.randint(low=0, high=2, size=(100, 5)),
        columns=["A", "B", "C", "D", "E"],
    )


@pytest.fixture
def data13():
    np.random.seed(0)
    # test data for chow-liu
    model = DiscreteBayesianNetwork(
        [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]
    )
    cpd_a = TabularCPD("A", 2, [[0.4], [0.6]])
    cpd_b = TabularCPD(
        "B",
        3,
        [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_c = TabularCPD(
        "C", 2, [[0.3, 0.4], [0.7, 0.6]], evidence=["A"], evidence_card=[2]
    )
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
    # test data for TAN
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


@pytest.fixture
def alarm_df():
    return get_example_model("alarm").simulate(int(1e4), seed=42)


@pytest.mark.parametrize(
    "weight_fn", ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"]
)
@pytest.mark.parametrize("n_jobs", [2, 1])
def test_estimate_chow_liu(data12, data13, weight_fn, n_jobs):
    est = TreeSearch(data12, root_node="A", n_jobs=n_jobs)
    dag = est.estimate(
        estimator_type="chow-liu",
        edge_weights_fn=weight_fn,
        show_progress=False,
    )

    # check number of nodes and edges are as expected
    assert set(dag.nodes()) == {"A", "B", "C", "D", "E"}
    assert nx.is_tree(dag)

    # learn tree structure using A as root node
    est = TreeSearch(data13, root_node="A", n_jobs=n_jobs)
    dag = est.estimate(
        estimator_type="chow-liu",
        edge_weights_fn=weight_fn,
        show_progress=False,
    )

    # check number of nodes and edges are as expected
    assert set(dag.nodes()) == {"A", "B", "C", "D", "E", "F"}
    assert set(dag.edges()) == {
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("B", "E"),
        ("C", "F"),
    }

    # check tree structure exists
    assert dag.has_edge("A", "B")
    assert dag.has_edge("A", "C")
    assert dag.has_edge("B", "D")
    assert dag.has_edge("B", "E")
    assert dag.has_edge("C", "F")


@pytest.mark.parametrize(
    "weight_fn", ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"]
)
@pytest.mark.parametrize("n_jobs", [2, 1])
def test_estimate_tan(data22, weight_fn, n_jobs):
    # learn graph structure
    est = TreeSearch(data22, root_node="R", n_jobs=n_jobs)
    dag = est.estimate(
        estimator_type="tan",
        class_node="A",
        edge_weights_fn=weight_fn,
        show_progress=False,
    )

    # check number of nodes and edges are as expected
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

    # check directed edge between class and independent variables
    assert dag.has_edge("A", "B")
    assert dag.has_edge("A", "C")
    assert dag.has_edge("A", "D")
    assert dag.has_edge("A", "E")

    # check tree structure exists over independent variables
    assert dag.has_edge("R", "B")
    assert dag.has_edge("R", "C")
    assert dag.has_edge("R", "D")
    assert dag.has_edge("R", "E")


def test_estimate_chow_liu_auto_root_node(data12):
    # learn tree structure using auto root node
    est = TreeSearch(data12)

    # root node selection
    weights = est._get_weights(data12)
    sum_weights = weights.sum(axis=0)
    maxw_idx = np.argsort(sum_weights)[::-1]
    root_node = data12.columns[maxw_idx[0]]

    dag = est.estimate(estimator_type="chow-liu", show_progress=False)
    nodes = list(dag.nodes())
    assert nodes[0] == root_node
    assert nodes == ["D", "A", "C", "B", "E"]


def test_estimate_tan_auto_class_node(data22):
    # learn tree structure using auto root and class node
    est = TreeSearch(data22)

    # root and class node selection
    weights = est._get_weights(data22)
    sum_weights = weights.sum(axis=0)
    maxw_idx = np.argsort(sum_weights)[::-1]
    root_node = data22.columns[maxw_idx[0]]
    class_node = data22.columns[maxw_idx[1]]

    dag = est.estimate(estimator_type="tan", class_node=class_node, show_progress=False)
    nodes = list(dag.nodes())
    assert nodes[0] == root_node
    assert nodes[-1] == class_node
    assert sorted(nodes) == sorted(["C", "R", "A", "D", "E", "B"])


def test_tan_real_dataset(alarm_df):
    # Expected values taken from bnlearn.
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
        ("VENTLUNG", "PAP"),
        ("PCWP", "VENTLUNG"),
        ("VENTLUNG", "EXPCO2"),
        ("VENTLUNG", "ARTCO2"),
        ("PAP", "PULMEMBOLUS"),
        ("ARTCO2", "SAO2"),
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
    est = TreeSearch(alarm_df[features + [target]], root_node=features[0])
    edges = est.estimate(
        estimator_type="tan", class_node=target, show_progress=False
    ).edges()
    assert set(expected_edges) == set(edges)


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)
