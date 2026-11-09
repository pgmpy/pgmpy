import networkx as nx
import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor

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


@pytest.mark.parametrize("backend", ["narwhals", "polars", "pyarrow", "modin", "cudf"])
def test_chow_liu_dataframe_backends(data12, backend):
    if backend == "narwhals":
        nw = pytest.importorskip("narwhals.stable.v2")
        data = nw.from_native(data12)
    elif backend == "polars":
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(data12)
    elif backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        data = pa.Table.from_pandas(data12)
    elif backend == "modin":
        mpd = pytest.importorskip("modin.pandas")
        data = mpd.DataFrame(data12)
    elif backend == "cudf":
        cd = pytest.importorskip("cudf")
        data = cd.DataFrame(data12)

    est_expected = ChowLiu(root_node="A", show_progress=False, n_jobs=1).fit(data12)
    est_actual = ChowLiu(root_node="A", show_progress=False, n_jobs=1).fit(data)

    assert set(est_actual.causal_graph_.nodes()) == set(est_expected.causal_graph_.nodes())
    assert set(est_actual.causal_graph_.edges()) == set(est_expected.causal_graph_.edges())
    assert list(est_actual.feature_names_in_) == list(est_expected.feature_names_in_)
    assert np.array_equal(est_actual.adjacency_matrix_.values, est_expected.adjacency_matrix_.values)

    # Automatic root selection across backends
    est_auto_exp = ChowLiu(root_node=None, show_progress=False, n_jobs=1).fit(data12)
    est_auto_act = ChowLiu(root_node=None, show_progress=False, n_jobs=1).fit(data)
    assert set(est_auto_act.causal_graph_.nodes()) == set(est_auto_exp.causal_graph_.nodes())
    assert set(est_auto_act.causal_graph_.edges()) == set(est_auto_exp.causal_graph_.edges())


@pytest.mark.parametrize("backend", ["narwhals", "polars", "pyarrow"])
def test_chow_liu_categorical_backends(backend):
    df_pd = pd.DataFrame(
        {
            "A": pd.Categorical(["low", "high", "low", "high"] * 10),
            "B": pd.Categorical(["yes", "no", "yes", "no"] * 10),
        }
    )
    if backend == "narwhals":
        nw = pytest.importorskip("narwhals.stable.v2")
        data = nw.from_native(df_pd)
    elif backend == "polars":
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(df_pd)
    elif backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        data = pa.Table.from_pandas(df_pd)

    est_expected = ChowLiu(root_node="A", show_progress=False, n_jobs=1).fit(df_pd)
    est_actual = ChowLiu(root_node="A", show_progress=False, n_jobs=1).fit(data)

    assert set(est_actual.causal_graph_.nodes()) == set(est_expected.causal_graph_.nodes())
    assert set(est_actual.causal_graph_.edges()) == set(est_expected.causal_graph_.edges())


@pytest.fixture(autouse=True)
def shutdown_executor():
    yield
    get_reusable_executor().shutdown(wait=True)
