import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery import VarSort


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [VarSort()],
    expected_failed_checks=expected_failed_checks,
)
def test_var_sort_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def causal_chain_data():
    rng = np.random.default_rng(seed=42)
    n = 1000
    x = rng.normal(0, 1, n)
    y = x + rng.normal(0, 1.5, n)
    z = y + rng.normal(0, 2.0, n)

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def test_adjacency_matrix(causal_chain_data):
    est = VarSort()
    est.fit(causal_chain_data)

    adj = est.adjacency_matrix_
    assert isinstance(adj, pd.DataFrame)
    assert adj.shape == (3, 3)
    assert set(adj.columns) == {"X", "Y", "Z"}


def test_feature_names(causal_chain_data):
    est = VarSort()
    est.fit(causal_chain_data)
    assert hasattr(est, "n_features_in_")
    assert est.n_features_in_ == 3
    assert list(est.feature_names_in_) == ["X", "Y", "Z"]


def test_warns_on_constant_column():
    rng = np.random.default_rng(seed=0)
    data = pd.DataFrame(rng.standard_normal((100, 2)), columns=["X", "Y"])
    data["C"] = 0.0

    est = VarSort()
    with pytest.warns(UserWarning, match="Variable 'C' is constant"):
        assert est.fit(data) is est

    assert "C" in est.causal_graph_
    assert est.causal_graph_.degree("C") == 0


def test_score(causal_chain_data):
    est = VarSort()
    est.fit(causal_chain_data)

    score = est.score(X=causal_chain_data)
    assert isinstance(score, float)

    true_dag = DAG([("X", "Y"), ("Y", "Z")])
    shd_score = est.score(true_graph=true_dag)
    assert isinstance(shd_score, (int, float, np.integer))


@pytest.mark.parametrize("return_type", ["dag", "pdag"])
def test_recovers_collider(return_type):
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.standard_normal((1000, 3)), columns=["X", "Y", "Z"])
    data["Z"] += 2.5 * data["X"] + 2.5 * data["Y"]
    data = data[["Z", "Y", "X"]]
    original = data.copy()
    est = VarSort(return_type=return_type)

    assert est.fit(data) is est
    if return_type == "dag":
        assert isinstance(est.causal_graph_, DAG)
        assert set(est.causal_graph_.edges()) == {("X", "Z"), ("Y", "Z")}
    else:
        assert isinstance(est.causal_graph_, PDAG)
        assert est.causal_graph_.directed_edges == {("X", "Z"), ("Y", "Z")}
        assert not est.causal_graph_.undirected_edges
    assert set(est.causal_order_[:2]) == {"X", "Y"}
    assert est.causal_order_[-1] == "Z"
    pd.testing.assert_frame_equal(
        est.adjacency_matrix_,
        pd.DataFrame([[0, 0, 0], [1, 0, 0], [1, 0, 0]], index=data.columns, columns=data.columns),
    )
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("return_type", ["dag", "pdag"])
def test_tied_scores_keep_input_order(return_type):
    data = pd.DataFrame(np.tile([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], (50, 1)), columns=["Y", "X"])
    est = VarSort(return_type=return_type).fit(data)

    assert est.causal_order_ == ["Y", "X"]
    assert set(est.causal_graph_.nodes()) == {"X", "Y"}
    assert not est.causal_graph_.edges()
    assert not est.adjacency_matrix_.to_numpy().any()


def test_var_sort_is_scale_sensitive(causal_chain_data):
    raw = VarSort().fit(causal_chain_data)
    scaled = causal_chain_data * [100, 1, 0.01]
    rescaled = VarSort().fit(scaled)
    assert set(raw.causal_graph_.edges()) != set(rescaled.causal_graph_.edges())


def test_pdag_represents_reversible_edges():
    data = np.tile([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], (50, 1))
    frame = pd.DataFrame({"Y": 2 * data[:, 0] + 0.5 * data[:, 1], "X": data[:, 0]})
    est = VarSort(return_type="PDAG").fit(frame)

    assert isinstance(est.causal_graph_, PDAG)
    assert not est.causal_graph_.directed_edges
    assert est.causal_graph_.undirected_edges == {("X", "Y")}
    pd.testing.assert_frame_equal(
        est.adjacency_matrix_,
        pd.DataFrame([[0, 1], [1, 0]], index=frame.columns, columns=frame.columns),
    )


def test_invalid_return_type(causal_chain_data):
    with pytest.raises(ValueError, match="return_type must be one of: dag, pdag"):
        VarSort(return_type="invalid").fit(causal_chain_data)
