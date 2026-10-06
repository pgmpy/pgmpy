import numpy as np
import numpy.testing as np_test
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import PC, BootstrapEstimator, HillClimbSearch


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [BootstrapEstimator(estimator=HillClimbSearch(return_type="dag"), n_bootstraps=2, n_jobs=1)],
    expected_failed_checks=expected_failed_checks,
)
def test_bootstrap_estimator_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def rand_data():
    """
    Generates a 5-node dataset with the following true causal graph:
    A -> C <- B
    C -> D
    D -> E <- B

    Edges: A->C, B->C, B->E, C->D, D->E
    """
    np.random.seed(42)
    data = pd.DataFrame()
    data["A"] = np.random.randint(0, 2, size=500)
    data["B"] = np.random.randint(0, 2, size=500)
    data["C"] = (data["A"] & data["B"]) ^ np.random.binomial(1, 0.05, size=500)
    data["D"] = data["C"] ^ np.random.binomial(1, 0.05, size=500)
    data["E"] = (data["D"] & data["B"]) ^ np.random.binomial(1, 0.05, size=500)
    return data.astype("category")


def edge_type_counts(edges):
    """Number of bootstrap graphs with each edge type ("->", "<-", "--") for every pair of variables."""
    return {
        pair: tuple(int((row == edge_type).sum()) for edge_type in ("->", "<-", "--")) for pair, row in edges.iterrows()
    }


def test_bootstrap_rand_data(rand_data):

    # --- 1. HillClimbSearch (DAG) ---
    hc = BootstrapEstimator(estimator=HillClimbSearch(return_type="dag"), show_progress=False, seed=0)
    hc.fit(rand_data)

    assert hc.bootstrap_edges_.shape == (10, 10)
    assert list(hc.bootstrap_edges_.columns) == list(range(10))
    assert edge_type_counts(hc.bootstrap_edges_) == {
        ("A", "B"): (3, 4, 0),
        ("A", "C"): (6, 4, 0),
        ("A", "D"): (0, 0, 0),
        ("A", "E"): (0, 0, 0),
        ("B", "C"): (5, 5, 0),
        ("B", "D"): (0, 2, 0),
        ("B", "E"): (8, 2, 0),
        ("C", "D"): (6, 4, 0),
        ("C", "E"): (2, 0, 0),
        ("D", "E"): (10, 0, 0),
    }

    # A - B is adjacent in 7 of the 10 graphs, mostly as B -> A. The tie between B -> C and C -> B goes to B -> C.
    expected_edges_hc = {("A", "C"), ("B", "A"), ("B", "C"), ("B", "E"), ("C", "D"), ("D", "E")}
    assert set(hc.causal_graph_.edges()) == expected_edges_hc

    expected_adj_hc = np.array(
        [
            [0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    np_test.assert_array_equal(hc.adjacency_matrix_.values, expected_adj_hc)

    # --- 2. PC (PDAG) ---
    pc = BootstrapEstimator(estimator=PC(return_type="pdag"), show_progress=False, seed=42)
    pc.fit(rand_data)

    assert edge_type_counts(pc.bootstrap_edges_) == {
        ("A", "B"): (0, 0, 0),
        ("A", "C"): (9, 0, 1),
        ("A", "D"): (0, 0, 0),
        ("A", "E"): (0, 0, 0),
        ("B", "C"): (9, 0, 0),
        ("B", "D"): (0, 0, 0),
        ("B", "E"): (9, 0, 0),
        ("C", "D"): (9, 0, 1),
        ("C", "E"): (0, 0, 0),
        ("D", "E"): (9, 0, 1),
    }

    expected_edges_pc = {("B", "C"), ("C", "D"), ("D", "E"), ("B", "E"), ("A", "C")}
    assert set(pc.causal_graph_.edges()) == expected_edges_pc

    expected_adj_pc = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    np_test.assert_array_equal(pc.adjacency_matrix_.values, expected_adj_pc)


def test_bootstrap_bootstrap_warm_start(rand_data):
    est = BootstrapEstimator(
        estimator=HillClimbSearch(return_type="dag"),
        n_bootstraps=10,
        warm_start=True,
        seed=0,
        show_progress=False,
    )
    est.fit(rand_data)

    initial_samples = est.bootstrap_samples_.copy()
    initial_edges = est.bootstrap_edges_.copy()

    # Increase n_bootstraps to 20 and re-fit with warm_start
    est.n_bootstraps = 20
    est.fit(rand_data)

    assert list(est.bootstrap_edges_.columns) == list(range(20))
    assert edge_type_counts(est.bootstrap_edges_) == {
        ("A", "B"): (6, 7, 0),
        ("A", "C"): (13, 7, 0),
        ("A", "D"): (0, 0, 0),
        ("A", "E"): (0, 0, 0),
        ("B", "C"): (11, 9, 0),
        ("B", "D"): (1, 4, 0),
        ("B", "E"): (15, 5, 0),
        ("C", "D"): (15, 5, 0),
        ("C", "E"): (4, 2, 0),
        ("D", "E"): (18, 2, 0),
    }

    # Verify first 10 bootstrap samples and graphs are preserved and identical in 20 bootstrap fit
    np_test.assert_array_equal(est.bootstrap_samples_[:10], initial_samples)
    pd.testing.assert_frame_equal(est.bootstrap_edges_[list(range(10))], initial_edges)


def test_bootstrap_warm_start_validation(rand_data):
    est = BootstrapEstimator(
        estimator=HillClimbSearch(return_type="dag"),
        n_bootstraps=5,
        warm_start=True,
        show_progress=False,
    )
    est.fit(rand_data)

    # Test re-fitting with different dataset sample size
    est.n_bootstraps = 10
    with pytest.raises(ValueError, match="Cannot warm_start with a different dataset size"):
        est.fit(rand_data.iloc[:200])

    # Test re-fitting with different dataset features
    diff_features_df = rand_data.rename(columns={"A": "Z"})
    with pytest.raises(ValueError, match="Cannot warm_start with a different dataset features"):
        est.fit(diff_features_df)

    # Test re-fitting with decreased n_bootstraps
    est.n_bootstraps = 3
    with pytest.raises(ValueError, match="must be larger or equal to"):
        est.fit(rand_data)


def test_bootstrap_get_consensus_graph(rand_data):
    est = BootstrapEstimator(
        estimator=HillClimbSearch(return_type="dag"),
        n_bootstraps=10,
        threshold=0.3,
        show_progress=False,
        seed=0,
    )
    est.fit(rand_data)

    expected_base_edges = {("A", "C"), ("B", "A"), ("B", "C"), ("B", "E"), ("C", "D"), ("D", "E")}
    assert set(est.causal_graph_.edges()) == expected_base_edges

    # A - C, B - C and C - D are adjacent in every bootstrap graph, but in different orientations.
    g_high = est.get_consensus_graph(threshold=0.8)
    expected_high_edges = {("A", "C"), ("B", "C"), ("B", "E"), ("C", "D"), ("D", "E")}
    assert set(g_high.edges()) == expected_high_edges

    # Pairs that are never adjacent are left out even with threshold 0. B - D is only seen as D -> B, which would close
    # the cycle B -> C -> D -> B, so it is skipped.
    g_zero = est.get_consensus_graph(threshold=0.0)
    assert set(g_zero.edges()) == expected_base_edges | {("C", "E")}


def test_bootstrap_get_adjacency_matrix(rand_data):
    est = BootstrapEstimator(
        estimator=HillClimbSearch(return_type="dag"),
        n_bootstraps=10,
        threshold=0.3,
        show_progress=False,
        seed=0,
    )
    est.fit(rand_data)

    expected_base_adj_matrix = np.array(
        [
            [0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    np_test.assert_array_equal(est.adjacency_matrix_.values, expected_base_adj_matrix)

    m_high = est.get_adjacency_matrix(threshold=0.8)
    expected_high_adj_matrix = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    np_test.assert_array_equal(m_high.values, expected_high_adj_matrix)


def test_bootstrap_threshold_invalid(rand_data):
    est = BootstrapEstimator(estimator=HillClimbSearch(return_type="dag"), show_progress=False)
    est.fit(rand_data)
    for invalid_threshold in [-0.5, 1.5]:
        with pytest.raises(ValueError):
            est.get_consensus_graph(threshold=invalid_threshold)
        with pytest.raises(ValueError):
            est.get_adjacency_matrix(threshold=invalid_threshold)


@pytest.mark.parametrize(
    "params",
    [{"threshold": -0.5}, {"threshold": 1.5}, {"sample_size": 0}, {"sample_size": 1.5}, {"n_bootstraps": 0}],
)
def test_bootstrap_fit_invalid_params(rand_data, params):
    est = BootstrapEstimator(estimator=HillClimbSearch(return_type="dag"), show_progress=False, **params)
    with pytest.raises(ValueError, match=list(params)[0]):
        est.fit(rand_data)
