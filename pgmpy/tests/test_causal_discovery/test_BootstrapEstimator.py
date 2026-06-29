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


def test_bootstrap_rand_data(rand_data):

    # --- 1. HillClimbSearch (DAG) ---
    hc = BootstrapEstimator(estimator=HillClimbSearch(return_type="dag"), show_progress=False, seed=0)
    hc.fit(rand_data)

    hc_graph = hc.causal_graph_

    expected_edges_hc = {("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("B", "E")}
    assert set(hc_graph.edges()) == expected_edges_hc

    expected_adj_hc = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    np_test.assert_array_equal(hc.adjacency_matrix_.values, expected_adj_hc)

    expected_edge_prob_hc = np.array(
        [
            [0.0, 0.3, 0.6, 0.0, 0.0],
            [0.4, 0.0, 0.5, 0.0, 0.8],
            [0.4, 0.5, 0.0, 0.6, 0.2],
            [0.0, 0.2, 0.4, 0.0, 1.0],
            [0.0, 0.2, 0.0, 0.0, 0.0],
        ]
    )
    np_test.assert_allclose(hc.edge_prob_.values, expected_edge_prob_hc)

    expected_direction_prob_hc = {
        ("A", "B"): 0.42857142857142855,
        ("A", "C"): 0.6,
        ("B", "A"): 0.5714285714285714,
        ("B", "C"): 0.5,
        ("B", "E"): 0.8,
        ("C", "A"): 0.4,
        ("C", "B"): 0.5,
        ("C", "D"): 0.6,
        ("C", "E"): 1.0,
        ("D", "B"): 1.0,
        ("D", "C"): 0.4,
        ("D", "E"): 1.0,
        ("E", "B"): 0.2,
    }
    assert hc.direction_prob_.keys() == expected_direction_prob_hc.keys()
    for k in expected_direction_prob_hc:
        np_test.assert_allclose(hc.direction_prob_[k], expected_direction_prob_hc[k])

    # --- 2. PC (PDAG) ---
    pc = BootstrapEstimator(estimator=PC(return_type="pdag"), show_progress=False, seed=42)
    pc.fit(rand_data)

    pc_graph = pc.causal_graph_

    expected_edges_pc = {("B", "C"), ("C", "D"), ("D", "E"), ("B", "E"), ("A", "C")}
    assert set(pc_graph.edges()) == expected_edges_pc

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

    expected_edge_prob_pc = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0, 0.9],
            [0.1, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.1, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.1, 0.0],
        ]
    )
    np_test.assert_allclose(pc.edge_prob_.values, expected_edge_prob_pc)

    expected_direction_prob_pc = {
        ("A", "C"): 0.9,
        ("B", "C"): 1.0,
        ("B", "E"): 1.0,
        ("C", "A"): 0.0,
        ("C", "D"): 0.9,
        ("D", "C"): 0.0,
        ("D", "E"): 0.9,
        ("E", "D"): 0.0,
    }
    assert pc.direction_prob_.keys() == expected_direction_prob_pc.keys()
    for k in expected_direction_prob_pc:
        np_test.assert_allclose(pc.direction_prob_[k], expected_direction_prob_pc[k])


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
    initial_graphs = est.bootstrap_graphs_.copy()

    expected_edge_prob_10 = np.array(
        [
            [0.0, 0.3, 0.6, 0.0, 0.0],
            [0.4, 0.0, 0.5, 0.0, 0.8],
            [0.4, 0.5, 0.0, 0.6, 0.2],
            [0.0, 0.2, 0.4, 0.0, 1.0],
            [0.0, 0.2, 0.0, 0.0, 0.0],
        ]
    )
    np_test.assert_allclose(est.edge_prob_.values, expected_edge_prob_10)

    # Increase n_bootstraps to 20 and re-fit with warm_start
    est.n_bootstraps = 20
    est.fit(rand_data)

    expected_edge_prob_20 = np.array(
        [
            [0.0, 0.3, 0.65, 0.0, 0.0],
            [0.35, 0.0, 0.55, 0.05, 0.75],
            [0.35, 0.45, 0.0, 0.75, 0.2],
            [0.0, 0.2, 0.25, 0.0, 0.9],
            [0.0, 0.25, 0.1, 0.1, 0.0],
        ]
    )
    np_test.assert_allclose(est.edge_prob_.values, expected_edge_prob_20)

    # Verify first 10 bootstrap samples and graphs are preserved and identical in 20 bootstrap fit
    np_test.assert_array_equal(est.bootstrap_samples_[:10], initial_samples)
    np_test.assert_array_equal(est.bootstrap_graphs_[:10], initial_graphs)


def test_bootstrap_warm_start_invalid_dataset(rand_data):
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

    g_high = est.get_consensus_graph(threshold=0.8)
    expected_high_edges = {("B", "E"), ("D", "E")}
    assert set(g_high.edges()) == expected_high_edges


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
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
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
