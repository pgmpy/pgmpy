import numpy as np
import numpy.testing as np_test
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import PC, BootstrapEstimator, HillClimbSearch
from pgmpy.example_models import load_model


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [BootstrapEstimator(estimator=PC())],
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
    """
    Tests edge discovery on 5-node causal graph:
    A -> C <- B, C -> D -> E, B -> E
    """
    hc = BootstrapEstimator(estimator=HillClimbSearch(return_type="dag"), show_progress=False, seed=42)
    pc = BootstrapEstimator(estimator=PC(), show_progress=False, seed=42)

    hc.fit(rand_data)
    pc.fit(rand_data)

    hc_graph = hc.causal_graph_
    pc_graph = pc.causal_graph_

    expected_edges = {("A", "C"), ("B", "C"), ("B", "E"), ("C", "D"), ("D", "E")}
    assert set(hc_graph.edges()) == expected_edges
    assert set(pc_graph.edges()) == expected_edges


def test_bootstrap_asia():
    asia_model = load_model("bnlearn/asia")
    df = asia_model.simulate(n_samples=500, seed=42)

    base_est = HillClimbSearch(return_type="dag")
    est = BootstrapEstimator(estimator=base_est, n_bootstraps=20, sample_size=0.80, threshold=0.3, seed=42, n_jobs=2)
    est.fit(df)

    expected_edges = {
        ("either", "lung"),
        ("either", "tub"),
        ("either", "smoke"),
        ("either", "xray"),
        ("either", "dysp"),
        ("lung", "tub"),
        ("lung", "dysp"),
        ("smoke", "bronc"),
        ("bronc", "dysp"),
    }

    assert set(est.causal_graph_.edges()) == expected_edges

    expected_edge_prob = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.75, 0.0, 0.0, 0.45, 0.05, 0.0],
            [0.15, 0.25, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.45, 0.0, 0.7, 0.6, 0.65, 0.5],
            [0.0, 0.0, 0.4, 0.3, 0.0, 0.05, 0.6, 0.0],
            [0.0, 0.55, 0.05, 0.2, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.05, 0.0, 0.0],
        ]
    )
    np_test.assert_allclose(
        est.edge_prob_.sort_index(axis=0).sort_index(axis=1).values,
        expected_edge_prob,
    )

    expected_direction_prob = {
        ("tub", "either"): 0.3157894736842105,
        ("either", "tub"): 0.6842105263157895,
        ("either", "dysp"): 1.0,
        ("either", "lung"): 0.7,
        ("either", "smoke"): 0.75,
        ("either", "xray"): 0.5,
        ("dysp", "lung"): 0.2,
        ("dysp", "asia"): 1.0,
        ("dysp", "bronc"): 0.25,
        ("lung", "tub"): 1.0,
        ("lung", "either"): 0.3,
        ("lung", "dysp"): 0.8,
        ("lung", "smoke"): 0.3333333333333333,
        ("smoke", "either"): 0.25,
        ("smoke", "dysp"): 1.0,
        ("smoke", "lung"): 0.6666666666666666,
        ("smoke", "bronc"): 0.55,
        ("xray", "either"): 0.5,
        ("xray", "smoke"): 1.0,
        ("bronc", "tub"): 1.0,
        ("bronc", "dysp"): 0.75,
        ("bronc", "smoke"): 0.45,
    }

    assert len(est.direction_prob_) == len(expected_direction_prob)

    for k, v in expected_direction_prob.items():
        assert np.isclose(est.direction_prob_[k], v)

    # Test with PC estimator
    base_est_pc = PC()
    est_pc = BootstrapEstimator(
        estimator=base_est_pc,
        n_bootstraps=15,
        sample_size=0.85,
        threshold=0.35,
        seed=42,
        n_jobs=2,
    )
    est_pc.fit(df)

    assert set(est_pc.causal_graph_.edges()) == {
        ("tub", "xray"),
        ("lung", "either"),
        ("xray", "either"),
        ("lung", "xray"),
    }

    assert {tuple(sorted(edge)) for edge in est_pc.causal_graph_.undirected_edges} == {
        ("bronc", "dysp"),
        ("bronc", "smoke"),
    }

    assert ((est_pc.edge_prob_ > 0.0) & (est_pc.edge_prob_ < 1.0)).any().any()


def test_bootstrap_alarm():
    alarm_model = load_model("bnlearn/alarm")
    df = alarm_model.simulate(n_samples=2000, seed=42)

    # Test with HillClimbSearch
    base_est = HillClimbSearch(return_type="dag")
    est_hc = BootstrapEstimator(
        estimator=base_est,
        n_bootstraps=6,
        sample_size=0.75,
        threshold=0.35,
        seed=42,
    )
    est_hc.fit(df)

    assert len(est_hc.causal_graph_.edges()) > 0
    assert ((est_hc.edge_prob_ > 0.0) & (est_hc.edge_prob_ < 1.0)).any().any()

    # Test with PC estimator
    base_est_pc = PC()
    est_pc = BootstrapEstimator(
        estimator=base_est_pc,
        n_bootstraps=4,
        sample_size=0.85,
        threshold=0.25,
        seed=42,
    )
    est_pc.fit(df)

    assert len(est_pc.causal_graph_.edges()) > 0
    assert len(est_pc.causal_graph_.undirected_edges) > 0
    assert ((est_pc.edge_prob_ > 0.0) & (est_pc.edge_prob_ < 1.0)).any().any()
