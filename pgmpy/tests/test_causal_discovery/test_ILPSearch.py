import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import ExpertKnowledge, ILPSearch
from pgmpy.example_models import load_model


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [ILPSearch()],
    expected_failed_checks=expected_failed_checks,
)
def test_ilp_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def rand_data():
    """
    Generate 5-node continuous linear SEM dataset.

    Ground Truth DAG Structure:
        X0 ----> X1 ----> X2
        |                  |
        v                  v
        X3 --------------> X4
    """
    np.random.seed(42)
    n = 1000
    x0 = np.random.normal(size=n)
    x1 = 0.7 * x0 + np.random.normal(scale=0.4, size=n)
    x2 = 0.6 * x1 + np.random.normal(scale=0.4, size=n)
    x3 = 0.8 * x0 + np.random.normal(scale=0.4, size=n)
    x4 = 0.5 * x2 + 0.7 * x3 + np.random.normal(scale=0.4, size=n)

    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2, "X3": x3, "X4": x4})


@pytest.fixture
def cancer_data():
    model = load_model("bnlearn/cancer")
    df = model.simulate(n_samples=5000, seed=42)
    return df.apply(lambda col: col.astype("category").cat.codes).astype(float)


def test_rand_data(rand_data):
    est = ILPSearch(penalty="l0", l_penalty=0.01)
    est.fit(rand_data)

    expected_edges = {
        ("X1", "X4"),
        ("X1", "X3"),
        ("X1", "X0"),
        ("X2", "X4"),
        ("X2", "X3"),
        ("X2", "X0"),
        ("X2", "X1"),
        ("X3", "X4"),
        ("X3", "X0"),
        ("X4", "X0"),
    }
    assert set(est.causal_graph_.edges()) == expected_edges
    assert est.adjacency_matrix_.loc["X1", "X4"] == 1


def test_rand_data_expert_knowledge(rand_data):
    # Test required and forbidden edge constraints with full edge verification
    ek1 = ExpertKnowledge(required_edges=[("X0", "X1")], forbidden_edges=[("X1", "X0")])
    est1 = ILPSearch(penalty="l0", l_penalty=0.01, expert_knowledge=ek1)
    est1.fit(rand_data)

    expected_edges_ek1 = {
        ("X0", "X1"),
        ("X0", "X4"),
        ("X1", "X4"),
        ("X2", "X0"),
        ("X2", "X1"),
        ("X2", "X3"),
        ("X2", "X4"),
        ("X3", "X0"),
        ("X3", "X1"),
        ("X3", "X4"),
    }
    assert set(est1.causal_graph_.edges()) == expected_edges_ek1

    # Test restricted candidate search space with full edge verification
    search_space = [("X0", "X1"), ("X1", "X2"), ("X0", "X3"), ("X2", "X4"), ("X3", "X4")]
    ek2 = ExpertKnowledge(search_space=search_space)
    est2 = ILPSearch(penalty="l0", l_penalty=0.01, expert_knowledge=ek2)
    est2.fit(rand_data)

    expected_edges_ek2 = {("X0", "X1"), ("X1", "X2"), ("X0", "X3"), ("X2", "X4"), ("X3", "X4")}
    assert set(est2.causal_graph_.edges()) == expected_edges_ek2


def test_cancer_data(cancer_data):
    est = ILPSearch(penalty="l0", l_penalty=0.001)
    est.fit(cancer_data)

    expected_edges = {("Cancer", "Xray"), ("Cancer", "Smoker"), ("Cancer", "Dyspnoea")}
    assert set(est.causal_graph_.edges()) == expected_edges
