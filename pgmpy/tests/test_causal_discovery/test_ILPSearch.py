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
        "check_dtype_object": (
            "ILPSearch requires continuous numeric data; object arrays with integers are classified as discrete."
        ),
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
    Generate 4-node continuous linear SEM dataset (single connected component with a collider):
        X0 ----> X2 <---- X1
                  |
                  v
                 X3
    """
    np.random.seed(42)
    n = 2000
    x0 = np.random.normal(size=n)
    x1 = np.random.normal(size=n)
    x2 = 0.8 * x0 + 0.8 * x1 + np.random.normal(scale=0.3, size=n)
    x3 = 0.8 * x2 + np.random.normal(scale=0.3, size=n)

    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2, "X3": x3})


@pytest.fixture
def cancer_data():
    model = load_model("bnlearn/cancer")
    df = model.simulate(n_samples=5000, seed=42)
    df = df[["Cancer", "Pollution", "Smoker", "Xray", "Dyspnoea"]]
    return df.apply(lambda col: col.astype("category").cat.codes).astype(float)


@pytest.fixture
def sachs_data():
    model = load_model("bnlearn/sachs")
    df = model.simulate(n_samples=5000, seed=42)
    return df.apply(lambda col: col.astype("category").cat.codes).astype(float)


def test_rand_data(rand_data):
    est = ILPSearch(l_penalty=0.1)
    est.fit(rand_data)

    expected_edges = {("X0", "X2"), ("X1", "X2"), ("X3", "X0"), ("X3", "X1"), ("X3", "X2")}
    assert set(est.causal_graph_.edges()) == expected_edges

    expected_adj = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
        ]
    )
    np.testing.assert_array_equal(est.adjacency_matrix_.to_numpy(), expected_adj)


def test_solver_options(rand_data):
    options = {"time_limit": 10.0, "disp": False, "presolve": True}
    est = ILPSearch(l_penalty=0.1, options=options).fit(rand_data)
    assert est.milp_result_ is not None
    assert est.milp_result_.x is not None


def test_rand_data_expert_knowledge(rand_data):

    # 1. Required and forbidden edge constraints
    ek_req = ExpertKnowledge(required_edges=[("X0", "X2")], forbidden_edges=[("X3", "X2")])
    est_req = ILPSearch(l_penalty=0.1, expert_knowledge=ek_req).fit(rand_data)
    assert ("X0", "X2") in est_req.causal_graph_.edges()
    assert ("X3", "X2") not in est_req.causal_graph_.edges()

    # 2. Restrict candidate search space to adjacent skeleton
    search_space = [
        ("X0", "X2"),
        ("X2", "X0"),
        ("X1", "X2"),
        ("X2", "X1"),
        ("X2", "X3"),
        ("X3", "X2"),
    ]
    ek_space = ExpertKnowledge(search_space=search_space)
    est_space = ILPSearch(l_penalty=0.1, expert_knowledge=ek_space).fit(rand_data)
    expected_space_edges = {("X0", "X2"), ("X1", "X2"), ("X3", "X2")}
    assert set(est_space.causal_graph_.edges()) == expected_space_edges

    # 3. Enforce topological tiers: X0, X1 (tier 0) -> X2 (tier 1) -> X3 (tier 2)
    ek_temp = ExpertKnowledge(temporal_order=[["X0", "X1"], ["X2"], ["X3"]])
    est_temp = ILPSearch(l_penalty=0.1, expert_knowledge=ek_temp).fit(rand_data)
    expected_temp_edges = {("X0", "X2"), ("X0", "X3"), ("X1", "X2"), ("X1", "X3"), ("X2", "X3")}
    assert set(est_temp.causal_graph_.edges()) == expected_temp_edges

    # 4. Explicitly configured marginal independence screening
    ek_md = ExpertKnowledge(search_space="marginally_dependent")
    est_md = ILPSearch(l_penalty=0.1, expert_knowledge=ek_md).fit(rand_data)
    assert len(est_md.causal_graph_.edges()) > 0
    assert ("X0", "X1") not in est_md.causal_graph_.edges()


def test_invalid_non_continuous_data():
    df_discrete = pd.DataFrame({"A": ["low", "high", "low"], "B": ["yes", "no", "yes"]})
    with pytest.raises(ValueError, match="requires continuous"):
        ILPSearch().fit(df_discrete)


def test_cancer_data_unconstrained(cancer_data):
    est = ILPSearch(l_penalty=0.0001)
    est.fit(cancer_data)

    # Recovers all 4 true adjacencies connected to Cancer with zero false positive edges
    expected_edges = {
        ("Cancer", "Dyspnoea"),
        ("Cancer", "Pollution"),
        ("Cancer", "Smoker"),
        ("Cancer", "Xray"),
    }
    assert set(est.causal_graph_.edges()) == expected_edges

    # Columns: ['Cancer', 'Pollution', 'Smoker', 'Xray', 'Dyspnoea']
    # Cancer (row 0) points to Pollution (1), Smoker (2), Xray (3), Dyspnoea (4)
    expected_adj = np.array(
        [
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(est.adjacency_matrix_.to_numpy(), expected_adj)


def test_cancer_data_expert_knowledge(cancer_data):
    # Enforcing upstream causes directs edges into Cancer, recovering the exact ground truth DAG
    ek = ExpertKnowledge(required_edges=[("Pollution", "Cancer"), ("Smoker", "Cancer")])
    est = ILPSearch(l_penalty=0.0001, expert_knowledge=ek)
    est.fit(cancer_data)

    expected_edges = {
        ("Pollution", "Cancer"),
        ("Smoker", "Cancer"),
        ("Cancer", "Xray"),
        ("Cancer", "Dyspnoea"),
    }
    assert set(est.causal_graph_.edges()) == expected_edges

    # Columns: ['Cancer', 'Pollution', 'Smoker', 'Xray', 'Dyspnoea']
    expected_adj = np.array(
        [
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(est.adjacency_matrix_.to_numpy(), expected_adj)


def test_sachs_data_unconstrained(sachs_data):
    est = ILPSearch(l_penalty=0.005)
    est.fit(sachs_data)

    assert len(est.causal_graph_.edges()) > 0
    assert ("Raf", "Mek") in est.causal_graph_.edges()
    assert ("Plcg", "PIP2") in est.causal_graph_.edges()


def test_sachs_data_expert_knowledge(sachs_data):

    # Providing cellular signaling cascade tiers ensures strictly forward causal flow
    temporal_tiers = [
        ["PKC", "Plcg"],
        ["PKA", "PIP3", "Raf"],
        ["Mek", "PIP2", "Jnk", "P38"],
        ["Erk"],
        ["Akt"],
    ]
    ek = ExpertKnowledge(temporal_order=temporal_tiers)
    est = ILPSearch(l_penalty=0.005, expert_knowledge=ek)
    est.fit(sachs_data)

    # Verifies forward cascade orientation
    assert ("Mek", "Erk") in est.causal_graph_.edges()
    assert ("Erk", "Akt") in est.causal_graph_.edges()
    assert ("Raf", "Mek") in est.causal_graph_.edges()
    assert ("PKC", "PKA") in est.causal_graph_.edges()
