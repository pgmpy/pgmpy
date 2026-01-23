import networkx as nx
import numpy as np
import pytest

from pgmpy.base import DAG
from pgmpy.estimators.CITests import chi_square
from pgmpy.metrics import ImpliedCIs
from pgmpy.utils import get_example_model

SEED = 42
N_SAMPLES = 1_000
EDGE_PROB = 0.1


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(SEED)


def simulate(model, n_samples=N_SAMPLES, seed=SEED):
    return model.simulate(n_samples, seed=seed)


def random_dag_from_nodes(nodes, rng, edge_prob=EDGE_PROB):
    n = len(nodes)
    adj = np.tril(rng.choice([0, 1], p=[1 - edge_prob, edge_prob], size=(n, n)), k=-1)
    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    nx.relabel_nodes(g, {i: nodes[i] for i in range(n)}, copy=False)
    dag = DAG(g.edges())
    dag.add_nodes_from(nodes)
    return dag


def implied_ci_tests(X, causal_graph, ci_test=chi_square):
    return ImpliedCIs(ci_test=ci_test)(X=X, causal_graph=causal_graph)


def assert_pvalues(df, expected, col="p-value", ndigits=4):
    got = df[col].to_numpy().round(ndigits).tolist()
    assert got == expected


def test_implied_cis_cancer():
    model = get_example_model("cancer")
    df = simulate(model)

    tests = implied_ci_tests(df, model)
    assert tests.shape[0] == 6
    assert_pvalues(tests, [0.9816, 1.0, 0.3491, 0.8061, 0.8960, 0.9917])


def test_implied_cis_alarm_true_and_random(rng):
    model = get_example_model("alarm")
    df = simulate(model)

    tests_true = implied_ci_tests(df, model)
    assert tests_true.shape[0] == 620

    model_random = random_dag_from_nodes(list(model.nodes()), rng=rng)
    tests_random = implied_ci_tests(df, model_random)
    assert tests_random.shape[0] == 601
