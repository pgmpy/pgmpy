import networkx as nx
import numpy as np
import pytest

from pgmpy.base import DAG
from pgmpy.estimators.CITests import chi_square
from pgmpy.metrics import FisherC
from pgmpy.utils import get_example_model

SEED = 42
N_SAMPLES = 1_000
EDGE_PROB = 0.1  # probability of an edge in the lower triangle


def _random_dag_with_same_nodes(model, rng, edge_prob: float = EDGE_PROB) -> DAG:
    """Generate a random DAG over the same node set as `model`."""
    nodes = list(model.nodes())
    n = len(nodes)

    adj = np.tril(rng.choice([0, 1], p=[1 - edge_prob, edge_prob], size=(n, n)), k=-1)
    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    nx.relabel_nodes(g, {i: nodes[i] for i in range(n)}, copy=False)

    return DAG(g.edges())


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="module")
def models_and_data(rng):
    out = {}
    for name in ("cancer", "alarm"):
        model = get_example_model(name)
        out[name] = {
            "true": model,
            "random": _random_dag_with_same_nodes(model, rng),
            "data": model.simulate(N_SAMPLES, seed=SEED),
        }
    return out


@pytest.mark.parametrize(
    "model_name, graph_key, ndigits, expected",
    [
        ("cancer", "true", 4, 0.9967),
        ("cancer", "random", 4, 0.0001),
        ("alarm", "true", 4, 0.0005),
        ("alarm", "random", 4, 0.0),
    ],
)
def test_fisherc(models_and_data, model_name, graph_key, ndigits, expected):
    bundle = models_and_data[model_name]
    p_value = FisherC(ci_test=chi_square).evaluate(
        X=bundle["data"], causal_graph=bundle[graph_key]
    )
    assert round(p_value, ndigits) == expected


@pytest.mark.parametrize(
    "model_name, graph_key, ndigits, expected_pval, expected_rmsea",
    [
        ("cancer", "true", 4, 0.9967, 0),
        ("cancer", "random", 4, 0.0001, 0.0602),
        ("alarm", "true", 4, 0.0005, 0.0117),
        ("alarm", "random", 4, 0.0, 0.0476),
    ],
)
def test_rmsea(
    models_and_data, model_name, graph_key, ndigits, expected_pval, expected_rmsea
):
    bundle = models_and_data[model_name]
    p_value, rmsea = FisherC(ci_test=chi_square, compute_rmsea=True).evaluate(
        X=bundle["data"], causal_graph=bundle[graph_key]
    )
    assert round(p_value, ndigits) == expected_pval
    assert round(rmsea, ndigits) == expected_rmsea
