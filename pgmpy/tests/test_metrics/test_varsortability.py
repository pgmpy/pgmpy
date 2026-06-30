import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.example_models import load_model
from pgmpy.metrics import VarSortability


@pytest.fixture
def causal_chain_increasing_variance():
    np.random.seed(42)
    n = 1000
    x = np.random.normal(0, 1.0, n)
    y = 2.5 * x + np.random.normal(0, 0.5, n)
    z = 2.0 * y + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


@pytest.fixture
def causal_chain_decreasing_variance():
    np.random.seed(123)
    n = 1000
    x = np.random.normal(0, 5.0, n)
    y = 0.1 * x + np.random.normal(0, 0.5, n)
    z = 0.1 * y + np.random.normal(0, 0.3, n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


@pytest.fixture
def cancer_model_and_data():
    model = load_model("bnlearn/cancer")
    data = model.simulate(1000, seed=42, show_progress=False)
    return model, data


def test_varsortability_high_score_for_increasing_variance(causal_chain_increasing_variance):
    true_dag = DAG([("X", "Y"), ("Y", "Z")])
    scorer = VarSortability()
    result = scorer.evaluate(causal_chain_increasing_variance, true_dag)

    assert "varsortability" in result
    assert isinstance(result["varsortability"], float)
    assert 0.7 < result["varsortability"] <= 1.0


def test_varsortability_low_score_for_decreasing_variance(causal_chain_decreasing_variance):
    true_dag = DAG([("X", "Y"), ("Y", "Z")])
    scorer = VarSortability()
    result = scorer.evaluate(causal_chain_decreasing_variance, true_dag)

    assert result["varsortability"] < 0.3


def test_varsortability_vacuously_true_for_empty_dag():
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 3), columns=["X", "Y", "Z"])

    empty_dag = DAG()
    empty_dag.add_nodes_from(["X", "Y", "Z"])

    scorer = VarSortability()
    result = scorer.evaluate(data, empty_dag)

    assert result["varsortability"] == 1.0


def test_varsortability_evaluate_method(causal_chain_increasing_variance):
    true_dag = DAG([("X", "Y"), ("Y", "Z")])
    scorer = VarSortability()
    result = scorer.evaluate(causal_chain_increasing_variance, true_dag)

    assert isinstance(result, dict)
    assert "varsortability" in result


def test_varsortability_call_method(causal_chain_increasing_variance):
    true_dag = DAG([("X", "Y"), ("Y", "Z")])
    scorer = VarSortability()

    result_eval = scorer.evaluate(causal_chain_increasing_variance, true_dag)
    result_call = scorer(causal_chain_increasing_variance, true_dag)

    assert result_eval == result_call


def test_varsortability_invalid_graph_type(causal_chain_increasing_variance):
    scorer = VarSortability()
    with pytest.raises(ValueError):
        scorer.evaluate(causal_chain_increasing_variance, "not_a_graph")


def test_varsortability_mismatched_nodes(causal_chain_increasing_variance):
    scorer = VarSortability()
    true_dag = DAG([("A", "B"), ("B", "C")])  # Different nodes

    with pytest.raises(ValueError):
        scorer.evaluate(causal_chain_increasing_variance, true_dag)


def test_varsortability_invalid_data_type(causal_chain_increasing_variance):
    scorer = VarSortability()
    true_dag = DAG([("X", "Y"), ("Y", "Z")])

    with pytest.raises(ValueError):
        scorer.evaluate([1, 2, 3], true_dag)
