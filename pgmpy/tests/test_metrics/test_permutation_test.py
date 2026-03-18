#!/usr/bin/env python

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators.CITests import ci_registry
from pgmpy.metrics import PermutationTest
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import get_example_model


@pytest.fixture
def data_simple():
    np.random.seed(42)
    n_samples = 500
    X = np.random.binomial(1, 0.5, n_samples)
    Y = np.random.binomial(1, 0.2 + 0.6 * X)
    Z = np.random.binomial(1, 0.2 + 0.6 * Y)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


@pytest.fixture
def model_simple():
    return DiscreteBayesianNetwork([("X", "Y"), ("Y", "Z")])


@pytest.fixture
def perm_test():
    return PermutationTest()


def test_input_validation(model_simple, data_simple, perm_test):
    with pytest.raises(ValueError):
        perm_test.evaluate("not_a_dataframe", model_simple, show_progress=False)

    with pytest.raises(ValueError):
        bad_data = data_simple[["X", "Y"]]
        perm_test.evaluate(bad_data, model_simple, show_progress=False)

    with pytest.raises(ValueError):
        perm_test.evaluate(
            data_simple, model_simple, ci_test="unsupported_test", show_progress=False
        )


def test_wrong_model_detection(model_simple, data_simple, perm_test):
    wrong_model = DiscreteBayesianNetwork([("Y", "Z"), ("X", "Z")])

    correct = perm_test.evaluate(
        data_simple, model_simple, n_permutations=10, show_progress=False
    )
    wrong = perm_test.evaluate(
        data_simple, wrong_model, n_permutations=10, show_progress=False
    )

    assert wrong["p_value_falsified"] > correct["p_value_falsified"]


# -------------------------------
# Helper Function Tests
# -------------------------------


@pytest.fixture
def model_helper():
    return DiscreteBayesianNetwork([("A", "B"), ("B", "C"), ("C", "D")])


@pytest.fixture
def data_helper():
    np.random.seed(42)
    n = 50
    A = np.random.binomial(1, 0.5, n)
    B = np.random.binomial(1, 0.3 + 0.4 * A)
    C = np.random.binomial(1, 0.2 + 0.6 * B)
    D = np.random.binomial(1, 0.1 + 0.7 * C)
    return pd.DataFrame({"A": A, "B": B, "C": C, "D": D})


@pytest.fixture
def child_model():
    model = get_example_model("child")
    data = model.simulate(n_samples=1000)

    return (model, data)


@pytest.fixture
def insurance_model():
    model = get_example_model("insurance")
    data = model.simulate(n_samples=1000)

    return (model, data)


def test_get_non_descendants(model_helper):
    assert set(model_helper._get_non_descendants("A")) == set()
    assert set(model_helper._get_non_descendants("B")) == {"A"}
    assert set(model_helper._get_non_descendants("C")) == {"A", "B"}
    assert set(model_helper._get_non_descendants("D")) == {"A", "B", "C"}
    assert set(model_helper._get_non_descendants("C", exclude_parents=True)) == {"A"}
    assert set(model_helper._get_non_descendants("D", exclude_parents=True)) == {
        "A",
        "B",
    }


def test_child_model(child_model, perm_test):
    model, data = child_model
    result = perm_test.evaluate(data, model, return_summary=True, show_progress=False)
    assert result["p_value_falsifiable"] <= 0.05
    assert result["p_value_falsified"] <= 0.05


def test_insurance_model(insurance_model, perm_test):
    model, data = insurance_model
    result = perm_test.evaluate(data, model, return_summary=True, show_progress=False)
    assert result["p_value_falsifiable"] <= 0.05
    assert result["p_value_falsified"] <= 0.05


def test_lmc_violations(model_helper, data_helper, perm_test):
    ci_test_chosen = ci_registry.get_test("chi_square", data=data_helper)
    n_violations, n_triples = perm_test._lmc_violations(
        model_helper, data_helper, ci_test_chosen, significance_level=0.05
    )
    assert isinstance(n_violations, int)
    assert n_violations == 0
    assert n_triples > 0

    for _ in range(5):
        nodes = list(model_helper.nodes())
        permuted_nodes = np.random.permutation(nodes)
        perm_mapping = dict(zip(nodes, permuted_nodes))
        nx_permuted_dag = nx.relabel_nodes(model_helper, perm_mapping, copy=True)
        permuted_dag = type(model_helper)()
        permuted_dag.add_nodes_from(nx_permuted_dag.nodes())
        permuted_dag.add_edges_from(nx_permuted_dag.edges())
        n_violations_perm, _ = perm_test._lmc_violations(
            permuted_dag, data_helper, ci_test_chosen, significance_level=0.05
        )
        assert n_violations_perm >= n_violations or n_violations_perm >= 0


def test_tpa_violations(model_helper, perm_test):
    nodes = list(model_helper.nodes())
    permuted_nodes = np.random.permutation(nodes)
    perm_mapping = dict(zip(nodes, permuted_nodes))
    nx_permuted_dag = nx.relabel_nodes(model_helper, perm_mapping, copy=True)
    permuted_dag = type(model_helper)()
    permuted_dag.add_nodes_from(nx_permuted_dag.nodes())
    permuted_dag.add_edges_from(nx_permuted_dag.edges())
    n_violations, n_triples = perm_test._tpa_violations(permuted_dag, model_helper)
    assert isinstance(n_violations, int)
    assert n_violations >= 0
    assert n_triples > 0


def test_all_permuations(model_helper, data_helper, perm_test):
    results = perm_test.evaluate(
        data_helper,
        model_helper,
        n_permutations=-1,
        show_progress=False,
        return_summary=False,
    )
    assert results["n_lmc_violations"] == 0
    assert results["n_within_mec"] == 2

    results_2 = perm_test.evaluate(
        data_helper,
        model_helper,
        n_permutations=10_000,
        show_progress=False,
        return_summary=False,
    )
    assert results_2["n_lmc_violations"] == 0
    assert results_2["n_within_mec"] == 2
