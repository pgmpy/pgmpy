#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest

from pgmpy import global_vars
from pgmpy.estimators.CITests import get_callable_ci_test
from pgmpy.metrics import implied_cis, permutation_t
from pgmpy.metrics.permutation_t import (
    _count_lmc_violations,
)
from pgmpy.models import DiscreteBayesianNetwork

logger = global_vars.logger
falsify_graph = permutation_t
permutation_based_falsification_test = permutation_t


@pytest.fixture
def data_simple():
    np.random.seed(42)
    n_samples = 500
    X = np.random.binomial(1, 0.5, n_samples)
    Y = np.random.binomial(1, 0.3 + 0.4 * X)
    Z = np.random.binomial(1, 0.2 + 0.6 * Y)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


@pytest.fixture
def model_simple():
    return DiscreteBayesianNetwork([("X", "Y"), ("Y", "Z")])


@pytest.fixture
def data_continuous():
    np.random.seed(42)
    n = 500
    X = np.random.normal(0, 1, n)
    Y = 0.5 * X + np.random.normal(0, 0.5, n)
    Z = 0.7 * Y + np.random.normal(0, 0.3, n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def test_with_return_summary(model_simple, data_simple):
    result = permutation_t(
        model_simple,
        data_simple,
        n_permutations=5,
        return_summary=True,
        show_progress=False,
    )

    assert "summary" in result
    summary = result["summary"]

    expected_keys = [
        "permutation_violations",
        "significance_level",
        "ci_test",
        "mean_permutation_violations",
        "std_permutation_violations",
        "min_permutation_violations",
        "max_permutation_violations",
    ]

    for key in expected_keys:
        assert key in summary

    assert isinstance(summary["permutation_violations"], list)
    assert len(summary["permutation_violations"]) == 5


def test_continuous_data_support(model_simple, data_continuous):
    result = permutation_t(
        model_simple,
        data_continuous,
        ci_test="pearsonr",
        n_permutations=5,
        show_progress=False,
    )

    assert isinstance(result["falsifiable"], bool)
    assert isinstance(result["falsified"], bool)


def test_input_validation(model_simple, data_simple):
    with pytest.raises(TypeError):
        permutation_t(model_simple, "not_a_dataframe", show_progress=False)

    with pytest.raises(ValueError):
        bad_data = data_simple[["X", "Y"]]
        permutation_t(model_simple, bad_data, show_progress=False)

    with pytest.raises(ValueError):
        permutation_t(
            model_simple, data_simple, ci_test="unsupported_test", show_progress=False
        )


def test_edge_cases(model_simple, data_simple):
    r = permutation_t(model_simple, data_simple, n_permutations=1, show_progress=False)
    assert r["n_permutations"] == 1

    single_model = DiscreteBayesianNetwork()
    single_model.add_node("A")
    df = pd.DataFrame({"A": [0, 1, 0, 1]})
    result = permutation_t(single_model, df, n_permutations=3, show_progress=False)
    assert isinstance(result["falsifiable"], bool)


def test_wrong_model_detection(model_simple, data_simple):
    wrong_model = DiscreteBayesianNetwork([("Z", "Y"), ("Y", "X")])

    correct = permutation_t(
        model_simple, data_simple, n_permutations=10, show_progress=False
    )
    wrong = permutation_t(
        wrong_model, data_simple, n_permutations=10, show_progress=False
    )

    assert wrong["lmc_violations"] >= correct["lmc_violations"]


def test_progress_bar_enabled(monkeypatch):
    monkeypatch.setattr("pgmpy.config.SHOW_PROGRESS", True)
    model = DiscreteBayesianNetwork([("X", "Y")])
    df = pd.DataFrame({"X": [0, 1, 0, 1], "Y": [1, 1, 0, 0]})
    r = permutation_t(model, df, n_permutations=2, show_progress=True)
    assert isinstance(r["falsifiable"], bool)


def test_exception_handling_in_ci_test():
    model = DiscreteBayesianNetwork([("X", "Y"), ("Y", "Z")])
    df = pd.DataFrame({"X": [1, 1, 1, 1], "Y": [0, 1, 0, 1], "Z": [0, 0, 0, 0]})
    result = permutation_t(model, df, n_permutations=3, show_progress=False)

    assert isinstance(result["falsifiable"], bool)
    assert result["lmc_violations"] >= 0


# -------------------------------
# Helper Function Tests
# -------------------------------


@pytest.fixture
def model_helper():
    return DiscreteBayesianNetwork([("A", "B"), ("B", "C"), ("A", "D")])


@pytest.fixture
def data_helper():
    np.random.seed(42)
    n = 50
    A = np.random.binomial(1, 0.5, n)
    B = np.random.binomial(1, 0.3 + 0.4 * A)
    C = np.random.binomial(1, 0.2 + 0.6 * B)
    D = np.random.binomial(1, 0.1 + 0.7 * A)
    return pd.DataFrame({"A": A, "B": B, "C": C, "D": D})


def test_get_non_descendants(model_helper):
    assert set(model_helper._get_non_descendants("A")) == set()
    assert set(model_helper._get_non_descendants("B")) == {"A", "D"}
    assert set(model_helper._get_non_descendants("C")) == {"A", "B", "D"}
    assert set(model_helper._get_non_descendants("D")) == {"A", "B", "C"}


def test_count_lmc_violations(model_helper, data_helper):
    ci_test_chosen = get_callable_ci_test("chi_square", data=data_helper)
    count = _count_lmc_violations(
        data_helper,
        implied_cis(model_helper, data_helper, ci_test=ci_test_chosen),
        ci_test_chosen,
        significance_level=0.05,
    )
    assert isinstance(count, int)
    assert count >= 0


def test_count_lmc_violations_small_data(model_helper, data_helper):
    df = data_helper.head(5)
    ci_test_chosen = get_callable_ci_test("gcm", data=df)
    count = _count_lmc_violations(
        df,
        implied_cis(model_helper, df, ci_test=ci_test_chosen),
        ci_test_chosen,
        significance_level=0.05,
    )
    assert isinstance(count, int)
    assert count >= 0
