#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators.CITests import ci_registry
from pgmpy.metrics import implied_cis, permutation_test
from pgmpy.metrics.permutation_test import (
    _count_lmc_violations,
    _create_permuted_CIs,
)
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


def test_input_validation(model_simple, data_simple):
    with pytest.raises(TypeError):
        permutation_test(model_simple, "not_a_dataframe", show_progress=False)

    with pytest.raises(ValueError):
        bad_data = data_simple[["X", "Y"]]
        permutation_test(model_simple, bad_data, show_progress=False)

    with pytest.raises(ValueError):
        permutation_test(
            model_simple, data_simple, ci_test="unsupported_test", show_progress=False
        )


def test_wrong_model_detection(model_simple, data_simple):
    wrong_model = DiscreteBayesianNetwork([("Y", "Z"), ("X", "Z")])

    correct = permutation_test(
        model_simple, data_simple, n_permutations=10, show_progress=False
    )
    wrong = permutation_test(
        wrong_model, data_simple, n_permutations=10, show_progress=False
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


def test_count_lmc_violations(model_helper, data_helper):
    ci_test_chosen = ci_registry.get_test("chi_square", data=data_helper)
    implied_CIs = implied_cis(model_helper, data_helper, ci_test=ci_test_chosen)
    count = _count_lmc_violations(
        data_helper,
        implied_CIs,
        ci_test_chosen,
        significance_level=0.05,
    )
    assert count == 0

    for _ in range(5):
        permuted_CIs = _create_permuted_CIs(implied_CIs, list(model_helper.nodes()))
        count = _count_lmc_violations(
            data_helper,
            permuted_CIs,
            ci_test_chosen,
            significance_level=0.05,
        )
        assert count > 0


def test_count_lmc_violations_small_data(model_helper, data_helper):
    df = data_helper.head(5)
    ci_test_chosen = ci_registry.get_test("chi_square", data=df)
    count = _count_lmc_violations(
        df,
        implied_cis(model_helper, df, ci_test=ci_test_chosen),
        ci_test_chosen,
        significance_level=0.05,
    )
    assert isinstance(count, int)
    assert count >= 0


def test_child_model(child_model):
    model, data = child_model
    result = permutation_test(model, data, return_summary=True, show_progress=False)
    assert result["p_value_falsifiable"] <= 0.05
    assert result["p_value_falsified"] <= 0.05


def test_insurance_model(insurance_model):
    model, data = insurance_model
    result = permutation_test(model, data, return_summary=True, show_progress=False)
    assert result["p_value_falsifiable"] <= 0.05
    assert result["p_value_falsified"] <= 0.05
