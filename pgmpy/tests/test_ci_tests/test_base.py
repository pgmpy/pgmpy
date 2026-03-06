import pandas as pd
import pytest

from pgmpy.ci_tests import get_ci_test
from pgmpy.ci_tests.chi_square import ChiSquare
from pgmpy.ci_tests.independence_match import IndependenceMatch
from pgmpy.ci_tests.pearsonr import Pearsonr
from pgmpy.ci_tests.pillai_trace import PillaiTrace


def _discrete_data():
    return pd.DataFrame(
        {
            "X": pd.Categorical([0, 1, 0, 1]),
            "Y": pd.Categorical([1, 0, 1, 0]),
        }
    )


def _continuous_data():
    return pd.DataFrame({"X": [0.1, 0.2, 0.3, 0.4], "Y": [1.0, 1.1, 1.2, 1.3]})


def _mixed_data():
    return pd.DataFrame(
        {
            "X": [0.1, 0.2, 0.3, 0.4],
            "Y": pd.Categorical([1, 0, 1, 0]),
        }
    )


def test_get_ci_test_returns_input_instance_as_is():
    ci_test = ChiSquare(data=_discrete_data())
    assert get_ci_test(test=ci_test) is ci_test


@pytest.mark.parametrize(
    "data, expected_cls",
    [
        (_discrete_data(), ChiSquare),
        (_continuous_data(), Pearsonr),
        (_mixed_data(), PillaiTrace),
    ],
)
def test_get_ci_test_infers_default_test_from_data_type(data, expected_cls):
    ci_test = get_ci_test(data=data)
    assert isinstance(ci_test, expected_cls)


def test_get_ci_test_string_lookup_is_case_insensitive():
    ci_test = get_ci_test(test="CHI_SQUARE", data=_discrete_data())
    assert isinstance(ci_test, ChiSquare)


def test_get_ci_test_instantiates_non_data_test_without_data():
    ci_test = get_ci_test(test="independence_match")
    assert isinstance(ci_test, IndependenceMatch)
