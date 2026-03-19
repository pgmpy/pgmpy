import pytest

from pgmpy.ci_tests.independence_match import IndependenceMatch
from pgmpy.independencies import Independencies


@pytest.mark.parametrize(
    "Z, expect_independent",
    [
        ([], True),
        (["Z"], True),
        (["W"], False),
    ],
)
def test_independence_match_with_init_independencies(Z, expect_independent):
    independencies = Independencies(("X", "Y"), ("X", "Y", "Z"))
    ci_test = IndependenceMatch(independencies=independencies)

    independent = ci_test.test(X="X", Y="Y", Z=Z, boolean=True)
    assert (
        independent == expect_independent
    ), f"Expected independence={expect_independent}, got {independent}"
