import pytest

from pgmpy.ci_tests.pearsonr import Pearsonr


@pytest.mark.parametrize(
    "data_key, Z, expect_independent",
    [
        ("cont_ind", [], True),  # Independent
        ("cont_cind", ["Z"], True),  # Conditionally Independent
        (
            "cont_cind_mul",
            ["Z1", "Z2"],
            True,
        ),  # Conditionally Independent with Multiple Parents
        ("cont_vstruct", ["Z"], False),  # V-structure (should be dependent)
    ],
)
def test_pearsonr(test_data, data_key, Z, expect_independent):
    data = test_data[data_key]
    ci_test = Pearsonr(data)

    coef, p_value = ci_test.test(X="X", Y="Y", Z=Z, boolean=False)

    if expect_independent:
        assert abs(coef) <= 0.1, f"Expected |coef| <= 0.1, got {coef}"
        assert p_value >= 0.05, f"Expected p-value >= 0.05, got {p_value}"
    else:
        assert abs(coef) > 0.9, f"Expected |coef| > 0.9 , got {coef}"
        assert p_value < 0.05, f"Expected p-value < 0.05, got {p_value}"

    independent = ci_test.test(X="X", Y="Y", Z=Z, boolean=True, significance_level=0.05)
    assert (
        independent == expect_independent
    ), f"Expected independence={expect_independent}, got {independent}"
