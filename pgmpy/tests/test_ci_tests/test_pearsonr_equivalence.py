import pytest

from pgmpy.ci_tests.pearsonr_equivalence import PearsonrEquivalence


@pytest.mark.parametrize(
    "data_key, Z, delta, expect_independent",
    [
        ("cont_ind", [], 0.1, True),  # Independent
        ("cont_cind", ["Z"], 0.1, True),  # Conditionally Independent
        (
            "cont_cind_mul",
            ["Z1", "Z2"],
            0.1,
            True,
        ),  # Conditionally Independent with Multiple Parents
        ("cont_vstruct", ["Z"], 0.1, False),  # V-structure (should be dependent)
        (
            "cont_vstruct",
            ["Z"],
            0.999999,
            True,
        ),  # V-structure with higher delta (should be independent)
    ],
)
def test_pearsonr_equivalence(test_data, data_key, Z, delta, expect_independent):
    data = test_data[data_key]
    ci_test = PearsonrEquivalence(data, delta_threshold=delta)

    coef, p_value = ci_test.test(X="X", Y="Y", Z=Z, boolean=False)

    if expect_independent:
        assert p_value < 0.05, f"Expected p-value < 0.05, got {p_value}"
    else:
        assert p_value >= 0.05, f"Expected p-value >= 0.05, got {p_value}"

    independent = ci_test.test(X="X", Y="Y", Z=Z, boolean=True, significance_level=0.05)
    assert (
        independent == expect_independent
    ), f"Expected independence={expect_independent}, got {independent}"
