import pytest

from pgmpy.ci_tests.power_divergence import PowerDivergence


@pytest.mark.parametrize(
    "data_key, Z, expect_independent",
    [
        ("disc_ind", [], True),  # Independent
        ("disc_cind", ["Z"], True),  # Conditionally Independent
        (
            "disc_cind_mul",
            ["Z1", "Z2"],
            True,
        ),  # Conditionally Independent with Multiple Parents
        ("disc_vstruct", ["Z"], False),  # V-structure (should be dependent)
    ],
)
def test_power_divergence(test_data, data_key, Z, expect_independent):
    data = test_data[data_key]
    ci_test = PowerDivergence(data)

    result = ci_test.test(X="X", Y="Y", Z=Z, boolean=False)
    p_value = result[1]

    if expect_independent:
        assert p_value >= 0.05, f"Expected p-value >= 0.05, got {p_value}"
    else:
        assert p_value < 0.05, f"Expected p-value < 0.05, got {p_value}"

    independent = ci_test.test(X="X", Y="Y", Z=Z, boolean=True, significance_level=0.05)
    assert (
        independent == expect_independent
    ), f"Expected independence={expect_independent}, got {independent}"
