import pytest

from pgmpy.ci_tests.pillai_trace import PillaiTrace


@pytest.mark.parametrize(
    "data_key, Z, expect_independent",
    [
        ("cont_ind", [], True),
        ("cont_cind", ["Z"], True),
        ("cont_cind_mul", ["Z1", "Z2"], True),
        ("cont_vstruct", ["Z"], False),
        ("disc_ind", [], True),
        ("disc_cind", ["Z"], True),
        ("disc_cind_mul", ["Z1", "Z2"], True),
        ("disc_vstruct", ["Z"], False),
    ],
)
def test_pillai_trace(test_data, data_key, Z, expect_independent):
    pytest.importorskip("xgboost")

    data = test_data[data_key].sample(n=1000, random_state=42)

    ci_test = PillaiTrace(data, seed=42)
    coef, p_value = ci_test.test(X="X", Y="Y", Z=Z, boolean=False)

    if expect_independent:
        assert p_value >= 0.05, f"Expected p-value >= 0.05, got {p_value}"
    else:
        assert p_value < 0.05, f"Expected p-value < 0.05, got {p_value}"

    independent = ci_test.test(X="X", Y="Y", Z=Z, boolean=True, significance_level=0.05)
    assert (
        independent == expect_independent
    ), f"Expected independence={expect_independent}, got {independent}"
