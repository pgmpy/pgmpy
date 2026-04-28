import pandas as pd

from pgmpy.tests.test_ci_tests import _multivariate_fixtures


def test_build_pillai_data_is_seeded():
    first = _multivariate_fixtures._build_pillai_data()
    second = _multivariate_fixtures._build_pillai_data()

    assert first.keys() == second.keys()

    for key in first:
        assert len(first[key]) == len(second[key])

        for first_df, second_df in zip(first[key], second[key]):
            pd.testing.assert_frame_equal(first_df, second_df)
