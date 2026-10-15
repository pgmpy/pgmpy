import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process.kernels import RBF

from pgmpy.ci_tests import HSIC


@pytest.fixture
def hsic_data():
    rng = np.random.default_rng(seed=42)

    df_ind = pd.DataFrame(rng.standard_normal((200, 3)), columns=["X", "Y", "Z"])

    X = rng.normal(size=300)
    Y = 2 * X + rng.normal(scale=0.3, size=300)
    df_dep = pd.DataFrame({"X": X, "Y": Y})

    X = rng.normal(size=300)
    Y = X**2 + rng.normal(scale=0.3, size=300)
    df_nonlinear = pd.DataFrame({"X": X, "Y": Y})

    return df_ind, df_dep, df_nonlinear


class TestHSIC:
    def test_matches_causal_learn(self, hsic_data):
        # Reference values from causal-learn v0.1.4.5:
        # KCI_UInd().compute_pvalue(df[["X"]].to_numpy(), df[["Y"]].to_numpy())
        for df, statistic, p_value in zip(hsic_data, (122.8551, 8651.7687, 3634.9518), (0.1267, 0.0, 0.0)):
            test = HSIC(data=df)
            test.run_test("X", "Y", [])
            assert test.statistic_ == pytest.approx(statistic, abs=0.01)
            assert test.p_value_ == pytest.approx(p_value, abs=0.01)
            assert test.p_value_ > 0.0

    def test_options(self, hsic_data):
        df_ind, df_dep, _ = hsic_data
        for kwargs in (
            {"bandwidth": "median"},
            {"null_dist": "permutation", "n_permutations": 200, "seed": 0},
            {"kernel": RBF(2.0)},
        ):
            assert HSIC(data=df_ind, **kwargs)("X", "Y", [], significance_level=0.05), kwargs
            assert not HSIC(data=df_dep, **kwargs)("X", "Y", [], significance_level=0.05), kwargs

    def test_kernels(self, hsic_data):
        _, df_dep, _ = hsic_data
        # A single kernel is equivalent to a dict with that kernel for every column.
        per_column = HSIC(data=df_dep, kernel={"X": RBF(2.0), "Y": RBF(2.0)})
        assert HSIC(data=df_dep, kernel=RBF(2.0)).run_test("X", "Y", []) == per_column.run_test("X", "Y", [])
        # Columns missing from the dict use the default RBF kernel (length scale 0.5 for n=300).
        partial = HSIC(data=df_dep, kernel={"X": RBF(2.0)})
        full = HSIC(data=df_dep, kernel={"X": RBF(2.0), "Y": RBF(0.5)})
        assert partial.run_test("X", "Y", []) == full.run_test("X", "Y", [])
        assert partial.clone().run_test("X", "Y", []) == partial.run_test("X", "Y", [])

        # length_scale = median of the non-zero standardized distances / sqrt(2). Six of X's ten pairwise distances
        # are ties; the non-zero ones are all sqrt(5). Y's median distance is 2 / sqrt(2.5).
        df_small = pd.DataFrame({"X": [0.0, 0.0, 0.0, 0.0, 1.0], "Y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        test = HSIC(data=df_small, bandwidth="median")
        assert test.kernel_dict["X"].length_scale == pytest.approx(np.sqrt(2.5))
        assert test.kernel_dict["Y"].length_scale == pytest.approx(np.sqrt(0.8))
        # The Gamma null moments need n >= 6; smaller samples get p = 1.
        assert test.run_test("X", "Y", [])[1] == 1.0

    def test_invalid_params(self, hsic_data):
        df_ind, _, _ = hsic_data
        for kwargs in (
            {"bandwidth": "bad"},
            {"null_dist": "bad"},
            {"n_permutations": 0},
            {"kernel": (RBF(0.5), RBF(0.5))},
            {"kernel": {"X": "not-a-kernel"}},
            {"kernel": {"W": RBF(0.5)}},
        ):
            with pytest.raises(ValueError, match=next(iter(kwargs))):
                HSIC(data=df_ind, **kwargs)
