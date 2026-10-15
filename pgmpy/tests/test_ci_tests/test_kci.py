import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process.kernels import RBF, DotProduct

from pgmpy.base import DAG
from pgmpy.ci_tests import HSIC, KCI, get_ci_test


@pytest.fixture
def kci_data():
    df_ind = pd.DataFrame(np.random.default_rng(seed=42).standard_normal((200, 3)), columns=["X", "Y", "Z"])
    df_cind = DAG.from_dagitty("dag{ Z -> X [beta=0.5]; Z -> Y [beta=0.8] }").simulate(n_samples=500, seed=42)
    df_vstruct = DAG.from_dagitty("dag{ X -> Z [beta=0.5]; Y -> Z [beta=0.8] }").simulate(n_samples=800, seed=42)
    df_multi = DAG.from_dagitty(
        "dag{ Z1 -> X [beta=0.6]; Z2 -> X [beta=-0.4]; Z1 -> Y [beta=0.3]; Z2 -> Y [beta=0.7]; X -> Y [beta=0.3] }"
    ).simulate(n_samples=400, seed=42)
    return df_ind, df_cind, df_vstruct, df_multi


class TestKCI:
    def test_matches_causal_learn(self, kci_data):
        df_ind, df_cind, df_vstruct, df_multi = kci_data
        # Reference statistics and p-values from causal-learn v0.1.4.5:
        # KCI_CInd().compute_pvalue(df[["X"]].to_numpy(), df[["Y"]].to_numpy(), df[Z].to_numpy())
        # causal-learn reports no effect size; the expected effect sizes are the normalized KCI statistic.
        for df, Z, statistic, p_value, effect_size in (
            (df_ind, ["Z"], 71.2603, 0.1883, 0.0271),
            (df_cind, ["Z"], 95.9394, 0.5655, 0.0076),
            (df_vstruct, ["Z"], 832.8506, 0.0, 0.0234),
            (df_multi, ["Z1", "Z2"], 458.2227, 0.0, 0.1063),
        ):
            test = KCI(data=df)
            test.run_test("X", "Y", Z)
            assert test.statistic_ == pytest.approx(statistic, abs=0.01)
            assert test.p_value_ == pytest.approx(p_value, abs=0.01)
            assert test.effect_size_ == pytest.approx(effect_size, abs=1e-4)

        # Without conditioning variables, KCI is HSIC.
        assert KCI(data=df_ind).run_test("X", "Y", []) == HSIC(data=df_ind).run_test("X", "Y", [])
        assert isinstance(get_ci_test(test="kci", data=df_ind), KCI)

    def test_options(self, kci_data):
        _, df_cind, df_vstruct, _ = kci_data
        for bandwidth in ("heuristic", "median"):
            cind, vstruct = KCI(data=df_cind, bandwidth=bandwidth), KCI(data=df_vstruct, bandwidth=bandwidth)
            assert cind("X", "Y", ["Z"], significance_level=0.05), bandwidth
            assert not vstruct("X", "Y", ["Z"], significance_level=0.05), bandwidth
            assert 0.0 <= cind.effect_size_ < vstruct.effect_size_ <= 1.0, bandwidth

        # Sample sizes on the other sides of the heuristic width thresholds (200 and 1200).
        lgbn = DAG.from_dagitty("dag{ Z -> X [beta=0.5]; Z -> Y [beta=0.8] }")
        for n_samples in (150, 1250):
            test = KCI(data=lgbn.simulate(n_samples=n_samples, seed=42))
            assert not test("X", "Y", [], significance_level=0.05), n_samples
            assert test("X", "Y", ["Z"], significance_level=0.05), n_samples

    def test_kernels(self, kci_data):
        _, df_cind, _, df_multi = kci_data
        # The default length scale is 0.7 for n=500 and one conditioning variable. A single kernel is used for X, Y,
        # and Z; a dict only sets the X and Y kernels, so Z keeps the default RBF kernel.
        default = KCI(data=df_cind).run_test("X", "Y", ["Z"])
        assert KCI(data=df_cind, kernel=RBF(0.7)).run_test("X", "Y", ["Z"]) == default
        assert KCI(data=df_cind, kernel={"X": RBF(0.7), "Y": RBF(0.7)}).run_test("X", "Y", ["Z"]) == default
        assert KCI(data=df_cind, kernel={"Z": RBF(5.0)}).run_test("X", "Y", ["Z"]) == default
        partial = KCI(data=df_cind, kernel={"X": RBF(2.0)})
        assert partial.clone().run_test("X", "Y", ["Z"]) == partial.run_test("X", "Y", ["Z"])

        # With linear kernels, the effect size is the squared partial correlation of X and Y given Z.
        linear = KCI(data=df_multi, kernel=DotProduct())
        linear.run_test("X", "Y", ["Z1", "Z2"])
        A = np.column_stack([np.ones(len(df_multi)), df_multi[["Z1", "Z2"]]])
        rx, ry = (df_multi[col] - A @ np.linalg.lstsq(A, df_multi[col], rcond=None)[0] for col in ("X", "Y"))
        assert linear.effect_size_ == pytest.approx(np.corrcoef(rx, ry)[0, 1] ** 2, rel=1e-4)

    def test_invalid_params(self, kci_data):
        df_ind, _, _, _ = kci_data
        for kwargs in (
            {"epsilon": 0},
            {"epsilon": np.inf},
            {"bandwidth": "bad"},
            {"kernel": (RBF(0.5), RBF(0.5), RBF(0.5))},
        ):
            with pytest.raises(ValueError, match=next(iter(kwargs))):
                KCI(data=df_ind, **kwargs)
