import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process.kernels import RBF

from pgmpy.base import DAG
from pgmpy.ci_tests import HSIC, KCI, get_ci_test


@pytest.fixture
def kci_data():
    df_ind = pd.DataFrame(
        np.random.default_rng(seed=42).standard_normal((200, 3)),
        columns=["X", "Y", "Z"],
    )

    lgbn_cind = DAG.from_dagitty("dag{ Z -> X [beta=0.5]; Z -> Y [beta=0.8] }")
    df_cind = lgbn_cind.simulate(n_samples=500, seed=42)

    lgbn_vstruct = DAG.from_dagitty("dag{ X -> Z [beta=0.5]; Y -> Z [beta=0.8] }")
    df_vstruct = lgbn_vstruct.simulate(n_samples=800, seed=42)

    return df_ind, df_cind, df_vstruct


class TestKCI:
    def test_clone(self, kci_data):
        df_ind, _, _ = kci_data

        assert HSIC(data=df_ind).clone().kernel is None
        assert KCI(data=df_ind).clone().kernel is None

    def test_unconditional_fallback(self, kci_data):
        df_ind, _, _ = kci_data
        stat_kci, p_kci = KCI(data=df_ind).run_test("X", "Y", [])
        stat_hsic, p_hsic = HSIC(data=df_ind).run_test("X", "Y", [])

        assert stat_kci == stat_hsic
        assert p_kci == p_hsic
        assert isinstance(get_ci_test(test="kci", data=df_ind), KCI)

    def test_conditional(self, kci_data):
        _, df_cind, df_vstruct = kci_data

        # Conditionally independent: X _|_ Y | Z
        test = KCI(data=df_cind)
        assert test("X", "Y", ["Z"], significance_level=0.05)
        assert test.statistic_ == pytest.approx(86.1546, abs=0.01)
        assert test.p_value_ == pytest.approx(0.6499, abs=0.01)

        # V-structure: conditioning on collider induces dependence
        test = KCI(data=df_vstruct)
        assert not test("X", "Y", ["Z"], significance_level=0.05)
        assert test.statistic_ == pytest.approx(542.8695, abs=0.01)
        assert test.p_value_ < 0.05

        # Constant conditioning columns are ignored without numerical warnings.
        df_cind = df_cind.assign(C=1.0)
        with np.errstate(divide="raise", invalid="raise"):
            assert KCI(data=df_cind)("X", "Y", ["Z", "C"], significance_level=0.05)

    def test_custom_kernels(self, kci_data):
        _, _, df_vstruct = kci_data
        # Single kernel shared by X, Y, Z.
        assert not KCI(data=df_vstruct, kernel=RBF(0.5))("X", "Y", ["Z"], significance_level=0.05)
        # Tuple: separate kernels for X, Y, Z.
        assert not KCI(data=df_vstruct, kernel=(RBF(0.5), RBF(0.5), RBF(0.5)))("X", "Y", ["Z"], significance_level=0.05)
        with pytest.raises(ValueError, match="kernel"):
            KCI(data=df_vstruct, kernel=(RBF(0.5), RBF(0.5), "not-a-kernel"))

    def test_sample_size_and_kci_bandwidth(self):
        lgbn = DAG.from_dagitty("dag{ Z -> X [beta=0.5]; Z -> Y [beta=0.8] }")

        df_small = lgbn.simulate(n_samples=150, seed=42)
        test = KCI(data=df_small)
        assert not test("X", "Y", [], significance_level=0.05)
        assert test("X", "Y", ["Z"], significance_level=0.05)

        df_large = lgbn.simulate(n_samples=1250, seed=42)
        assert KCI(data=df_large)("X", "Y", ["Z"], significance_level=0.05)
        assert KCI(data=df_large, bandwidth="median")("X", "Y", ["Z"], significance_level=0.05)

    def test_invalid_epsilon(self, kci_data):
        df_ind, _, _ = kci_data

        with pytest.raises(ValueError, match="epsilon"):
            KCI(data=df_ind, epsilon=0)


class TestKCICompareCausalLearn:
    """Cross-validate against causal-learn v0.1.4.5 (numpy 2.4.3).

    Reproduction code for reference values::

        import numpy as np
        from causallearn.utils.KCI.KCI import KCI_CInd

        rng = np.random.default_rng(seed=7)
        Z = rng.normal(size=500)
        X = 0.5*Z + rng.normal(scale=0.5, size=500)
        Y = 0.8*Z + rng.normal(scale=0.5, size=500)
        KCI_CInd().compute_pvalue(X[:,None], Y[:,None], Z[:,None])  # stat=69.4555, p=0.4918

        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300); Y = rng.normal(size=300)
        Z = 0.5*X + 0.8*Y + rng.normal(scale=0.3, size=300)
        KCI_CInd().compute_pvalue(X[:,None], Y[:,None], Z[:,None])  # stat=637.5696, p=0.0
    """

    def test_matches_causal_learn_kci_cind(self):
        rng = np.random.default_rng(seed=7)
        Z = rng.normal(size=500)
        X = 0.5 * Z + rng.normal(scale=0.5, size=500)
        Y = 0.8 * Z + rng.normal(scale=0.5, size=500)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        test = KCI(data=df)
        test.run_test("X", "Y", ["Z"])
        assert test.statistic_ == pytest.approx(69.4555, abs=0.01)
        assert test.p_value_ == pytest.approx(0.4918, abs=0.01)

        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300)
        Y = rng.normal(size=300)
        Z = 0.5 * X + 0.8 * Y + rng.normal(scale=0.3, size=300)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        test = KCI(data=df)
        test.run_test("X", "Y", ["Z"])
        assert test.statistic_ == pytest.approx(637.5696, abs=0.01)
        assert 0.0 < test.p_value_ < 0.001
