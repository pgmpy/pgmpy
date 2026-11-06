import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tigramite")

from tigramite.independence_tests.parcorr import ParCorr

from pgmpy.causal_discovery.PCMCI import PCMCI


class TestPCMCI:
    @pytest.fixture
    def data(self):
        """Create a multivariate time-series dataset."""
        rng = np.random.default_rng(42)
        n_samples = 100

        x = rng.normal(size=n_samples)
        y = 0.8 * x + rng.normal(scale=0.5, size=n_samples)

        return pd.DataFrame(
            {"x": x, "y": y},
            index=pd.MultiIndex.from_arrays(
                [
                    ["series_1"] * n_samples,
                    np.arange(n_samples),
                ],
                names=["series_id", "time"],
            ),
        )

    @pytest.fixture
    def estimator(self):
        """Create a PCMCI estimator with a Tigramite CI test."""
        return PCMCI(
            cond_ind_test=ParCorr(),
            tau_min=0,
            tau_max=1,
            pc_alpha=0.05,
            fdr_method="fdr_bh",
            verbosity=0,
            link_assumptions=None,
        )

    def test_fit(self, estimator, data):
        """Verify that PCMCI fits and exposes discovery results."""
        estimator.fit(data)

        expected_shape = (2, 2, 2)

        assert estimator.graph_.shape == expected_shape
        assert estimator.p_matrix_.shape == expected_shape
        assert estimator.val_matrix_.shape == expected_shape

        assert estimator.get_causal_links() is estimator.graph_
