import warnings

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.distributions.nominal import NominalDistribution


class TestNominalDistribution:
    """Tests for Nominal distributions."""

    def test_default(self):
        probs = [[0.1, 0.9], [0.7, 0.3]]
        categories = ["A", "B"]
        dist = NominalDistribution(probs, categories)

        assert dist.name == "NominalDistribution"
        assert dist.get_class_tag("python_version") is None
        assert dist.get_class_tag("python_dependencies") is None
        assert dist.get_class_tag("distr:measuretype") == "discrete"
        assert dist.get_class_tag("distr:paramtype") == "parametric"
        assert dist.get_class_tag("capabilities:approx") == []
        assert dist.get_class_tag("capabilities:exact") == [
            "pmf",
            "log_pmf",
        ]
        assert dist.get_class_tag("broadcast_init") == "off"

    def test_public_import(self):
        """The class is importable from the package, not just the module."""
        from pgmpy.distributions import NominalDistribution as PublicCat

        assert PublicCat is NominalDistribution

    def test_interface_compatibility(self):
        """ensure interface compatibility by skpro.utils.estimator_checks.check_estimator"""
        from skpro.utils.estimator_checks import check_estimator

        probs = [[0.1, 0.9], [0.7, 0.3]]
        dist = NominalDistribution(probs=probs, categories=[1, 2])
        check_estimator(dist, raise_exceptions=True, verbose=False)

        probs = [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]]
        dist = NominalDistribution(probs=probs, categories=[1, 2, 3])
        check_estimator(dist, raise_exceptions=True, verbose=False)

        # Nominal categories with string labels must also pass the interface checks.
        probs = [[0.1, 0.9], [0.7, 0.3]]
        dist = NominalDistribution(probs=probs, categories=["A", "B"])
        check_estimator(dist, raise_exceptions=True, verbose=False)

    def test_init(self):
        """test"""
        # Case 0: default
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, "C"]

        with pytest.raises(TypeError) as exc:
            dist = NominalDistribution(probs=probs, categories=categories)
        # message must refer to categories, not the renamed `probs` parameter
        assert "probs" not in str(exc.value)

        # Case 1: probs: list
        probs = [[0.1, 0.9], [0.7, 0.3]]
        categories = [1, 2]

        dist = NominalDistribution(probs=probs, categories=categories)

        assert dist.probs == probs
        assert dist.categories == categories
        assert dist.columns == ["variable"]

        # Case 2: probs: numpy
        probs = [[0.1, 0.9], [0.7, 0.3]]
        probs = np.asarray(probs, dtype=float)
        categories = [1, 2]

        dist = NominalDistribution(probs=probs, categories=categories)

        assert dist.categories == categories
        assert dist.columns == ["variable"]

        # Case 3: categories: str
        probs = [[0.1, 0.9], [0.7, 0.3]]
        categories = ["A", "B"]

        dist = NominalDistribution(probs=probs, categories=categories)

        assert dist.probs == probs
        assert dist.categories == categories
        assert dist.columns == ["variable"]

        # Case 4: categories: int
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]

        dist = NominalDistribution(probs=probs, categories=categories)

        assert dist.probs == probs
        assert dist.categories == categories
        assert dist.columns == ["variable"]

        # Case 5: index
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = ["A", "B", "C"]

        dist = NominalDistribution(probs=probs, categories=categories, index=["studentA", "studentB"])

        assert dist.probs == probs
        assert dist.categories == categories
        assert list(dist.index) == ["studentA", "studentB"]
        assert list(dist.columns) == ["variable"]

        # Case 6: columns
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = ["A", "B", "C"]

        dist = NominalDistribution(
            probs=probs, categories=categories, index=["studentA", "studentB"], columns=["grade"]
        )

        assert dist.probs == probs
        assert dist.categories == categories
        assert list(dist.index) == ["studentA", "studentB"]
        assert list(dist.columns) == ["grade"]

        # Case 7: wrong probs
        probs = [[0.1, 0.2, 0.9], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]

        with pytest.raises(ValueError):
            dist = NominalDistribution(probs=probs, categories=categories)

        # Case 8: wrong categories (non-unique)
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 1, 2]

        with pytest.raises(ValueError) as exc:
            dist = NominalDistribution(probs=probs, categories=categories)
        # message must refer to categories, not the renamed `probs` parameter
        assert "probs" not in str(exc.value)

        # Case 9: wrong index
        probs = [[0.1, 0.2, 0.8], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        index = ["A", "B", "C"]

        with pytest.raises(ValueError):
            dist = NominalDistribution(probs=probs, categories=categories, index=index)

        # Case 10: wrong columns
        probs = [[0.1, 0.2, 0.8], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        columns = ["A", "B", "C"]

        with pytest.raises(ValueError):
            dist = NominalDistribution(probs=probs, categories=categories, columns=columns)

        # Case 13: wrong shape(probs, categories)
        probs = [[0.1, 0.2], [0.5, 0.3]]
        categories = [1, 2, 3]

        with pytest.raises(ValueError):
            dist = NominalDistribution(probs=probs, categories=categories)

        # Case 14: wrong probs(negative)
        probs = [[0.1, 0.2], [-0.5, 0.3]]
        categories = [1, 2]

        with pytest.raises(ValueError):
            dist = NominalDistribution(probs=probs, categories=categories)

        # Case 15: random_state is stored verbatim and exposed via get_params
        dist = NominalDistribution(probs=[[0.1, 0.9]], categories=[1, 2], random_state=42)
        assert dist.random_state == 42
        assert dist.get_params()["random_state"] == 42

    def test_nominal_methods_unsupported(self):
        """Order- and arithmetic-based methods are undefined for nominal categoricals."""
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        dist = NominalDistribution(probs=probs, categories=categories)

        with pytest.raises(NotImplementedError):
            dist.cdf([[1], [2]])
        with pytest.raises(NotImplementedError):
            dist.ppf([[0.5], [0.5]])
        with pytest.raises(NotImplementedError):
            dist.mean()
        with pytest.raises(NotImplementedError):
            dist.var()
        with pytest.raises(NotImplementedError):
            dist.energy()

    def test_pmf(self):
        """test"""
        # Case 1: x: int
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        x = [[1], [1]]

        dist = NominalDistribution(probs=probs, categories=categories)

        expected = pd.DataFrame({"variable": [0.1, 0.5]})
        pd.testing.assert_frame_equal(dist.pmf(x), expected)

        # Case 2: x: str
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = ["A", "B", "C"]
        x = [["A"], ["C"]]

        dist = NominalDistribution(probs=probs, categories=categories)

        expected = pd.DataFrame({"variable": [0.1, 0.2]})
        pd.testing.assert_frame_equal(dist.pmf(x), expected)

        # Case 3: wrong x's ndim
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        x = [1, 1]

        dist = NominalDistribution(probs=probs, categories=categories)

        with pytest.raises(ValueError):
            dist.pmf(x)

        # Case 4: wrong x's value
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        x = [["A"], ["B"]]

        dist = NominalDistribution(probs=probs, categories=categories)

        expected = pd.DataFrame({"variable": [0.0, 0.0]})
        pd.testing.assert_frame_equal(dist.pmf(x), expected)

        # Case 5: broadcasting
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        x = [[1]]

        dist = NominalDistribution(probs=probs, categories=categories)

        expected = pd.DataFrame({"variable": [0.1, 0.5]})
        pd.testing.assert_frame_equal(dist.pmf(x), expected)

    def test_log_pmf(self):
        """test"""
        # Case 1: x: int
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        x = [[1], [1]]

        dist = NominalDistribution(probs=probs, categories=categories)

        expected = pd.DataFrame({"variable": np.log([0.1, 0.5])})
        pd.testing.assert_frame_equal(dist.log_pmf(x), expected)

        # Case 2: x: str
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = ["A", "B", "C"]
        x = [["A"], ["C"]]

        dist = NominalDistribution(probs=probs, categories=categories)

        expected = pd.DataFrame({"variable": np.log([0.1, 0.2])})
        pd.testing.assert_frame_equal(dist.log_pmf(x), expected)

        # Case 3: wrong x's ndim
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        x = [1, 1]

        dist = NominalDistribution(probs=probs, categories=categories)

        with pytest.raises(ValueError):
            dist.log_pmf(x)

        # Case 4: wrong x's value
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        x = [["A"], ["B"]]

        dist = NominalDistribution(probs=probs, categories=categories)

        # unknown categories -> -inf, and no spurious divide-by-zero warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = dist.log_pmf(x)
        expected = pd.DataFrame({"variable": [-np.inf, -np.inf]})
        pd.testing.assert_frame_equal(res, expected)

        # Case 5: broadcasting
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        x = [[1]]

        dist = NominalDistribution(probs=probs, categories=categories)

        expected = pd.DataFrame({"variable": np.log([0.1, 0.5])})
        pd.testing.assert_frame_equal(dist.log_pmf(x), expected)

    def test_sample(self):
        """Sampling is reproducible across runs when ``random_state`` is set."""
        probs = [[0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
        categories = [1, 2, 3]

        # Same seed on independent instances -> identical samples (reproducible
        # across runs/processes), for both the single- and multi-sample paths.
        d1 = NominalDistribution(probs=probs, categories=categories, random_state=42)
        d2 = NominalDistribution(probs=probs, categories=categories, random_state=42)
        pd.testing.assert_frame_equal(d1.sample(), d2.sample())
        pd.testing.assert_frame_equal(d1.sample(3), d2.sample(3))

        assert not d1.sample(3).equals(d1.sample(3))

        # Different seeds produce different draws (the seed is actually used).
        d3 = NominalDistribution(probs=probs, categories=categories, random_state=123)
        assert not d1.sample(20).equals(d3.sample(20))

        # Sampling does not depend on the global NumPy RNG.
        d1 = NominalDistribution(probs=probs, categories=categories, random_state=42)
        first = d1.sample(3)
        np.random.seed(0)
        np.random.random(10)  # perturb the global RNG
        d1 = NominalDistribution(probs=probs, categories=categories, random_state=42)
        second = d1.sample(3)
        pd.testing.assert_frame_equal(first, second)

        # Single-sample structure: one row per distribution, with self's index/columns.
        single = d1.sample()
        assert single.shape == (2, 1)
        assert list(single.columns) == ["variable"]

        # Multi-sample structure and validity.
        res = d1.sample(3)
        assert res.shape == (6, 1)
        assert list(res.columns) == ["variable"]
        assert isinstance(res.index, pd.MultiIndex)
        assert res.index.names == ["sample", None]
        assert set(np.unique(res.values)).issubset(set(categories))

        # Wrong n_samples value.
        with pytest.raises(TypeError):
            d1.sample("A")

    @pytest.mark.skipif(
        not _check_soft_dependencies("matplotlib", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_plot(self):
        # Case 1: default
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        probs = [[0.2, 0.4, 0.3, 0.1], [0.4, 0.4, 0.1, 0.1]]
        categories = ["A", "B", "C", "D"]
        index = ["studentA", "studentB"]
        columns = ["grade"]

        dist = NominalDistribution(probs=probs, categories=categories, index=index, columns=columns)
        fig, axes = dist.plot(fun="pmf")
        try:
            assert isinstance(fig, plt.Figure)
            assert isinstance(axes, np.ndarray)
            assert axes.shape == (2,)

            assert axes[0].get_ylabel() == "studentA"
            assert axes[1].get_ylabel() == "studentB"

            assert axes[0].get_title() == "grade"
            assert axes[1].get_xlabel() == "state names"

            for ax in axes:
                assert ax.get_ylim() == pytest.approx((0.0, 1.0))
        finally:
            plt.close(fig)

        # Case 2: "cdf", "pdf"
        with pytest.raises(NotImplementedError):
            dist.plot(fun="pdf")

        with pytest.raises(NotImplementedError):
            dist.plot(fun="cdf")
