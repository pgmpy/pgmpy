import copy
import warnings

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.parameterization.distributions.nominal import NominalDistribution

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)


class TestNominalDistribution:
    """Tests for Nominal distributions."""

    def test_default(self):
        probs = [[0.1, 0.9], [0.7, 0.3]]
        categories = ["A", "B"]
        dist = NominalDistribution(probs, categories)

        # The tags pgmpy relies on: log_likelihood scores a discrete measure with log_pmf, and BaseParameter's sampling
        # doesn't repeat rows of a distribution whose init isn't broadcast.
        assert dist.name == "NominalDistribution"
        assert dist.get_class_tag("distr:measuretype") == "discrete"
        assert dist.get_class_tag("capabilities:exact") == ["pmf", "log_pmf"]
        assert dist.get_class_tag("broadcast_init") == "off"

    def test_public_import(self):
        """The class is importable from the package, not just the module."""
        from pgmpy.parameterization.distributions import NominalDistribution as PublicCat

        assert PublicCat is NominalDistribution

    @pytest.mark.parametrize("scalar", [False, True])
    def test_interface_compatibility(self, scalar):
        """ensure interface compatibility by skpro.utils.estimator_checks.check_estimator"""
        from skpro.utils.estimator_checks import check_estimator

        probs = [[0.1, 0.9], [0.7, 0.3]]
        probs = probs[0] if scalar else probs
        dist = NominalDistribution(probs=probs, categories=[1, 2])
        check_estimator(dist, raise_exceptions=True, verbose=False)

        probs = [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]]
        probs = probs[0] if scalar else probs
        dist = NominalDistribution(probs=probs, categories=[1, 2, 3])
        check_estimator(dist, raise_exceptions=True, verbose=False)

        # Nominal categories with string labels must also pass the interface checks.
        probs = [[0.1, 0.9], [0.7, 0.3]]
        probs = probs[0] if scalar else probs
        dist = NominalDistribution(probs=probs, categories=["A", "B"])
        check_estimator(dist, raise_exceptions=True, verbose=False)

    def test_init(self):
        """test"""
        # Case 0: any distinct hashable labels are valid categories, including mixed types, tuples, pandas containers
        # and numpy scalars.
        for categories in (
            [1, "B", ("c", 3)],
            pd.Index([1, 2, 3]),
            pd.Series(["A", "B", "C"], index=[5, 6, 7]),
            pd.Categorical(["A", "B", "C"]),
            [np.int64(1), 2, 3],
        ):
            dist = NominalDistribution(probs=[0.2, 0.3, 0.5], categories=categories)
            assert dist.pmf(list(categories)[1]) == pytest.approx(0.3)

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

        with pytest.raises(ValueError, match="must sum to 1"):
            NominalDistribution(probs=probs, categories=categories)

        # Case 8: wrong categories (non-unique)
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 1, 2]

        with pytest.raises(ValueError, match="Categories must contain unique values"):
            NominalDistribution(probs=probs, categories=categories)

        # Case 9: wrong index
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        index = ["A", "B", "C"]

        with pytest.raises(ValueError, match="length of index must match"):
            NominalDistribution(probs=probs, categories=categories, index=index)

        # Case 10: wrong columns
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        columns = ["A", "B", "C"]

        with pytest.raises(ValueError, match="exactly one column name"):
            NominalDistribution(probs=probs, categories=categories, columns=columns)

        # Case 11: categories given as a single string
        with pytest.raises(TypeError, match="single string"):
            NominalDistribution(probs=[0.5, 0.5], categories="AB")

        # Case 12: missing values in categories
        with pytest.raises(ValueError, match="missing values"):
            NominalDistribution(probs=[0.5, 0.5], categories=["A", None])

        # Case 13: wrong shape(probs, categories)
        probs = [[0.1, 0.9], [0.5, 0.5]]
        categories = [1, 2, 3]

        with pytest.raises(ValueError, match="shape of categories and probs : 3, 2"):
            NominalDistribution(probs=probs, categories=categories)

        # Case 14: wrong probs(negative)
        probs = [[0.1, 0.2], [-0.5, 0.3]]
        categories = [1, 2]

        with pytest.raises(ValueError, match="non-negative"):
            NominalDistribution(probs=probs, categories=categories)

        # Case 15: row sums within TabularCPD's tolerance (0.01) are accepted and rescaled to 1.
        dist = NominalDistribution(probs=np.float32([0.1, 0.2, 0.7]), categories=["A", "B", "C"])
        assert dist.pmf("C") == pytest.approx(0.7)
        dist = NominalDistribution(probs=[0.3333, 0.3333, 0.3333], categories=["A", "B", "C"])
        assert dist.pmf("A") == pytest.approx(1 / 3)
        with pytest.raises(ValueError, match="must sum to 1"):
            NominalDistribution(probs=[0.49, 0.49], categories=["A", "B"])

        # random_state is an argument of sample(), not of the constructor.
        assert "random_state" not in dist.get_params()

        scalar = NominalDistribution(probs=[0.1, 0.9], categories=[1, 2])
        assert scalar.shape == ()
        assert scalar.index is None and scalar.columns is None
        assert scalar.get_params()["probs"] == [0.1, 0.9]

        batch = NominalDistribution(probs=[[0.1, 0.9]], categories=[1, 2])
        assert batch.shape == (1, 1)
        labeled = NominalDistribution(probs=[0.1, 0.9], categories=[1, 2], index=["row"], columns=["state"])
        pd.testing.assert_frame_equal(labeled.pmf(1), pd.DataFrame({"state": [0.1]}, index=["row"]))

    def test_nominal_methods_unsupported(self):
        """Order-, arithmetic- and density-based methods are undefined for nominal categoricals."""
        probs = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        categories = [1, 2, 3]
        dist = NominalDistribution(probs=probs, categories=categories)
        scalar = NominalDistribution(probs=[0.2, 0.8], categories=["A", "B"])

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
        with pytest.raises(NotImplementedError):
            dist.pdfnorm()
        with pytest.raises(NotImplementedError):
            scalar.surv("A")
        with pytest.raises(NotImplementedError):
            scalar.haz("A")

        # probs has one column per category, so skpro's DataFrame view of the parameters is not available.
        with pytest.raises(RuntimeError):
            dist.to_df()
        with pytest.raises(RuntimeError):
            dist.get_params_df()

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

        for scalar in (dist.iat[1, 0], dist.at[1, "variable"]):
            assert scalar.shape == ()
            assert np.isscalar(scalar.pmf(1))
            assert scalar.pmf(1) == 0.5
            assert scalar.pmf("unknown") == 0.0
        batch = dist.iloc[[1], [0]]
        assert batch.shape == (1, 1)
        pd.testing.assert_frame_equal(batch.pmf(1), expected.iloc[[1]])

        # Case 6: out-of-range column positions
        with pytest.raises(IndexError):
            dist.iat[0, 5]
        with pytest.raises(IndexError):
            dist.iloc[0, 5]

        # Case 7: missing values in the query
        with pytest.raises(ValueError, match="missing values"):
            dist.pmf(pd.DataFrame({"variable": [1, None]}).convert_dtypes())
        with pytest.raises(ValueError, match="missing values"):
            scalar.pmf(pd.NA)

        # Case 8: tuple labels round-trip through sample() and pmf()
        dist = NominalDistribution(probs=[[0.5, 0.5], [0.5, 0.5]], categories=[("a", 1), ("b", 2)])
        pd.testing.assert_frame_equal(dist.pmf(dist.sample()), pd.DataFrame({"variable": [0.5, 0.5]}))

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

        scalar = NominalDistribution(probs=[0.2, 0.8, 0.0], categories=["A", "B", "C"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isscalar(scalar.log_pmf("A"))
            assert scalar.log_pmf("A") == pytest.approx(np.log(0.2))
            assert scalar.log_pmf("C") == -np.inf
            assert scalar.log_pmf("unknown") == -np.inf

    @pytest.mark.parametrize(
        "index",
        [
            pd.Index(["studentB", "studentA"], name="student"),
            pd.Index(["studentB", "studentA"], name="sample"),
            pd.MultiIndex.from_tuples([("yes", "good"), ("no", "poor")], names=["Studied", "Sleep"]),
        ],
        ids=["named-index", "sample-named-index", "multi-index"],
    )
    def test_sample(self, index):
        """Sampling is reproducible when ``random_state`` is passed to ``sample``."""
        probs = [[0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
        categories = [1, 2, 3]
        d1 = NominalDistribution(probs=probs, categories=categories)
        d2 = NominalDistribution(probs=probs, categories=categories)

        # The same int seed gives identical draws on every call and across instances, for both the single- and
        # multi-sample paths.
        pd.testing.assert_frame_equal(d1.sample(random_state=42), d2.sample(random_state=42))
        pd.testing.assert_frame_equal(d1.sample(3, random_state=42), d1.sample(3, random_state=42))
        pd.testing.assert_frame_equal(d1.sample(3, random_state=42), d2.sample(3, random_state=42))

        # A Generator is used as-is and advances across calls.
        rng = np.random.default_rng(42)
        first = d1.sample(20, random_state=rng)
        assert not first.equals(d1.sample(20, random_state=rng))
        pd.testing.assert_frame_equal(first, d1.sample(20, random_state=np.random.default_rng(42)))

        # Different seeds produce different draws (the seed is actually used).
        assert not d1.sample(20, random_state=42).equals(d1.sample(20, random_state=123))

        # Without a seed, every call draws fresh samples, also on copies.
        assert not copy.deepcopy(d1).sample(20).equals(d1.sample(20))

        # Sampling does not depend on the global NumPy RNG.
        first = d1.sample(3, random_state=42)
        np.random.seed(0)
        np.random.random(10)  # perturb the global RNG
        pd.testing.assert_frame_equal(first, d1.sample(3, random_state=42))

        # Single-sample structure: one row per distribution, with self's index/columns.
        single = d1.sample()
        assert single.shape == (2, 1)
        assert list(single.columns) == ["variable"]

        # Multi-sample structure and validity.
        res = d1.sample(3)
        assert res.shape == (6, 1)
        assert list(res.columns) == ["variable"]
        assert res["variable"].dtype.kind == "i"
        assert isinstance(res.index, pd.MultiIndex)
        assert res.index.names == [None, None]
        assert set(np.unique(res.values)).issubset(set(categories))

        # Draw frequencies follow each row's probabilities.
        freq = d1.sample(20000, random_state=0)["variable"].groupby(level=1).value_counts(normalize=True)
        np.testing.assert_allclose(freq.unstack(fill_value=0).to_numpy(), probs, atol=0.02)

        # Wrong n_samples value.
        with pytest.raises(TypeError):
            d1.sample("A")

        scalar = NominalDistribution(probs=[0.0, 1.0], categories=["A", "B"])
        assert scalar.sample() == "B"
        pd.testing.assert_frame_equal(scalar.sample(3), pd.DataFrame(["B", "B", "B"]))

        dist = NominalDistribution(
            probs=[[0.0, 1.0], [1.0, 0.0]], categories=["fail", "pass"], index=index, columns=["Exam"]
        )
        expected = pd.DataFrame({"Exam": ["pass", "fail"]}, index=index)
        pd.testing.assert_frame_equal(dist.sample(), expected)

        # The draw level is unnamed, so it cannot clash with the names of the index levels.
        samples = dist.sample(2)
        assert samples.index.names == [None, *index.names]
        for sample in range(2):
            pd.testing.assert_frame_equal(samples.xs(sample, level=0), expected)
        pd.testing.assert_frame_equal(dist.sample(0), samples.iloc[:0])
