import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps, _check_soft_dependencies, _safe_import

from pgmpy.parameterization import DistributionAdapter, NominalDistribution

Delta = _safe_import("skpro.distributions.Delta")
Empirical = _safe_import("skpro.distributions.Empirical")
Hurdle = _safe_import("skpro.distributions.Hurdle")
Histogram = _safe_import("skpro.distributions.Histogram")
IID = _safe_import("skpro.distributions.IID")
LogNormal = _safe_import("skpro.distributions.LogNormal")
Normal = _safe_import("skpro.distributions.Normal")
Pareto = _safe_import("skpro.distributions.Pareto")
Poisson = _safe_import("skpro.distributions.Poisson")

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)

ROWS = pd.DataFrame(index=["p", "q", "r"])


class TestDistributionAdapter:
    def test_init_and_from_values(self):
        normal = Normal(mu=1.5, sigma=2.0)

        # The target is discrete only for a NominalDistribution, whatever skpro's measure type, e.g. of a Delta. The
        # distribution is given, so the adapter needs no data.
        assert DistributionAdapter(normal).get_tag("variable_type") == ["continuous"]
        assert DistributionAdapter(Delta(c=2.5)).get_tag("variable_type") == ["continuous"]
        assert DistributionAdapter(NominalDistribution([0.3, 0.7], ["a", "b"])).get_tag("variable_type") == ["discrete"]
        assert DistributionAdapter.get_class_tag("requires_data") is False

        # A NominalDistribution is a fixed table, and a Normal a linear Gaussian with zero coefficients, so both support
        # exact inference; other distributions don't.
        table = DistributionAdapter(NominalDistribution([0.3, 0.7], ["a", "b"]))
        assert table.get_tag("capability:factor") and table.get_tag("capability:exact_inference")
        gaussian = DistributionAdapter(IID(normal, index=pd.Index([0]), columns=pd.Index(["X"])))
        assert gaussian.get_tag("capability:exact_inference") and not gaussian.get_tag("capability:factor")
        assert all(gaussian.get_tag(f"assumption:{name}") for name in ("linearity", "additive_noise", "gaussian_noise"))
        assert not DistributionAdapter(Delta(c=2.5)).get_tag("capability:exact_inference")

        # from_values creates a fitted adapter without data, as do() needs. fit only records the target and parents.
        do = DistributionAdapter.from_values("X", Delta(c=2.5), ["b", "a"])
        assert do.is_fitted and (do.variable_, do.evidence_) == ("X", ["a", "b"])
        assert (do.sample(pd.DataFrame({"a": [0, 1], "b": [1, 2]}), random_state=0)["X"] == 2.5).all()
        fitted = DistributionAdapter(normal).fit(None, pd.Series([0.3, -1.2], name="X"))
        assert fitted == DistributionAdapter.from_values("X", Normal(mu=1.5, sigma=2.0))
        histogram = Histogram(bins=[0.0, 1.0, 2.0], bin_mass=[0.5, 0.5])  # skbase can't clone it
        assert DistributionAdapter(histogram).fit(None, pd.Series([0.3, 1.2], name="X")).predict_proba().mean() == 1.0

        # An IID is unwrapped to its scalar distribution, and anything else must be scalar.
        iid = IID(normal, index=pd.Index([0]), columns=pd.Index(["X"]))
        assert DistributionAdapter.from_values("X", iid) == DistributionAdapter.from_values("X", normal)
        with pytest.raises(TypeError, match="skpro distribution"):
            DistributionAdapter("not a distribution")
        with pytest.raises(ValueError, match="scalar"):
            DistributionAdapter(Normal(mu=[[0.0, 1.0]], sigma=1.0))

        # The tags and the check follow the distribution after set_params, also a nested one.
        nominal = NominalDistribution([0.3, 0.7], ["a", "b"])
        assert DistributionAdapter(iid).set_params(distribution__distribution=nominal).get_tag("variable_type") == [
            "discrete"
        ]
        with pytest.raises(ValueError, match="scalar"):
            DistributionAdapter(Normal(mu=0.0, sigma=1.0)).set_params(distribution__mu=[[0.0, 1.0]])

    def test_predict_proba(self):
        # Each row of X gets the distribution, labelled with X's index and the target. A distribution whose parameters
        # are numbers is rebuilt for the rows, a NominalDistribution repeats its probabilities, and any other is
        # repeated by IID.
        for distribution, expected_type in (
            (Normal(mu=1.5, sigma=2.0), "Normal"),
            (Pareto(scale=1, alpha=3), "Pareto"),
            (Empirical(spl=pd.Series([1.0, 2.0, 3.0, 10.0])), "IID"),
            (Hurdle(p=0.3, distribution=Poisson(mu=2.0)), "IID"),
        ):
            dist = DistributionAdapter.from_values("y", distribution).predict_proba(ROWS)
            assert (type(dist).__name__, dist.index.tolist(), dist.columns.tolist()) == (
                expected_type,
                list("pqr"),
                ["y"],
            )
            for method in {"cdf", "pdf"} & set(distribution.get_tag("capabilities:exact")):
                values = getattr(dist, method)(np.full((3, 1), 2.0)).to_numpy().ravel()
                np.testing.assert_allclose(values, getattr(distribution, method)(2.0))
        nominal = DistributionAdapter.from_values("y", NominalDistribution([0.3, 0.7], ["a", "b"])).predict_proba(ROWS)
        assert (nominal.index.tolist(), nominal.columns.tolist(), list(nominal.categories)) == (
            list("pqr"),
            ["y"],
            ["a", "b"],
        )
        np.testing.assert_allclose(np.asarray(nominal.probs), [[0.3, 0.7]] * 3)

        # predict gives the mean of a continuous distribution and the most probable state of a nominal one.
        normal = DistributionAdapter.from_values("y", Normal(mu=1.5, sigma=2.0))
        pd.testing.assert_frame_equal(normal.predict(ROWS), pd.DataFrame({"y": [1.5] * 3}, index=ROWS.index))
        nominal = DistributionAdapter.from_values("y", NominalDistribution([0.3, 0.7], ["a", "b"]))
        pd.testing.assert_frame_equal(nominal.predict(ROWS), pd.DataFrame({"y": ["b"] * 3}, index=ROWS.index))

        # log_likelihood uses the log-density of a continuous measure and the log-mass of a discrete one, such as a
        # nominal distribution or a Delta.
        y = pd.Series([0.0, 1.5], name="y")
        np.testing.assert_allclose(normal.log_likelihood(None, y)["y"], [-1.8933, -1.6121], atol=1e-4)
        labels = pd.Series(["a", "b"], name="y")
        np.testing.assert_allclose(nominal.log_likelihood(None, labels)["y"], np.log([0.3, 0.7]))
        delta = DistributionAdapter.from_values("y", Delta(c=1.5))
        np.testing.assert_allclose(delta.log_likelihood(None, y)["y"], [-np.inf, 0.0])

        # The distribution is copied in and out, so changing the original or a returned one leaves the adapter as it
        # was.
        normal = Normal(mu=1.5, sigma=2.0)
        adapter = DistributionAdapter.from_values("y", normal)
        normal.set_params(mu=100.0)
        adapter.predict_proba().set_params(mu=100.0)
        assert adapter.predict_proba().mean() == 1.5
        np.testing.assert_allclose(adapter.predict_proba(ROWS).mean()["y"], 1.5)

    def test_every_skpro_distribution(self):
        # Every scalar test case of every skpro distribution is given to each row with the same cdf and pmf. The
        # distributions below have only 2-D test cases, so they give their first entry, by skpro's iat, which fails for
        # some other distributions, e.g. QPD_Empirical, and HistogramQPD in skpro 2.16. skpro's IID, which repeats
        # distributions that can't be rebuilt, e.g. ZeroInflated, has no pmf though its tags say so.
        from skpro.registry import all_objects

        only_2d = {"Histogram", "NegativeBinomial", "Poisson", "QPD_B", "QPD_S", "QPD_U"}
        with np.errstate(divide="ignore", invalid="ignore"):
            for _, cls in all_objects(object_types="distribution"):
                if not _check_estimator_deps(cls, severity="none"):
                    continue
                params = cls.get_test_params()
                for case in params if isinstance(params, list) else [params]:
                    distribution = cls(**case)
                    if distribution.ndim != 0:
                        if cls.__name__ not in only_2d:
                            continue
                        distribution = distribution.iat[0, 0]
                    dist = DistributionAdapter.from_values("y", distribution).predict_proba(ROWS)
                    assert (dist.shape, dist.index.tolist(), dist.columns.tolist()) == ((3, 1), list("pqr"), ["y"]), cls
                    point = distribution.sample()
                    exact = set(distribution.get_tag("capabilities:exact"))
                    for method in {"cdf", "pmf"} & (exact - {"pmf"} if type(dist).__name__ == "IID" else exact):
                        expected = getattr(distribution, method)(point)
                        values = getattr(dist, method)(np.full((3, 1), point)).to_numpy().ravel()
                        np.testing.assert_allclose(values, expected, err_msg=f"{cls.__name__}.{method}")

    def test_sample_and_equality(self):
        normal = DistributionAdapter.from_values("y", Normal(mu=1.5, sigma=2.0))
        samples = normal.sample(n_samples=2000, random_state=0)["y"]
        assert samples.mean() == pytest.approx(1.5, abs=0.2) and samples.std() == pytest.approx(2.0, rel=0.1)
        pd.testing.assert_frame_equal(normal.sample(ROWS, n_samples=2, random_state=0), normal.sample(ROWS, 2, 0))
        assert normal.sample(ROWS, n_samples=2, random_state=0).shape == (6, 1)
        assert not normal.sample(ROWS, random_state=0).equals(normal.sample(ROWS, random_state=1))
        nominal = DistributionAdapter.from_values("y", NominalDistribution([0.3, 0.7], ["a", "b"]))
        assert nominal.sample(n_samples=20, random_state=0)["y"].isin(["a", "b"]).all()

        # Fitted adapters compare by their distribution's type and parameters.
        same = DistributionAdapter.from_values("y", Normal(mu=1.5, sigma=2.0))
        assert normal == same and len({normal, same}) == 1
        assert normal != DistributionAdapter.from_values("y", LogNormal(mu=1.5, sigma=2.0))
        assert normal != DistributionAdapter.from_values("y", Normal(mu=0.0, sigma=2.0))
