import numpy as np
import pandas as pd
import pytest
from scipy import stats
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.linear_model import LinearRegression

from pgmpy.parameterization import AdditiveNoiseMechanism, PyroAdapter, SklearnAdapter

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies(["pyro-ppl", "skpro"], severity="none"),
    reason="execute only if required dependency present",
)

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")
dist = _safe_import("pyro.distributions", pkg_name="pyro-ppl")


def sales(parents):
    # The parent is centered and scaled, and the priors are on the data's scale, so that SVI converges quickly.
    intercept = pyro.sample("intercept", dist.Normal(50.0, 20.0))
    coef = pyro.sample("coef", dist.Normal(0.0, 20.0))
    sigma = pyro.sample("sigma", dist.HalfNormal(10.0))
    return dist.Normal(intercept + coef * (parents["temp"] - 20.0) / 5.0, sigma)


def location(parents):
    # Point estimates only, which SVI fits by maximum likelihood.
    mu = pyro.param("mu", torch.tensor(0.0))
    sigma = pyro.param("sigma", torch.tensor(1.0), constraint=dist.constraints.positive)
    return dist.Normal(mu, sigma)


@pytest.fixture(scope="module")
def sales_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"temp": rng.normal(20, 5, size=300)})
    y = pd.Series(10 + 2 * X["temp"] + rng.normal(scale=3, size=300), name="sales")
    return X, y


@pytest.fixture(scope="module")
def sales_cpd(sales_data):
    return PyroAdapter(sales, num_samples=500, random_state=0).fit(*sales_data)


class TestFit:
    def test_tags_and_construction(self):
        assert PyroAdapter.get_class_tag("variable_type") == ["continuous"]
        assert PyroAdapter.get_class_tag("parent_data_types") == ["continuous"]
        assert PyroAdapter.get_class_tag("supports_weighted_data") is True
        assert PyroAdapter.get_class_tag("python_dependencies") == ["pyro-ppl", "skpro"]
        with pytest.raises(TypeError, match="callable"):
            PyroAdapter("not a function")
        with pytest.raises(ValueError, match="'svi' or 'mcmc'"):
            PyroAdapter(sales, inference="vi")
        for name in ("num_samples", "num_steps"):
            with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
                PyroAdapter(sales, **{name: 0})

    def test_svi(self, sales_cpd, sales_data):
        # The posterior means match least squares on the scaled parent.
        X, y = sales_data
        scaled = (X["temp"] - 20) / 5
        slope, intercept = np.polyfit(scaled, y, 1)
        residual_sd = np.std(y - intercept - slope * scaled)
        draws = sales_cpd.posterior_samples_
        assert set(draws) == {"intercept", "coef", "sigma"}
        assert all(values.shape == (500,) for values in draws.values())
        assert abs(draws["intercept"].mean().item() - intercept) < 0.2
        assert abs(draws["coef"].mean().item() - slope) < 0.2
        assert abs(draws["sigma"].mean().item() - residual_sd) < 0.2
        assert sales_cpd.losses_.shape == (1000,)
        assert sales_cpd.params_ == {}

    def test_mcmc_matches_the_conjugate_posterior(self):
        # With sigma known and normal priors, the posterior of the intercept and slope is normal, in closed form.
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"a": rng.normal(size=100)})
        y = pd.Series(1 + 2 * X["a"] + rng.normal(scale=0.5, size=100), name="y")

        def known_sigma(parents):
            intercept = pyro.sample("intercept", dist.Normal(0.0, 10.0))
            coef = pyro.sample("coef", dist.Normal(0.0, 10.0))
            return dist.Normal(intercept + coef * parents["a"], 0.5)

        design = np.column_stack([np.ones(100), X["a"]])
        precision = design.T @ design / 0.25 + np.eye(2) / 100
        mean = np.linalg.solve(precision, design.T @ y / 0.25)
        sd = np.sqrt(np.diag(np.linalg.inv(precision)))

        draws = PyroAdapter(known_sigma, inference="mcmc", num_samples=300, random_state=0).fit(X, y).posterior_samples_
        np.testing.assert_allclose([draws["intercept"].mean().item(), draws["coef"].mean().item()], mean, atol=0.02)
        np.testing.assert_allclose([draws["intercept"].std().item(), draws["coef"].std().item()], sd, rtol=0.2)

    def test_isolation(self):
        # Each node keeps its parameters in a store of its own: two nodes with the same parameter names don't overwrite
        # each other, and the user's global store keeps its values and gets no new ones. The optimizer passed in isn't
        # used itself, so it holds no state afterwards.
        store = pyro.get_param_store()
        with store.scope():
            pyro.param("user_value", torch.tensor(5.0))
            optim = pyro.optim.Adam({"lr": 0.05})
            spread = np.tile([-1.0, 1.0], 10)
            first = PyroAdapter(location, num_steps=500, optim=optim).fit(None, pd.Series(3.0 + spread, name="a"))
            second = PyroAdapter(location, num_steps=500, optim=optim).fit(None, pd.Series(-2.0 + spread, name="b"))
            assert abs(first.params_["mu"].item() - 3.0) < 0.05
            assert abs(second.params_["mu"].item() + 2.0) < 0.05
            assert list(store.keys()) == ["user_value"] and store["user_value"].item() == 5.0
            assert not optim.optim_objs
            # Predicting uses the node's own store too, after the other node's fit.
            assert first.predict(pd.DataFrame(index=range(2)))["a"].tolist() == pytest.approx([3.0, 3.0], abs=0.05)
            assert list(store.keys()) == ["user_value"]

    def test_predictions(self, sales_cpd, sales_data):
        # The posterior predictive is close to the least-squares normal: the fitted line, with the residuals' variance
        # plus a little for the uncertainty about the parameters.
        X, y = sales_data
        scaled = (X["temp"] - 20) / 5
        slope, intercept = np.polyfit(scaled, y, 1)
        residual_sd = np.std(y - intercept - slope * scaled)
        new = pd.DataFrame({"temp": [15.0, 25.0]}, index=["cold", "warm"])
        means = np.array([intercept - slope, intercept + slope])

        predictions = sales_cpd.predict(new)
        assert predictions.index.tolist() == ["cold", "warm"]
        np.testing.assert_allclose(predictions["sales"], means, atol=0.3)
        variances = sales_cpd.predict_proba(new).var()["sales"]
        assert ((variances > residual_sd**2 - 0.6) & (variances < residual_sd**2 + 0.8)).all()
        observed = pd.Series([41.0, 58.0], index=["cold", "warm"], name="sales")
        scores = sales_cpd.log_likelihood(new, observed)["sales"]
        np.testing.assert_allclose(scores, stats.norm.logpdf(observed, means, residual_sd), atol=0.05)

    def test_random_state(self, sales_data):
        # The same seed gives the same draws, and fitting leaves the global torch random state as it was.
        X, y = sales_data
        before = torch.random.get_rng_state().clone()
        fits = [PyroAdapter(sales, num_steps=50, num_samples=20, random_state=7).fit(X, y) for _ in range(2)]
        assert torch.equal(torch.random.get_rng_state(), before)
        for name in ("intercept", "coef", "sigma"):
            assert torch.equal(fits[0].posterior_samples_[name], fits[1].posterior_samples_[name])
        other = PyroAdapter(sales, num_steps=50, num_samples=20, random_state=8).fit(X, y)
        assert not torch.equal(other.posterior_samples_["coef"], fits[0].posterior_samples_["coef"])

    def test_weights(self):
        # A weight multiplies its row's log-likelihood, so weighted rows give the fit of duplicated rows: the weighted
        # mean and standard deviation, by maximum likelihood.
        settings = {"num_steps": 2000, "optim": pyro.optim.Adam({"lr": 0.05})}
        weighted = PyroAdapter(location, **settings).fit(
            None, pd.Series([1.0, 2.0, 4.0, 7.0], name="y"), sample_weight=[1.0, 2.0, 0.0, 3.0]
        )
        duplicated = PyroAdapter(location, **settings).fit(None, pd.Series([1.0, 2.0, 2.0, 7.0, 7.0, 7.0], name="y"))
        for name in ("mu", "sigma"):
            assert abs(weighted.params_[name].item() - duplicated.params_[name].item()) < 1e-4
        assert abs(weighted.params_["mu"].item() - 26 / 6) < 0.05

    def test_sample(self, sales_cpd):
        # Samples follow the parameterization's API: X's index under a level numbering the draws, the same values for
        # the same seed, and the global random state as it was.
        new = pd.DataFrame({"temp": [15.0, 25.0]}, index=["cold", "warm"])
        before = torch.random.get_rng_state().clone()
        samples = sales_cpd.sample(new, n_samples=3, random_state=1)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert samples.shape == (6, 1) and samples.index.get_level_values(1).tolist() == ["cold", "warm"] * 3
        pd.testing.assert_frame_equal(samples, sales_cpd.sample(new, n_samples=3, random_state=1))
        assert sales_cpd.sample(new.iloc[:0]).shape == (0, 1)

    def test_noise_of_an_additive_noise_mechanism(self, sales_data):
        # A Pyro root can be the noise of an additive noise mechanism: fitted on the residuals, and shifted by f(x) in
        # skpro's MeanScale, which subsets it.
        X, y = sales_data

        def laplace(parents):
            loc = pyro.sample("loc", dist.Normal(0.0, 1.0))
            scale = pyro.sample("scale", dist.HalfNormal(5.0))
            return dist.Laplace(loc, scale)

        noise = PyroAdapter(laplace, num_steps=300, num_samples=100, random_state=0)
        mechanism = AdditiveNoiseMechanism(SklearnAdapter(LinearRegression()), noise=noise).fit(X, y)
        new = X.iloc[:3]
        expected = mechanism.function_.predict(new)["sales"] + mechanism.noise_.predict_proba().mean()
        np.testing.assert_allclose(mechanism.predict_proba(new).mean()["sales"], expected, rtol=1e-5)
        assert mechanism.predict_proba(new).iat[0, 0].mean() == pytest.approx(expected.iloc[0], rel=1e-5)
        assert mechanism.sample(new, n_samples=2, random_state=0).shape == (6, 1)

    def test_fit_errors(self, sales_data):
        X, y = sales_data
        cases = [
            (lambda parents: parents["temp"], TypeError, "must return a Pyro distribution"),
            (lambda parents: dist.MultivariateNormal(torch.zeros(2), torch.eye(2)), ValueError, "event_shape"),
            (lambda parents: dist.Normal(torch.zeros(3), 1.0), ValueError, "batch shape"),
            (lambda parents: dist.Normal(pyro.sample("sales", dist.Normal(0.0, 1.0)), 1.0), ValueError, "'sales'"),
            (lambda parents: dist.Poisson(torch.tensor(1.0)), ValueError, "outside the support"),
        ]
        for fn, error, match in cases:
            with pytest.raises(error, match=match):
                PyroAdapter(fn, num_steps=1).fit(X, y)

        # NUTS needs latent sites, and can't fit pyro.param values.
        with pytest.raises(ValueError, match="pyro.param"):
            PyroAdapter(location, inference="mcmc").fit(None, y)
        with pytest.raises(ValueError, match="latent"):
            PyroAdapter(lambda parents: dist.Normal(0.0, 1.0), inference="mcmc").fit(None, y)

        # A location far beyond float32's range of squares makes the first loss infinite.
        with pytest.raises(ValueError, match="isn't finite"):
            PyroAdapter(lambda parents: dist.Normal(pyro.param("mu", torch.tensor(1e30)), 1.0), num_steps=1).fit(
                None, y
            )
