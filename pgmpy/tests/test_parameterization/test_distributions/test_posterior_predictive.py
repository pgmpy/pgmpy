import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import logsumexp
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy.parameterization import PosteriorPredictive, PyroAdapter, PyroSVI

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies(["pyro-ppl", "skpro"], severity="none"),
    reason="execute only if required dependency present",
)

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")
dist = _safe_import("pyro.distributions", pkg_name="pyro-ppl")


def line(parents):
    # Nothing to learn: every draw is the same normal, so the mixture is that normal.
    return dist.Normal(1.0 + 2.0 * parents["a"], 0.5)


def regression(parents):
    intercept = pyro.sample("intercept", dist.Normal(0.0, 5.0))
    coef = pyro.sample("coef", dist.Normal(0.0, 5.0))
    sigma = pyro.sample("sigma", dist.HalfNormal(2.0))
    return dist.Normal(intercept + coef * parents["a"], sigma)


@pytest.fixture(scope="module")
def train():
    X = pd.DataFrame({"a": np.linspace(-1, 1, 40)})
    noise = np.tile([-0.5, 0.5], 20)
    return X, pd.Series(1 + 2 * X["a"] + noise, name="y")


@pytest.fixture
def X():
    return pd.DataFrame({"a": [-1.0, 0.0, 2.0]}, index=["u", "v", "w"])


class TestPosteriorPredictive:
    def test_exact_normal(self, train, X):
        # Without latent sites the mixture is fn's distribution itself: N(1 + 2a, 0.5) for each row.
        dist_ = PyroAdapter(line, estimator=PyroSVI(num_steps=1)).fit(*train).predict_proba(X)
        assert isinstance(dist_, PosteriorPredictive)
        assert (dist_.index.tolist(), dist_.columns.tolist()) == (["u", "v", "w"], ["y"])
        assert dist_.get_tag("distr:measuretype") == "continuous"
        locs, x = np.array([-1.0, 1.0, 5.0]), np.array([[-1.2], [1.0], [6.0]])
        np.testing.assert_allclose(dist_.mean()["y"], locs, rtol=1e-6)
        np.testing.assert_allclose(dist_.var()["y"], [0.25] * 3, rtol=1e-6)
        np.testing.assert_allclose(dist_.log_pdf(x)["y"], stats.norm.logpdf(x[:, 0], locs, 0.5), rtol=1e-5)
        np.testing.assert_allclose(dist_.pdf(x)["y"], stats.norm.pdf(x[:, 0], locs, 0.5), rtol=1e-5)
        np.testing.assert_allclose(dist_.cdf(x)["y"], stats.norm.cdf(x[:, 0], locs, 0.5), rtol=1e-5)

    def test_mixture_over_draws(self, train, X):
        # Each row's density is the average of fn's densities over the posterior draws, its mean the average of their
        # means, and its variance the average variance plus the variance of the means.
        node = PyroAdapter(regression, estimator=PyroSVI(num_steps=300, num_samples=50, random_state=0)).fit(*train)
        draws = {name: values.numpy().astype(float) for name, values in node.posterior_samples_.items()}
        locs = draws["intercept"][:, None] + draws["coef"][:, None] * X["a"].to_numpy()
        scales = np.broadcast_to(draws["sigma"][:, None], locs.shape)
        x = np.array([0.0, 1.5, 3.0])

        dist_ = node.predict_proba(X)
        expected = logsumexp(stats.norm.logpdf(x, locs, scales), axis=0) - np.log(50)
        np.testing.assert_allclose(dist_.log_pdf(x.reshape(-1, 1))["y"], expected, rtol=1e-4)
        np.testing.assert_allclose(dist_.mean()["y"], locs.mean(axis=0), rtol=1e-4)
        np.testing.assert_allclose(dist_.var()["y"], (scales**2).mean(axis=0) + locs.var(axis=0), rtol=1e-4)
        np.testing.assert_allclose(node.predict(X)["y"], locs.mean(axis=0), rtol=1e-4)

        # The draws are fixed at fit, so the distribution is the same on every call.
        pd.testing.assert_frame_equal(node.predict_proba(X).mean(), dist_.mean())

    def test_support(self):
        # Outside its support a distribution has density and probability 0, and its cdf is 0 below the support.
        counts = PyroAdapter(lambda parents: dist.Poisson(torch.tensor(2.0)), estimator=PyroSVI(num_steps=1))
        counts.fit(None, pd.Series([0, 1, 2, 5], name="k"))
        poisson = counts.predict_proba(pd.DataFrame(index=range(3)))
        assert poisson.get_tag("distr:measuretype") == "discrete"
        np.testing.assert_allclose(
            poisson.log_pmf(np.array([[0.0], [3.0], [1.5]]))["k"], [*stats.poisson.logpmf([0, 3], 2), -np.inf]
        )
        assert (
            counts.log_likelihood(pd.DataFrame(index=["p"]), pd.Series([1.5], index=["p"], name="k"))["k"].item()
            == -np.inf
        )
        with pytest.raises(NotImplementedError, match="Poisson"):
            poisson.cdf(np.zeros((3, 1)))

        waiting = PyroAdapter(lambda parents: dist.Gamma(2.0, 1.0), estimator=PyroSVI(num_steps=1)).fit(
            None, pd.Series([1.0, 2.0], name="t")
        )
        gamma = waiting.predict_proba(pd.DataFrame(index=range(2)))
        x = np.array([[-1.0], [1.0]])
        np.testing.assert_allclose(gamma.pdf(x)["t"], [0.0, stats.gamma.pdf(1.0, a=2)], rtol=1e-5)
        np.testing.assert_allclose(gamma.cdf(x)["t"], [0.0, stats.gamma.cdf(1.0, a=2)], rtol=1e-5)

        # So does a family that wraps another, as LogNormal wraps a Normal.
        sizes = PyroAdapter(lambda parents: dist.LogNormal(0.0, 1.0), estimator=PyroSVI(num_steps=1)).fit(
            None, pd.Series([1.0, 2.0], name="s")
        )
        scores = sizes.log_likelihood(None, pd.Series([-1.0, 1.0], name="s"))["s"]
        np.testing.assert_allclose(scores, [-np.inf, stats.lognorm.logpdf(1.0, s=1.0)], rtol=1e-5)

    def test_subsetting(self, train, X):
        # Subsets keep their rows' parents, so composites such as AdditiveNoiseMechanism's MeanScale can take them.
        dist_ = PyroAdapter(line, estimator=PyroSVI(num_steps=1)).fit(*train).predict_proba(X)
        assert dist_.iloc[[1, 2]].mean()["y"].tolist() == pytest.approx([1.0, 5.0])
        assert dist_.loc[["w", "u"]].mean()["y"].tolist() == pytest.approx([5.0, -1.0])
        cell = dist_.iat[2, 0]
        assert cell.shape == () and cell.mean() == pytest.approx(5.0)
        frame = pd.DataFrame({"y": [5.0, -1.0]}, index=["w", "u"])
        np.testing.assert_allclose(dist_.pdf(frame)["y"], stats.norm.pdf([5.0, -1.0], [5.0, -1.0], 0.5), rtol=1e-5)

    def test_scalar_and_empty(self, train, X):
        # A root without X gives a scalar distribution, and no rows give empty results without evaluating fn.
        root = PyroAdapter(lambda parents: dist.Normal(3.0, 2.0), estimator=PyroSVI(num_steps=1)).fit(
            None, pd.Series([1.0, 5.0], name="r")
        )
        scalar = root.predict_proba()
        assert scalar.shape == () and scalar.mean() == pytest.approx(3.0) and scalar.var() == pytest.approx(4.0)
        assert scalar.log_pdf(3.0) == pytest.approx(stats.norm.logpdf(3.0, 3.0, 2.0))

        node = PyroAdapter(line, estimator=PyroSVI(num_steps=1)).fit(*train)
        empty = node.predict_proba(X.iloc[:0])
        assert empty.mean().shape == (0, 1) and empty.log_pdf(np.empty((0, 1))).shape == (0, 1)
        assert node.predict(X.iloc[:0]).shape == (0, 1)
        assert node.log_likelihood(X.iloc[:0], pd.Series([], dtype=float, name="y")).shape == (0, 1)

    def test_sample(self, train, X):
        # Without latent sites the values come from fn's distribution itself, N(1 + 2a, 0.5) for each row. Sampling uses
        # the distribution's own sampler, without skpro's approximations, and leaves the global random state as it was.
        dist_ = PyroAdapter(line, estimator=PyroSVI(num_steps=1)).fit(*train).predict_proba(X)
        before = torch.random.get_rng_state().clone()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            samples = dist_.sample(n_samples=4000, random_state=0)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert samples.shape == (12000, 1)
        assert samples.index.get_level_values(1)[:3].tolist() == ["u", "v", "w"]
        by_row = samples["y"].groupby(level=1)
        np.testing.assert_allclose(by_row.mean()[["u", "v", "w"]], [-1.0, 1.0, 5.0], atol=0.03)
        np.testing.assert_allclose(by_row.std()[["u", "v", "w"]], [0.5] * 3, atol=0.03)
        pd.testing.assert_frame_equal(dist_.sample(3, random_state=1), dist_.sample(3, random_state=1))
        assert dist_.sample(random_state=0).index.tolist() == ["u", "v", "w"]
        assert isinstance(dist_.iat[0, 0].sample(random_state=0), float)

    def test_sample_mixture(self):
        # From two observations the posterior of mu is wide, so the mixture's variance, 1 plus the posterior variance of
        # mu, is clearly larger than that of any one draw. Only picking a draw for each value reproduces it.
        root = PyroAdapter(
            lambda parents: dist.Normal(pyro.sample("mu", dist.Normal(0.0, 10.0)), 1.0),
            estimator=PyroSVI(num_samples=200, random_state=0),
        ).fit(None, pd.Series([0.0, 1.0], name="r"))
        mu = root.posterior_samples_["mu"].numpy().astype(float)
        values = root.sample(n_samples=20000, random_state=0)["r"]
        assert mu.var() > 0.3
        assert abs(values.mean() - mu.mean()) < 0.05
        assert abs(values.var() - (1 + mu.var())) < 0.08

    def test_sample_discrete(self):
        # A Poisson has no cdf in torch, so only its own sampler works; its values are counts.
        counts = PyroAdapter(lambda parents: dist.Poisson(torch.tensor(2.0)), estimator=PyroSVI(num_steps=1))
        values = counts.fit(None, pd.Series([0, 1, 2, 5], name="k")).sample(n_samples=20000, random_state=0)["k"]
        assert (values == np.round(values)).all()
        assert abs(values.mean() - 2) < 0.05 and abs(values.var() - 2) < 0.1
