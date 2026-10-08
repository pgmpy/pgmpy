import numpy as np
import pandas as pd
import pytest
from scipy import stats
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from sklearn.linear_model import LinearRegression, LogisticRegression

from pgmpy.parameterization import AdditiveNoiseMechanism, PyroAdapter, PyroNUTS, PyroSVI, SklearnAdapter, TabularCPD

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


def wet(parents):
    # Logistic regression of the ground's state on the scaled humidity.
    intercept = pyro.sample("intercept", dist.Normal(0.0, 2.0))
    coef = pyro.sample("coef", dist.Normal(0.0, 2.0))
    return dist.Bernoulli(logits=intercept + coef * (parents["humidity"] - 50.0) / 20.0)


def weather(parents):
    # One logit per season and weather state. The season comes as integer codes over its sorted states, and einsum
    # with "..." also covers the draws that W gets in front of its event dimensions when predicting.
    W = pyro.sample("W", dist.Normal(torch.zeros(4, 3), 2.0).to_event(2))
    onehot = torch.nn.functional.one_hot(parents["season"], 4).to(W.dtype)
    return dist.Categorical(logits=torch.einsum("nk,...kt->...nt", onehot, W))


@pytest.fixture(scope="module")
def sales_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"temp": rng.normal(20, 5, size=300)})
    y = pd.Series(10 + 2 * X["temp"] + rng.normal(scale=3, size=300), name="sales")
    return X, y


@pytest.fixture(scope="module")
def sales_cpd(sales_data):
    return PyroAdapter(sales, estimator=PyroSVI(num_samples=500, random_state=0)).fit(*sales_data)


class TestFit:
    def test_tags_and_construction(self):
        # The class models both types, and each instance one. The arguments say what the model is, then how to fit it.
        assert PyroAdapter.get_class_tag("variable_type") == ["discrete", "continuous"]
        assert PyroAdapter.get_class_tag("parent_data_types") == ["discrete", "continuous", "mixed"]
        assert PyroAdapter(sales, "discrete").get_tag("variable_type") == ["discrete"]
        assert PyroAdapter(sales).get_tag("variable_type") == ["continuous"]
        assert PyroAdapter.get_class_tag("supports_weighted_data") is True
        assert PyroAdapter.get_class_tag("python_dependencies") == ["pyro-ppl", "skpro"]
        with pytest.raises(TypeError, match="callable"):
            PyroAdapter("not a function")

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
        assert sales_cpd.diagnostics_["losses"].shape == (1000,)
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

        nuts = PyroNUTS(num_samples=300, random_state=0)
        draws = PyroAdapter(known_sigma, estimator=nuts).fit(X, y).posterior_samples_
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
            estimator = PyroSVI(num_steps=500, optim=optim)
            first = PyroAdapter(location, estimator=estimator).fit(None, pd.Series(3.0 + spread, name="a"))
            second = PyroAdapter(location, estimator=estimator).fit(None, pd.Series(-2.0 + spread, name="b"))
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
        fits = [
            PyroAdapter(sales, estimator=PyroSVI(num_steps=50, num_samples=20, random_state=7)).fit(X, y)
            for _ in range(2)
        ]
        assert torch.equal(torch.random.get_rng_state(), before)
        for name in ("intercept", "coef", "sigma"):
            assert torch.equal(fits[0].posterior_samples_[name], fits[1].posterior_samples_[name])
        other = PyroAdapter(sales, estimator=PyroSVI(num_steps=50, num_samples=20, random_state=8)).fit(X, y)
        assert not torch.equal(other.posterior_samples_["coef"], fits[0].posterior_samples_["coef"])

    def test_global_random_state(self, monkeypatch, sales_cpd, sales_data):
        # Predicting leaves the global torch random state as it was, as fitting and sampling do. Seeding touches only
        # the CPU generator, which is restored: torch.manual_seed would also seed the GPU and MPS generators for good.
        X, y = sales_data
        new = X.iloc[:3]
        before = torch.random.get_rng_state().clone()
        sales_cpd.predict_proba(new).mean()
        sales_cpd.predict(new)
        sales_cpd.log_likelihood(new, y.iloc[:3])
        assert torch.equal(torch.random.get_rng_state(), before)

        seeded = []
        monkeypatch.setattr(torch.cuda, "manual_seed_all", seeded.append)
        monkeypatch.setattr(torch.mps, "manual_seed", seeded.append)
        PyroAdapter(sales, estimator=PyroSVI(num_steps=2, num_samples=2, random_state=0)).fit(X, y).sample(new)
        PyroAdapter(sales, estimator=PyroNUTS(num_samples=4, warmup_steps=2, random_state=0)).fit(X, y)
        assert seeded == []

    def test_predictive_keeps_its_fit(self):
        # A returned distribution keeps the fit it came from, after a refit or a set_params of its adapter.
        estimator = PyroSVI(num_steps=200, num_samples=50, random_state=0)
        node = PyroAdapter(location, estimator=estimator).fit(None, pd.Series([0.0, 0.5, 1.0], name="y"))
        distribution = node.predict_proba()
        mean = float(distribution.mean())
        node.fit(None, pd.Series([9.0, 9.5, 10.0], name="y"))
        assert float(distribution.mean()) == mean != float(node.predict_proba().mean())
        node.set_params(estimator=PyroSVI(num_steps=1))
        assert float(distribution.mean()) == mean

    def test_weights(self):
        # A weight multiplies its row's log-likelihood, so weighted rows give the fit of duplicated rows: the weighted
        # mean and standard deviation, by maximum likelihood.
        estimator = PyroSVI(num_steps=2000, optim=pyro.optim.Adam({"lr": 0.05}))
        weighted = PyroAdapter(location, estimator=estimator).fit(
            None, pd.Series([1.0, 2.0, 4.0, 7.0], name="y"), sample_weight=[1.0, 2.0, 0.0, 3.0]
        )
        duplicated = PyroAdapter(location, estimator=estimator).fit(
            None, pd.Series([1.0, 2.0, 2.0, 7.0, 7.0, 7.0], name="y")
        )
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

        noise = PyroAdapter(laplace, estimator=PyroSVI(num_steps=300, num_samples=100, random_state=0))
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
                PyroAdapter(fn, estimator=PyroSVI(num_steps=1)).fit(X, y)

        # Only discrete variables have states.
        with pytest.raises(ValueError, match="continuous target"):
            PyroAdapter(sales, state_names={"sales": [1.0, 2.0]}, estimator=PyroSVI(num_steps=1)).fit(X, y)

        # NUTS needs latent sites, and can't fit pyro.param values.
        with pytest.raises(ValueError, match="pyro.param"):
            PyroAdapter(location, estimator=PyroNUTS()).fit(None, y)
        with pytest.raises(ValueError, match="latent"):
            PyroAdapter(lambda parents: dist.Normal(0.0, 1.0), estimator=PyroNUTS()).fit(None, y)

        # A location far beyond float32's range of squares makes the first loss infinite.
        with pytest.raises(ValueError, match="isn't finite"):
            PyroAdapter(
                lambda parents: dist.Normal(pyro.param("mu", torch.tensor(1e30)), 1.0), estimator=PyroSVI(num_steps=1)
            ).fit(None, y)


class TestDiscrete:
    def test_logistic_regression(self):
        # With weak priors and many rows, the posterior is close to the maximum likelihood fit. The labels are coded in
        # sorted order, "dry" as 0 and "wet" as 1, and predictions come back as labels.
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"humidity": rng.normal(50, 20, size=1000)})
        scaled = ((X["humidity"] - 50) / 20).to_frame()
        wet_probability = 1 / (1 + np.exp(0.5 - 1.5 * scaled["humidity"]))
        y = pd.Series(np.where(rng.random(1000) < wet_probability, "wet", "dry"), name="ground")
        reference = LogisticRegression(penalty=None).fit(scaled, y)
        cpd = PyroAdapter(wet, estimator=PyroSVI(random_state=0), variable_type="discrete").fit(X, y)
        assert abs(cpd.posterior_samples_["intercept"].mean().item() - reference.intercept_[0]) < 0.15
        assert abs(cpd.posterior_samples_["coef"].mean().item() - reference.coef_[0, 0]) < 0.15

        new, observed = X.iloc[:5], y.iloc[:5]
        dist_ = cpd.predict_proba(new)
        assert list(dist_.categories) == ["dry", "wet"] and dist_.index.equals(new.index)
        probs = np.asarray(dist_.probs)
        np.testing.assert_allclose(probs, reference.predict_proba(scaled.iloc[:5]), atol=0.03)
        assert set(cpd.predict(new)["ground"]) <= {"dry", "wet"}
        expected = np.log(probs[np.arange(5), (observed == "wet").astype(int)])
        np.testing.assert_allclose(cpd.log_likelihood(new, observed)["ground"], expected)
        assert set(cpd.sample(new, n_samples=3, random_state=0)["ground"]) <= {"dry", "wet"}

    def test_categorical_with_a_discrete_parent(self):
        # The season reaches fn as codes over its sorted states, so the posterior predictive is close to the weather's
        # frequencies in each season, as TabularCPD counts them.
        rng = np.random.default_rng(1)
        seasons = ["autumn", "spring", "summer", "winter"]
        states = ["cloudy", "rainy", "sunny"]
        table = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
        codes = rng.integers(0, 4, size=2000)
        X = pd.DataFrame({"season": np.array(seasons)[codes]})
        y = pd.Series([states[rng.choice(3, p=table[code])] for code in codes], name="weather")
        cpd = PyroAdapter(weather, estimator=PyroSVI(random_state=0), variable_type="discrete").fit(X, y)
        assert cpd.state_names_ == {"weather": states, "season": seasons}
        probs = np.asarray(cpd.predict_proba(pd.DataFrame({"season": seasons})).probs)
        np.testing.assert_allclose(probs, TabularCPD().fit(X, y).cpt_.T, atol=0.03)

    def test_states(self):
        # Given state_names fix the target's states, in their order: here "c" is coded 0, "a" 1 and "b" 2.
        y = pd.Series(["a", "b", "a"], name="y")
        fixed = lambda parents: dist.Categorical(probs=torch.tensor([0.5, 0.3, 0.2]))  # noqa: E731
        settings = {"estimator": PyroSVI(num_steps=1), "variable_type": "discrete"}
        root = PyroAdapter(fixed, state_names={"y": ["c", "a", "b"]}, **settings).fit(None, y).predict_proba()
        assert root.shape == () and [float(root.pmf(state)) for state in "cab"] == pytest.approx([0.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="unexpected states"):
            PyroAdapter(fixed, state_names={"y": ["a", "c"]}, **settings).fit(None, y)

        # A discrete parent reaches fn as integer codes over its sorted states: "off" is 0 and "on" is 1. A parent
        # listed in state_names is discrete even if it's numeric.
        switch = lambda parents: dist.Bernoulli(probs=torch.tensor([0.1, 0.9])[parents["x"]])  # noqa: E731
        z = pd.Series(["yes", "no", "yes"], name="z")
        cpd = PyroAdapter(switch, **settings).fit(pd.DataFrame({"x": ["on", "off", "on"]}), z)
        probs = np.asarray(cpd.predict_proba(pd.DataFrame({"x": ["off", "on"]})).probs)
        np.testing.assert_allclose(probs, [[0.9, 0.1], [0.1, 0.9]], atol=1e-6)
        with pytest.raises(ValueError, match="not seen in fit"):
            cpd.predict(pd.DataFrame({"x": ["dim"]}))
        numeric = PyroAdapter(switch, state_names={"x": [5, 7]}, **settings).fit(pd.DataFrame({"x": [7, 5, 7]}), z)
        np.testing.assert_allclose(
            np.asarray(numeric.predict_proba(pd.DataFrame({"x": [5]})).probs), [[0.9, 0.1]], atol=1e-6
        )

        # Rows with weight 0 count for nothing but keep their states, as in TabularCPD: "c" of the target here, and
        # "dim" of the parent.
        weighted = PyroAdapter(fixed, **settings).fit(None, pd.Series(list("abc"), name="y"), sample_weight=[1, 1, 0])
        assert weighted.state_names_ == {"y": ["a", "b", "c"]}
        dimmer = lambda parents: dist.Bernoulli(probs=torch.tensor([0.5, 0.1, 0.9])[parents["x"]])  # noqa: E731
        lamp = PyroAdapter(dimmer, **settings).fit(
            pd.DataFrame({"x": ["on", "off", "dim"]}), z, sample_weight=[1, 1, 0]
        )
        assert lamp.state_names_ == {"z": ["no", "yes"], "x": ["dim", "off", "on"]}

        # Labels of several types, which can't be sorted, are fine when state_names lists them, in its order.
        mixed = pd.Series([1, "b", ("c", 3)], name="y")
        root = PyroAdapter(fixed, state_names={"y": ["b", 1, ("c", 3)]}, **settings).fit(None, mixed).predict_proba()
        assert [float(root.pmf(state)) for state in ("b", 1)] == pytest.approx([0.5, 0.3])

    def test_discrete_errors(self):
        y = pd.Series(["a", "b", "c"], name="y")
        with pytest.raises(ValueError, match="variable_type must be"):
            PyroAdapter(sales, variable_type="categorical")
        # A discrete target needs a distribution over the codes of its states, and data its distribution allows.
        for fn, match in (
            (lambda parents: dist.Normal(0.0, 1.0), "discrete support"),
            (lambda parents: dist.Categorical(logits=torch.zeros(4)), "all its probability"),
            (lambda parents: dist.Bernoulli(probs=torch.tensor(0.5)), "outside the support"),
        ):
            with pytest.raises(ValueError, match=match):
                PyroAdapter(fn, estimator=PyroSVI(num_steps=1), variable_type="discrete").fit(None, y)

        # Codes outside the support get probability 0, also in a family that wraps another, as mask does.
        def binomial(parents):
            return dist.Binomial(2, torch.tensor(0.5)).mask(True)

        counts = pd.Series([0, 1, 2], name="k")
        cpd = PyroAdapter(binomial, "discrete", {"k": [0, 1, 2, 3]}, PyroSVI(num_steps=1)).fit(None, counts)
        np.testing.assert_allclose(np.asarray(cpd.predict_proba().probs), [0.25, 0.5, 0.25, 0.0])

    def test_discrete_weights(self):
        # A weight of 3 counts a row three times: maximum likelihood gives the weighted frequencies 3/6, 2/6 and 1/6.
        def frequencies(parents):
            return dist.Categorical(logits=pyro.param("logits", torch.zeros(3)))

        estimator = PyroSVI(num_steps=2000, optim=pyro.optim.Adam({"lr": 0.05}))
        cpd = PyroAdapter(frequencies, estimator=estimator, variable_type="discrete")
        cpd.fit(None, pd.Series(["a", "b", "b", "c"], name="y"), sample_weight=[3.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(np.asarray(cpd.predict_proba().probs), [3 / 6, 2 / 6, 1 / 6], atol=1e-3)

    def test_continuous_target_with_a_discrete_parent(self):
        # The group's codes index a vector of means, which maximum likelihood sets to the group means.
        def group_means(parents):
            return dist.Normal(pyro.param("mu", torch.zeros(2))[parents["group"]], 1.0)

        X = pd.DataFrame({"group": ["x", "y", "x", "y"]})
        y = pd.Series([1.0, 3.5, 1.5, 2.5], name="v")
        estimator = PyroSVI(num_steps=2000, optim=pyro.optim.Adam({"lr": 0.05}))
        cpd = PyroAdapter(group_means, estimator=estimator).fit(X, y)
        np.testing.assert_allclose(cpd.params_["mu"], [1.25, 3.0], atol=1e-3)
        np.testing.assert_allclose(cpd.predict(pd.DataFrame({"group": ["y", "x"]}))["v"], [3.0, 1.25], atol=1e-3)
