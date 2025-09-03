import unittest

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.utils._safe_import import _safe_import

pyro = _safe_import("pyro", pkg_name="pyro-ppl")
dist = _safe_import("pyro.distributions", pkg_name="pyro-ppl")
torch = _safe_import("torch")

from pgmpy import config
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.hybrid.FunctionalCPD import FunctionalCPD
from pgmpy.models import FunctionalBayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.utils import get_example_model
from pgmpy.utils._safe_import import _safe_import

torch = _safe_import("torch")


@unittest.skipUnless(
    _check_soft_dependencies("pyro-ppl", severity="none"),
    reason="execute only if required dependency present",
)
class TestFBNMethods(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")
        self.model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        self.cpd1 = FunctionalCPD(
            "x1",
            lambda _: dist.Normal(0, 1),
        )
        self.cpd2 = FunctionalCPD(
            "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        )
        self.cpd3 = FunctionalCPD(
            "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        )

        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)

    def test_cpds_simple(self):
        self.assertEqual("x1", self.cpd1.variable)
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)
        self.assertEqual(cpd.parents, self.cpd1.parents)
        self.assertEqual(cpd.parents, [])

    def test_add_cpds(self):
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)

        cpd = self.model.get_cpds("x2")
        self.assertEqual(cpd.variable, self.cpd2.variable)
        self.assertEqual(cpd.parents, self.cpd2.parents)

        cpd = self.model.get_cpds("x3")
        self.assertEqual(cpd.variable, self.cpd3.variable)
        self.assertEqual(cpd.parents, self.cpd3.parents)

        tab_cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        self.assertRaises(ValueError, self.model.add_cpds, tab_cpd)
        self.assertRaises(ValueError, self.model.add_cpds, 1)
        self.assertRaises(ValueError, self.model.add_cpds, 1, tab_cpd)

        self.assertRaises(ValueError, self.model.add_cpds, tab_cpd)

        # Test that duplicate CPDs get replaced.
        self.assertEqual(len(self.model.cpds), 3)
        self.model.add_cpds(self.cpd1)
        self.assertEqual(len(self.model.cpds), 3)

    def test_check_model(self):
        self.assertEqual(self.model.check_model(), True)

        self.model.add_edge("x1", "x4")
        cpd4 = FunctionalCPD(
            "x4", lambda parent: dist.Normal(parent["x2"] * -1 + 4, 3), ["x2"]
        )
        self.model.add_cpds(cpd4)

        self.assertRaises(ValueError, self.model.check_model)

    def test_simulate_linear_gaussian(self):
        lg_model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        lg_cpd1 = LinearGaussianCPD(variable="x1", beta=[1], std=1)
        lg_cpd2 = LinearGaussianCPD(
            variable="x2", beta=[-5, 0.5], std=1, evidence=["x1"]
        )
        lg_cpd3 = LinearGaussianCPD(variable="x3", beta=[4, -1], std=1, evidence=["x2"])
        lg_model.add_cpds(lg_cpd1, lg_cpd2, lg_cpd3)

        fn_model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        fn_cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(1, 1))
        fn_cpd2 = FunctionalCPD(
            "x2",
            lambda parent: dist.Normal(-5 + parent["x1"] * 0.5, 1),
            parents=["x1"],
        )
        fn_cpd3 = FunctionalCPD(
            "x3",
            lambda parent: dist.Normal(4 + parent["x2"] * -1, 1),
            parents=["x2"],
        )
        fn_model.add_cpds(fn_cpd1, fn_cpd2, fn_cpd3)

        n_samples = 5000
        seed = 42
        lg_samples = lg_model.simulate(n_samples=n_samples, seed=seed)
        fn_samples = fn_model.simulate(n_samples=n_samples, seed=seed)

        for var in ["x1", "x2", "x3"]:
            np.testing.assert_allclose(
                lg_samples[var].mean(),
                fn_samples[var].mean(),
                rtol=0.1,
                err_msg=f"Mean mismatch for {var}",
            )
            np.testing.assert_allclose(
                lg_samples[var].std(),
                fn_samples[var].std(),
                rtol=0.1,
                err_msg=f"Standard deviation mismatch for {var}",
            )

    def test_simulate_different_distributions(self):
        model = FunctionalBayesianNetwork(
            [
                ("exponential", "uniform"),
                ("uniform", "lognormal"),
                ("lognormal", "gamma"),
            ]
        )

        cpd1 = FunctionalCPD("exponential", lambda _: dist.Exponential(0.5))

        cpd2 = FunctionalCPD(
            "uniform",
            lambda parent: dist.Uniform(
                parent["exponential"], parent["exponential"] + 2
            ),
            parents=["exponential"],
        )

        cpd3 = FunctionalCPD(
            "lognormal",
            lambda parent: dist.LogNormal(np.log(parent["uniform"]), 1),
            parents=["uniform"],
        )

        cpd4 = FunctionalCPD(
            "gamma",
            lambda parent: dist.Gamma(2.0, parent["lognormal"] / 5),
            parents=["lognormal"],
        )

        model.add_cpds(cpd1, cpd2, cpd3, cpd4)
        n_samples = 10000
        samples = model.simulate(n_samples=n_samples, seed=42)

        self.assertEqual(len(samples), n_samples)
        self.assertEqual(
            set(samples.columns), {"exponential", "uniform", "lognormal", "gamma"}
        )

        self.assertTrue(np.all(samples["exponential"] >= 0))
        self.assertAlmostEqual(samples["exponential"].mean(), 2.0, delta=0.2)

        self.assertTrue(np.all(samples["uniform"] >= samples["exponential"]))
        self.assertTrue(np.all(samples["uniform"] <= samples["exponential"] + 2))

        self.assertTrue(np.all(samples["lognormal"] > 0))
        self.assertTrue(np.all(samples["gamma"] > 0))

    def test_svi_fit_normal(self):
        alpha = 0.5
        beta = 0.8
        x1 = np.random.normal(0.2, 0.9, size=1000)
        x2 = np.random.normal((x1 * alpha) + 0.5, 0.6)
        x3 = np.random.normal((x2 * beta) + 0.3, 0.7)
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        def x1_fn(parent):
            mu = pyro.param("x1_mu", torch.tensor(1.0, device=config.get_device()))
            sigma = pyro.param(
                "x1_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )
            return dist.Normal(mu, sigma)

        def x2_fn(parent):
            intercept = pyro.param(
                "x2_inter", torch.tensor(1.0, device=config.get_device())
            )
            sigma = pyro.param(
                "x2_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )
            alpha = pyro.param(
                "x2_alpha", torch.tensor(1.0, device=config.get_device())
            )
            return dist.Normal(intercept + (parent["x1"] * alpha), sigma)

        def x3_fn(parent):
            intercept = pyro.param(
                "x3_inter", torch.tensor(1.0, device=config.get_device())
            )
            sigma = pyro.param(
                "x3_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )
            alpha = pyro.param("x3_beta", torch.tensor(1.0, device=config.get_device()))
            return dist.Normal(intercept + (parent["x2"] * alpha), sigma)

        cpd1 = FunctionalCPD("x1", fn=x1_fn)
        cpd2 = FunctionalCPD("x2", fn=x2_fn, parents=["x1"])
        cpd3 = FunctionalCPD("x3", fn=x3_fn, parents=["x2"])

        model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        model.add_cpds(cpd1, cpd2, cpd3)

        params = model.fit(
            data,
            method="svi",
            optimizer=pyro.optim.Adam({"lr": 0.05}),
            seed=42,
            num_steps=100,
        )

        self.assertIn("x1_mu", params)
        self.assertIn("x1_sigma", params)
        self.assertIn("x2_inter", params)
        self.assertIn("x2_sigma", params)
        self.assertIn("x2_alpha", params)
        self.assertIn("x3_inter", params)
        self.assertIn("x3_sigma", params)
        self.assertIn("x3_beta", params)

        self.assertAlmostEqual(params["x1_mu"], 0.2, delta=0.1)
        self.assertAlmostEqual(params["x1_sigma"], 0.9, delta=0.1)
        self.assertAlmostEqual(params["x2_inter"], 0.5, delta=0.1)
        self.assertAlmostEqual(params["x2_sigma"], 0.6, delta=0.1)
        self.assertAlmostEqual(params["x2_alpha"], 0.5, delta=0.1)
        self.assertAlmostEqual(params["x3_inter"], 0.3, delta=0.1)
        self.assertAlmostEqual(params["x3_sigma"], 0.7, delta=0.1)
        self.assertAlmostEqual(params["x3_beta"], 0.8, delta=0.1)

    def test_svi_fit_different_distributions(self):
        x1 = np.random.beta(0.2, 0.8, size=1000)
        x2 = np.random.poisson(x1 + 0.3)
        x3 = np.random.poisson(x2 + 0.5)
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        def x1_prior(parents):
            concen1 = pyro.param(
                "x1_concen1", torch.tensor(1.0, device=config.get_device())
            )
            concen0 = pyro.param(
                "x1_concen0", torch.tensor(1.0, device=config.get_device())
            )
            return dist.Beta(concen1, concen0)

        def x2_prior(parents):
            rate = pyro.param("x2_rate", torch.tensor(1.0, device=config.get_device()))
            return dist.Poisson(rate + parents["x1"])

        def x3_prior(parents):
            rate = pyro.param("x3_rate", torch.tensor(1.0, device=config.get_device()))
            return dist.Poisson(rate + parents["x2"])

        cpd1 = FunctionalCPD("x1", fn=x1_prior)
        cpd2 = FunctionalCPD("x2", fn=x2_prior, parents=["x1"])
        cpd3 = FunctionalCPD("x3", fn=x3_prior, parents=["x2"])

        model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        model.add_cpds(cpd1, cpd2, cpd3)

        params = model.fit(
            data,
            method="SVI",
            optimizer=pyro.optim.Adam({"lr": 0.05}),
            seed=42,
            num_steps=100,
        )

        self.assertIn("x1_concen1", params)
        self.assertIn("x1_concen0", params)
        self.assertIn("x2_rate", params)
        self.assertIn("x3_rate", params)

        self.assertAlmostEqual(params["x1_concen1"], 0.2, delta=0.1)
        self.assertAlmostEqual(params["x1_concen0"], 0.8, delta=0.1)
        self.assertAlmostEqual(params["x2_rate"], 0.3, delta=0.1)
        self.assertAlmostEqual(params["x3_rate"], 0.5, delta=0.1)

    def test_mcmc_fit_normal(self):
        alpha = 0.5
        beta = 0.8
        x1 = np.random.normal(0.2, 0.9, size=1000)
        x2 = np.random.normal((x1 * alpha) + 0.5, 0.6)
        x3 = np.random.normal((x2 * beta) + 0.3, 0.7)
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        def t(v):
            return torch.tensor(v, dtype=config.get_dtype(), device=config.get_device())

        def prior_fn():
            return {
                "x1_mu": dist.Uniform(t(0.0), t(1.0)),
                "x1_sigma": dist.HalfNormal(t(2.0)),
                "x2_inter": dist.Uniform(t(0.0), t(1.0)),
                "x2_sigma": dist.HalfNormal(t(2.0)),
                "x2_alpha": dist.Uniform(t(0.0), t(1.0)),
                "x3_inter": dist.Uniform(t(0.0), t(1.0)),
                "x3_sigma": dist.HalfNormal(t(2.0)),
                "x3_beta": dist.Uniform(t(0.0), t(1.0)),
            }

        def x1_fn(priors, parents):
            return dist.Normal(priors["x1_mu"], priors["x1_sigma"])

        def x2_fn(priors, parents):
            return dist.Normal(
                priors["x2_inter"] + (priors["x2_alpha"] * parents["x1"]),
                priors["x2_sigma"],
            )

        def x3_fn(priors, parents):
            return dist.Normal(
                priors["x3_inter"] + (priors["x3_beta"] * parents["x2"]),
                priors["x3_sigma"],
            )

        cpd1 = FunctionalCPD("x1", fn=x1_fn)
        cpd2 = FunctionalCPD("x2", fn=x2_fn, parents=["x1"])
        cpd3 = FunctionalCPD("x3", fn=x3_fn, parents=["x2"])

        model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        model.add_cpds(cpd1, cpd2, cpd3)

        pyro.clear_param_store()
        params = model.fit(
            data,
            method="MCMC",
            prior_fn=prior_fn,
            seed=42,
            num_steps=100,
            nuts_kwargs={"target_accept_prob": 0.8},
            mcmc_kwargs={"num_chains": 1, "warmup_steps": 100},
        )

        for k in [
            "x1_mu",
            "x1_sigma",
            "x2_inter",
            "x2_sigma",
            "x2_alpha",
            "x3_inter",
            "x3_sigma",
            "x3_beta",
        ]:
            self.assertIn(k, params)

        self.assertAlmostEqual(params["x1_mu"].mean().item(), 0.2, delta=0.1)
        self.assertAlmostEqual(params["x1_sigma"].mean().item(), 0.9, delta=0.1)
        self.assertAlmostEqual(params["x2_inter"].mean().item(), 0.5, delta=0.1)
        self.assertAlmostEqual(params["x2_sigma"].mean().item(), 0.6, delta=0.1)
        self.assertAlmostEqual(params["x2_alpha"].mean().item(), 0.5, delta=0.1)
        self.assertAlmostEqual(params["x3_inter"].mean().item(), 0.3, delta=0.1)
        self.assertAlmostEqual(params["x3_sigma"].mean().item(), 0.7, delta=0.1)
        self.assertAlmostEqual(params["x3_beta"].mean().item(), 0.8, delta=0.1)

    def test_mcmc_fit_different_distributions(self):
        # synthetic data
        x1 = np.random.beta(0.2, 0.8, size=1000)
        x2 = np.random.poisson(x1 + 0.3)
        x3 = np.random.poisson(x2 + 0.5)
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        def t(v):
            return torch.tensor(v, dtype=config.get_dtype(), device=config.get_device())

        def prior_fn():
            return {
                "x1_concen1": dist.Uniform(t(0.0), t(1.0)),
                "x1_concen0": dist.Uniform(t(0.0), t(1.0)),
                "x2_rate": dist.Uniform(t(0.0), t(1.0)),
                "x3_rate": dist.Uniform(t(0.0), t(1.0)),
            }

        def x1_prior(priors, parents):
            return dist.Beta(priors["x1_concen1"], priors["x1_concen0"])

        def x2_prior(priors, parents):
            return dist.Poisson(priors["x2_rate"] + parents["x1"])

        def x3_prior(priors, parents):
            return dist.Poisson(priors["x3_rate"] + parents["x2"])

        cpd1 = FunctionalCPD("x1", x1_prior)
        cpd2 = FunctionalCPD("x2", x2_prior, parents=["x1"])
        cpd3 = FunctionalCPD("x3", x3_prior, parents=["x2"])

        model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        model.add_cpds(cpd1, cpd2, cpd3)

        pyro.clear_param_store()
        params = model.fit(
            data, method="MCMC", prior_fn=prior_fn, seed=42, num_steps=100
        )

        self.assertIn("x1_concen1", params)
        self.assertIn("x1_concen0", params)
        self.assertIn("x2_rate", params)
        self.assertIn("x3_rate", params)

        self.assertAlmostEqual(params["x1_concen1"].mean().item(), 0.2, delta=0.1)
        self.assertAlmostEqual(params["x1_concen0"].mean().item(), 0.8, delta=0.1)
        self.assertAlmostEqual(params["x2_rate"].mean().item(), 0.3, delta=0.1)
        self.assertAlmostEqual(params["x3_rate"].mean().item(), 0.5, delta=0.1)

    def test_fit_complex_svi(self):
        sim_model = get_example_model("ecoli70")
        df = sim_model.simulate(n_samples=int(1e3), seed=42)

        model = FunctionalBayesianNetwork(
            [
                ("b1191", "fixC"),
                ("fixC", "ygbD"),
                ("fixC", "yjbO"),
                ("fixC", "yceP"),
                ("yceP", "ibpB"),
                ("eutG", "yceP"),
                ("eutG", "ibpB"),
            ]
        )
        df = df.loc[:, list(model.nodes())]

        def fn_b1191_param(parents):
            mu = pyro.param("b1191_mu", torch.tensor(1.0, device=config.get_device()))
            sigma = pyro.param(
                "b1191_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )
            return dist.Normal(mu, sigma)

        def fn_eutG_param(parents):
            mu = pyro.param("eutG_mu", torch.tensor(1.0, device=config.get_device()))
            sigma = pyro.param(
                "eutG_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )
            return dist.Normal(mu, sigma)

        def fn_fixC_param(parents):
            mu = (
                pyro.param("fixC_inter", torch.tensor(1.0, device=config.get_device()))
                + pyro.param(
                    "fixC_alpha", torch.tensor(1.0, device=config.get_device())
                )
                * parents["b1191"]
            )
            sigma = pyro.param(
                "fixC_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )
            return dist.Normal(mu, sigma)

        def fn_ygbD_param(parents):
            mu = (
                pyro.param("ygbD_inter", torch.tensor(1.0, device=config.get_device()))
                + pyro.param(
                    "ygbD_alpha", torch.tensor(1.0, device=config.get_device())
                )
                * parents["fixC"]
            )
            sigma = pyro.param(
                "ygbD_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )

            return dist.Normal(mu, sigma)

        def fn_yjbO_param(parents):
            mu = (
                pyro.param("ygbO_inter", torch.tensor(1.0, device=config.get_device()))
                + pyro.param(
                    "ygbO_alpha", torch.tensor(1.0, device=config.get_device())
                )
                * parents["fixC"]
            )
            sigma = pyro.param(
                "ygbO_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )

            return dist.Normal(mu, sigma)

        def fn_yceP_param(parents):
            mu = (
                pyro.param("yceP_inter", torch.tensor(1.0, device=config.get_device()))
                + pyro.param(
                    "yceP_alpha0", torch.tensor(1.0, device=config.get_device())
                )
                * parents["eutG"]
                + pyro.param(
                    "yceP_alpha1", torch.tensor(1.0, device=config.get_device())
                )
                * parents["fixC"]
            )
            sigma = pyro.param(
                "yceP_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )

            return dist.Normal(mu, sigma)

        def fn_ibpB_param(parents):
            mu = (
                pyro.param("ibpB_inter", torch.tensor(1.0, device=config.get_device()))
                + pyro.param(
                    "ibpB_alpha0", torch.tensor(1.0, device=config.get_device())
                )
                * parents["eutG"]
                + pyro.param(
                    "ibpB_alpha1", torch.tensor(1.0, device=config.get_device())
                )
                * parents["yceP"]
            )
            sigma = pyro.param(
                "ibpB_sigma",
                torch.tensor(1.0, device=config.get_device()),
                constraint=torch.distributions.constraints.positive,
            )

            return dist.Normal(mu, sigma)

        b1191_cpd = FunctionalCPD("b1191", fn=fn_b1191_param)
        eutG_cpd = FunctionalCPD("eutG", fn=fn_eutG_param)
        fixC_cpd = FunctionalCPD("fixC", fn=fn_fixC_param, parents=["b1191"])
        ygbD_cpd = FunctionalCPD("ygbD", fn=fn_ygbD_param, parents=["fixC"])
        yjbO_cpd = FunctionalCPD("yjbO", fn=fn_yjbO_param, parents=["fixC"])
        yceP_cpd = FunctionalCPD("yceP", fn=fn_yceP_param, parents=["eutG", "fixC"])
        ibpB_cpd = FunctionalCPD("ibpB", fn=fn_ibpB_param, parents=["eutG", "yceP"])

        model.add_cpds(
            b1191_cpd, eutG_cpd, fixC_cpd, ygbD_cpd, yjbO_cpd, yceP_cpd, ibpB_cpd
        )

        params = model.fit(
            df,
            method="SVI",
            optimizer=pyro.optim.Adam({"lr": 0.05}),
            seed=42,
            num_steps=200,
        )

        self.assertIn("b1191_mu", params)
        self.assertIn("b1191_sigma", params)
        self.assertIn("eutG_mu", params)
        self.assertIn("fixC_inter", params)
        self.assertIn("fixC_alpha", params)
        self.assertIn("ygbD_inter", params)
        self.assertIn("ygbD_alpha", params)
        self.assertIn("ygbO_inter", params)
        self.assertIn("ygbO_alpha", params)
        self.assertIn("yceP_inter", params)
        self.assertIn("yceP_alpha0", params)
        self.assertIn("yceP_alpha1", params)
        self.assertIn("ibpB_inter", params)
        self.assertIn("ibpB_alpha0", params)
        self.assertIn("ibpB_alpha1", params)

        self.assertAlmostEqual(params["b1191_mu"], 1.273, delta=0.2)
        self.assertAlmostEqual(params["b1191_sigma"], 0.609, delta=0.25)

        self.assertAlmostEqual(params["eutG_mu"], 1.265, delta=0.2)
        self.assertAlmostEqual(params["eutG_sigma"], 0.691, delta=0.2)

        self.assertAlmostEqual(params["fixC_inter"], 0.316, delta=0.2)
        self.assertAlmostEqual(params["fixC_alpha"], 0.941, delta=0.2)
        self.assertAlmostEqual(params["fixC_sigma"], 1.131, delta=0.2)

        self.assertAlmostEqual(params["ygbD_inter"], 1.35, delta=0.2)
        self.assertAlmostEqual(params["ygbD_alpha"], 0.661, delta=0.2)
        self.assertAlmostEqual(params["ygbD_sigma"], 0.74, delta=0.2)

        self.assertAlmostEqual(params["ygbO_inter"], 1.591, delta=0.2)
        self.assertAlmostEqual(params["ygbO_alpha"], -0.071, delta=0.2)
        self.assertAlmostEqual(params["ygbO_sigma"], 1.851, delta=0.6)

        self.assertAlmostEqual(params["yceP_inter"], -0.128, delta=0.2)
        self.assertAlmostEqual(params["yceP_alpha0"], 1.141, delta=0.2)
        self.assertAlmostEqual(params["yceP_alpha1"], -0.327, delta=0.2)
        self.assertAlmostEqual(params["yceP_sigma"], 0.167, delta=0.3)

        self.assertAlmostEqual(params["ibpB_inter"], -0.423, delta=0.2)
        self.assertAlmostEqual(params["ibpB_alpha0"], 1.447, delta=0.2)
        self.assertAlmostEqual(params["ibpB_alpha1"], 0.125, delta=0.2)
        self.assertAlmostEqual(params["ibpB_sigma"], 0.461, delta=0.3)

    def test_fit_complex_mcmc(self):
        sim_model = get_example_model("ecoli70")
        df = sim_model.simulate(n_samples=int(1e3), seed=42)

        model = FunctionalBayesianNetwork(
            [
                ("b1191", "fixC"),
                ("fixC", "ygbD"),
                ("fixC", "yjbO"),
                ("fixC", "yceP"),
                ("yceP", "ibpB"),
                ("eutG", "yceP"),
                ("eutG", "ibpB"),
            ]
        )
        df = df.loc[:, list(model.nodes())]

        def t(v):
            return torch.tensor(v, dtype=config.get_dtype(), device=config.get_device())

        def prior_fn():
            return {
                # b1191 root
                "b1191_mu": dist.Uniform(t(-1.0), t(2.0)),
                "b1191_sigma": dist.HalfNormal(t(2.0)),
                # eutG root
                "eutG_mu": dist.Uniform(t(-1.0), t(2.0)),
                "eutG_sigma": dist.HalfNormal(t(2.0)),
                # fixC
                "fixC_inter": dist.Uniform(t(-1.0), t(2.0)),
                "fixC_alpha": dist.Uniform(t(-1.0), t(2.0)),
                "fixC_sigma": dist.HalfNormal(t(2.0)),
                # ygbD
                "ygbD_inter": dist.Uniform(t(-1.0), t(2.0)),
                "ygbD_alpha": dist.Uniform(t(-1.0), t(2.0)),
                "ygbD_sigma": dist.HalfNormal(t(2.0)),
                # yjbO
                "ygbO_inter": dist.Uniform(t(-1.0), t(2.0)),
                "ygbO_alpha": dist.Uniform(t(-1.0), t(2.0)),
                "ygbO_sigma": dist.HalfNormal(t(2.0)),
                # yceP
                "yceP_inter": dist.Uniform(t(-1.0), t(2.0)),
                "yceP_alpha0": dist.Uniform(t(-1.0), t(2.0)),
                "yceP_alpha1": dist.Uniform(t(-1.0), t(2.0)),
                "yceP_sigma": dist.HalfNormal(t(2.0)),
                # ibpB
                "ibpB_inter": dist.Uniform(t(-1.0), t(2.0)),
                "ibpB_alpha0": dist.Uniform(t(-1.0), t(2.0)),
                "ibpB_alpha1": dist.Uniform(t(-1.0), t(2.0)),
                "ibpB_sigma": dist.HalfNormal(t(2.0)),
            }

        def fn_b1191(priors, parents):
            return dist.Normal(priors["b1191_mu"], priors["b1191_sigma"])

        def fn_eutG(priors, parents):
            return dist.Normal(priors["eutG_mu"], priors["eutG_sigma"])

        def fn_fixC(priors, parents):
            mu = priors["fixC_inter"] + priors["fixC_alpha"] * parents["b1191"]
            return dist.Normal(mu, priors["fixC_sigma"])

        def fn_ygbD(priors, parents):
            mu = priors["ygbD_inter"] + priors["ygbD_alpha"] * parents["fixC"]
            return dist.Normal(mu, priors["ygbD_sigma"])

        def fn_yjbO(priors, parents):
            mu = priors["ygbO_inter"] + priors["ygbO_alpha"] * parents["fixC"]
            return dist.Normal(mu, priors["ygbO_sigma"])

        def fn_yceP(priors, parents):
            mu = (
                priors["yceP_inter"]
                + priors["yceP_alpha0"] * parents["eutG"]
                + priors["yceP_alpha1"] * parents["fixC"]
            )
            return dist.Normal(mu, priors["yceP_sigma"])

        def fn_ibpB(priors, parents):
            mu = (
                priors["ibpB_inter"]
                + priors["ibpB_alpha0"] * parents["eutG"]
                + priors["ibpB_alpha1"] * parents["yceP"]
            )
            return dist.Normal(mu, priors["ibpB_sigma"])

        model.add_cpds(
            FunctionalCPD("b1191", fn=fn_b1191),
            FunctionalCPD("eutG", fn=fn_eutG),
            FunctionalCPD("fixC", fn=fn_fixC, parents=["b1191"]),
            FunctionalCPD("ygbD", fn=fn_ygbD, parents=["fixC"]),
            FunctionalCPD("yjbO", fn=fn_yjbO, parents=["fixC"]),
            FunctionalCPD("yceP", fn=fn_yceP, parents=["eutG", "fixC"]),
            FunctionalCPD("ibpB", fn=fn_ibpB, parents=["eutG", "yceP"]),
        )

        pyro.clear_param_store()

        params = model.fit(
            df,
            method="MCMC",
            prior_fn=prior_fn,
            seed=42,
            num_steps=100,
            nuts_kwargs={"target_accept_prob": 0.8},
            mcmc_kwargs={"num_chains": 1, "warmup_steps": 100},
        )

        for k in [
            "b1191_mu",
            "b1191_sigma",
            "eutG_mu",
            "eutG_sigma",
            "fixC_inter",
            "fixC_alpha",
            "fixC_sigma",
            "ygbD_inter",
            "ygbD_alpha",
            "ygbD_sigma",
            "ygbO_inter",
            "ygbO_alpha",
            "ygbO_sigma",
            "yceP_inter",
            "yceP_alpha0",
            "yceP_alpha1",
            "yceP_sigma",
            "ibpB_inter",
            "ibpB_alpha0",
            "ibpB_alpha1",
            "ibpB_sigma",
        ]:
            self.assertIn(k, params)

        self.assertAlmostEqual(params["b1191_mu"].mean().item(), 1.273, delta=0.2)
        self.assertAlmostEqual(params["b1191_sigma"].mean().item(), 0.609, delta=0.25)

        self.assertAlmostEqual(params["eutG_mu"].mean().item(), 1.265, delta=0.2)
        self.assertAlmostEqual(params["eutG_sigma"].mean().item(), 0.691, delta=0.25)

        self.assertAlmostEqual(params["fixC_inter"].mean().item(), 0.316, delta=0.2)
        self.assertAlmostEqual(params["fixC_alpha"].mean().item(), 0.941, delta=0.2)
        self.assertAlmostEqual(params["fixC_sigma"].mean().item(), 1.131, delta=0.2)

        self.assertAlmostEqual(params["ygbD_inter"].mean().item(), 1.350, delta=0.2)
        self.assertAlmostEqual(params["ygbD_alpha"].mean().item(), 0.661, delta=0.2)
        self.assertAlmostEqual(params["ygbD_sigma"].mean().item(), 0.740, delta=0.2)

        self.assertAlmostEqual(params["ygbO_inter"].mean().item(), 1.591, delta=0.2)
        self.assertAlmostEqual(params["ygbO_alpha"].mean().item(), -0.071, delta=0.2)
        self.assertAlmostEqual(params["ygbO_sigma"].mean().item(), 1.851, delta=0.6)

        self.assertAlmostEqual(params["yceP_inter"].mean().item(), -0.128, delta=0.2)
        self.assertAlmostEqual(params["yceP_alpha0"].mean().item(), 1.141, delta=0.2)
        self.assertAlmostEqual(params["yceP_alpha1"].mean().item(), -0.327, delta=0.2)
        self.assertAlmostEqual(params["yceP_sigma"].mean().item(), 0.167, delta=0.3)

        self.assertAlmostEqual(params["ibpB_inter"].mean().item(), -0.423, delta=0.2)
        self.assertAlmostEqual(params["ibpB_alpha0"].mean().item(), 1.447, delta=0.2)
        self.assertAlmostEqual(params["ibpB_alpha1"].mean().item(), 0.125, delta=0.2)
        self.assertAlmostEqual(params["ibpB_sigma"].mean().item(), 0.461, delta=0.3)

    def tearDown(self):
        del self.model
        del self.cpd1
        del self.cpd2
        del self.cpd3


@unittest.skipUnless(
    _check_soft_dependencies("pyro-ppl", severity="none"),
    reason="execute only if required dependency present",
)
class TestFBNCreation(unittest.TestCase):
    def test_class_init_with_adj_matrix_dict_of_dict(self):
        adj = {"a": {"b": 4, "c": 3}, "b": {"c": 2}}
        self.graph = FunctionalBayesianNetwork(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])
        self.assertEqual(self.graph.adj["a"]["c"]["weight"], 3)

    def test_class_init_with_adj_matrix_dict_of_list(self):
        adj = {"a": ["b", "c"], "b": ["c"]}
        self.graph = FunctionalBayesianNetwork(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])

    def test_class_init_with_pd_adj_df(self):
        df = pd.DataFrame([[0, 3], [0, 0]])
        self.graph = FunctionalBayesianNetwork(df, latents=set([0]))
        self.assertEqual(self.graph.latents, set([0]))
        self.assertListEqual(sorted(self.graph.nodes()), [0, 1])
        self.assertEqual(self.graph.adj[0][1]["weight"], {"weight": 3})


@unittest.skipUnless(
    _check_soft_dependencies("pyro-ppl", severity="none"),
    reason="execute only if required dependency present",
)
class TestDAGParser(unittest.TestCase):
    def test_from_lavaan(self):
        model_str = "ind60 =~ x1"
        model_from_str = FunctionalBayesianNetwork.from_lavaan(string=model_str)
        expected_edges = set([("ind60", "x1")])
        self.assertEqual(set(model_from_str.edges()), expected_edges)

    def test_from_dagitty(self):
        model_str = """dag{X -> Y}"""
        model_from_str = FunctionalBayesianNetwork.from_dagitty(string=model_str)
        expected_edges = set([("X", "Y")])
        self.assertEqual(set(model_from_str.edges()), expected_edges)
