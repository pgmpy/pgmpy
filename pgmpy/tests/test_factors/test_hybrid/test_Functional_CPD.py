import unittest

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.models.LinearGaussianBayesianNetwork import LinearGaussianBayesianNetwork


class TestFCPD(unittest.TestCase):
    def test_class_init(self):
        """
        Test the initialization of the FunctionalCPD class.
        """
        cpd = FunctionalCPD(
            variable="x3",
            fn=lambda parent_sample: dist.Normal(
                1.0 + 0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"], 1
            ),
            parents=["x1", "x2"],
        )

        self.assertEqual(cpd.variable, "x3")
        self.assertEqual(cpd.parents, ["x1", "x2"])
        self.assertTrue(
            callable(cpd.fn), "The function passed to FunctionalCPD must be callable."
        )

    def test_linear_gaussian(self):
        """
        Test the equivalence of FunctionalCPD with LinearGaussianCPD sampling.
        """
        x1_cpd = LinearGaussianCPD("x1", [0], 1.0)

        x2_cpd = LinearGaussianCPD("x2", [0], 1.0)

        x3_cpd = LinearGaussianCPD(
            "x3",
            [1.0, 0.2, 0.3],
            1.0,
            evidence=["x1", "x2"],
        )
        num_samples = 10000

        lgbn = LinearGaussianBayesianNetwork([("x1", "x3"), ("x2", "x3")])
        lgbn.add_cpds(x1_cpd, x2_cpd, x3_cpd)

        linear_gaussian_samples = lgbn.simulate(num_samples, seed=42)

        functional_cpd = FunctionalCPD(
            variable="x3",
            fn=lambda parent_sample: dist.Normal(
                1.0 + 0.2 * parent_sample["x1"] + 0.3 * parent_sample["x2"], 1
            ),
            parents=["x1", "x2"],
        )

        functional_samples = functional_cpd.sample(
            num_samples, linear_gaussian_samples[["x1", "x2"]]
        )

        functional_mean = functional_samples.mean()
        functional_variance = functional_samples.var()
        linear_gaussian_mean = linear_gaussian_samples["x3"].mean()
        linear_gaussian_variance = linear_gaussian_samples["x3"].var()

        tolerance = 1e-1

        self.assertAlmostEqual(
            functional_mean,
            linear_gaussian_mean,
            delta=tolerance,
            msg=f"Functional mean ({functional_mean}) differs from LinearGaussian mean ({linear_gaussian_mean})",
        )
        self.assertAlmostEqual(
            functional_variance,
            linear_gaussian_variance,
            delta=tolerance,
            msg=f"Functional variance ({functional_variance}) differs from LinearGaussian variance ({linear_gaussian_variance})",
        )

    def test_different_distributions(self):
        exp_cpd = FunctionalCPD("exponential", lambda _: dist.Exponential(rate=2.0))

        exp_samples = exp_cpd.sample(n_samples=5000)
        self.assertTrue(np.all(exp_samples >= 0))
        self.assertAlmostEqual(np.mean(exp_samples), 0.5, delta=0.1)

        uni_cpd = FunctionalCPD(
            "uniform",
            lambda parent: dist.Uniform(
                low=parent["exponential"], high=parent["exponential"] + 5
            ),
            parents=["exponential"],
        )

        exp_samples = pd.DataFrame({"exponential": exp_samples})

        uni_samples = uni_cpd.sample(n_samples=5000, parent_sample=exp_samples)

        self.assertTrue(np.all(uni_samples >= exp_samples["exponential"]))
        self.assertTrue(np.all(uni_samples <= exp_samples["exponential"] + 5))
