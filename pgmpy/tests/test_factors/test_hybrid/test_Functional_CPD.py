import unittest

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch

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

    def test_sample_vectorized(self):
        """
        Test FunctionalCPD with vectorized sampling.
        """

        def vectorized_fn(parent_sample):
            x1 = torch.tensor(parent_sample["x1"].values, dtype=torch.float32)
            x2 = torch.tensor(parent_sample["x2"].values, dtype=torch.float32)
            mean = 1.0 + 0.5 * x1 + 0.25 * x2
            return dist.Normal(mean, torch.ones_like(mean))

        cpd = FunctionalCPD(
            variable="x3", fn=vectorized_fn, parents=["x1", "x2"], vectorized=True
        )

        parent_samples = pd.DataFrame(
            {"x1": np.random.randn(1000), "x2": np.random.randn(1000)}
        )

        samples = cpd.sample(n_samples=1000, parent_sample=parent_samples)
        self.assertEqual(len(samples), 1000)
        self.assertTrue(np.isfinite(samples).all())

    def test_sample_iterative(self):
        """
        Test FunctionalCPD with iterative sampling (vectorized=False).
        """

        def row_fn(row):
            mean = 1.0 + 0.5 * row["x1"] + 0.25 * row["x2"]
            return dist.Normal(mean, 1.0)

        cpd = FunctionalCPD(
            variable="x3", fn=row_fn, parents=["x1", "x2"], vectorized=False
        )

        parent_samples = pd.DataFrame(
            {"x1": np.random.randn(1000), "x2": np.random.randn(1000)}
        )

        samples = cpd.sample(n_samples=1000, parent_sample=parent_samples)
        self.assertEqual(len(samples), 1000)
        self.assertTrue(np.isfinite(samples).all())

    # Test parent sample none vectorized
    def test_vectorized_without_parent(self):
        """
        Test FunctionalCPD with vectorized sampling without parents.
        """

        def vectorized_fn(parent_sample):
            return dist.Normal(torch.zeros(1000), torch.ones(1000))

        cpd = FunctionalCPD(variable="z", fn=vectorized_fn, parents=[], vectorized=True)
        samples = cpd.sample(n_samples=1000)
        self.assertEqual(len(samples), 1000)
        self.assertTrue(np.isfinite(samples).all())

    # Test parent sample none iterative
    def test_iterative_without_parent(self):
        """
        Test FunctionalCPD with iterative sampling (vectorized=False) without parents.
        """

        def iterative_fn(parent_sample):
            return dist.Normal(0.0, 1.0)

        cpd = FunctionalCPD(variable="z", fn=iterative_fn, parents=[], vectorized=False)
        samples = cpd.sample(n_samples=1000)
        self.assertEqual(len(samples), 1000)
        self.assertTrue(np.isfinite(samples).all())

    # Benchmark test to see performance improvement between vectorized vs iterative. Use -s option to run this test to be able to see prints.
    def test_sampling_time_vectorized_vs_iterative(self):
        """
        Compare sampling time between vectorized=True and vectorized=False.
        """
        import time

        n_samples = 10000
        parent_samples = pd.DataFrame(
            {"x1": np.random.randn(n_samples), "x2": np.random.randn(n_samples)}
        )

        def fn_vectorized(parent_sample):
            x1 = torch.tensor(parent_sample["x1"].values, dtype=torch.float32)
            x2 = torch.tensor(parent_sample["x2"].values, dtype=torch.float32)
            mean = 1.0 + 0.5 * x1 + 0.25 * x2
            return dist.Normal(mean, torch.ones_like(mean))

        def fn_iterative(row):
            mean = 1.0 + 0.5 * row["x1"] + 0.25 * row["x2"]
            return dist.Normal(mean, 1.0)

        cpd_vec = FunctionalCPD(
            "x3", fn=fn_vectorized, parents=["x1", "x2"], vectorized=True
        )
        start_vec = time.time()
        _ = cpd_vec.sample(n_samples=n_samples, parent_sample=parent_samples)
        end_vec = time.time()

        cpd_iter = FunctionalCPD(
            "x3", fn=fn_iterative, parents=["x1", "x2"], vectorized=False
        )
        start_iter = time.time()
        _ = cpd_iter.sample(n_samples=n_samples, parent_sample=parent_samples)
        end_iter = time.time()

        print(f"\nVectorized sampling time   : {end_vec - start_vec:.4f} seconds")
        print(
            f"Iterative sampling time iterative  : {end_iter - start_iter:.4f} seconds"
        )
        self.assertTrue((end_vec - start_vec) < (end_iter - start_iter))
