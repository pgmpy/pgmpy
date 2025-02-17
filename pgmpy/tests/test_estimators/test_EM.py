import unittest

import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor

from pgmpy import config
from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import compat_fns, get_example_model


class TestEM(unittest.TestCase):
    def setUp(self):
        self.model1 = get_example_model("cancer")
        self.data1 = self.model1.simulate(int(1e4), seed=42)

        self.model2 = BayesianNetwork(self.model1.edges(), latents={"Smoker"})
        self.model2.add_cpds(*self.model1.cpds)
        self.data2 = self.model2.simulate(int(1e4), seed=42)

    def test_get_parameters(self):
        ## All observed
        est = EM(self.model1, self.data1)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

        ## Latent variables
        est = EM(self.model2, self.data2)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]
            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

    def test_get_parameters_initial_cpds(self):
        # All observed. Specify initial CPDs.
        est = EM(self.model1, self.data1)
        smoker_initial = TabularCPD(
            "Smoker", 2, [[0.1], [0.9]], state_names={"Smoker": ["True", "False"]}
        )
        cpds = est.get_parameters(
            init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

        # With latents. Specify initial CPDs only for latent.
        est = EM(self.model2, self.data2)
        cpds = est.get_parameters(
            init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            # The latent variable doesn't converge to the true value when
            # the initial CPD is specified.
            if orig_cpd.variables[0] == "Smoker":
                self.assertTrue(
                    np.allclose(est_cpd.values, np.array([0.123, 0.877]), atol=0.01)
                )
            else:
                self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

        # With latents. Specify initial CPDs for both latents and observed.
        est = EM(self.model2, self.data2)
        xray_initial = TabularCPD(
            variable="Xray",
            variable_card=2,
            values=[[0.1, 0.8], [0.9, 0.2]],
            evidence=["Cancer"],
            evidence_card=[2],
            state_names={"Xray": ["positive", "negative"], "Cancer": ["True", "False"]},
        )
        cpds = est.get_parameters(
            init_cpds={"Smoker": smoker_initial, "Xray": xray_initial},
            seed=42,
            n_jobs=1,
            show_progress=False,
        )

        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            # The latent variable doesn't converge to the true value when
            # the initial CPD is specified.
            if orig_cpd.variables[0] == "Smoker":
                self.assertTrue(
                    np.allclose(est_cpd.values, np.array([0.123, 0.877]), atol=0.01)
                )
            elif orig_cpd.variables[0] == "Xray":
                self.assertTrue(
                    np.allclose(
                        est_cpd.values,
                        np.array([[0.799, 0.093], [0.201, 0.907]]),
                        atol=0.01,
                    )
                )
            else:
                self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

    def tearDown(self):
        del self.model1
        del self.model2
        del self.data1
        del self.data2

        get_reusable_executor().shutdown(wait=True)


class TestEMTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.model1 = get_example_model("cancer")
        self.data1 = self.model1.simulate(int(1e4), seed=42)

        self.model2 = BayesianNetwork(self.model1.edges(), latents={"Smoker"})
        self.model2.add_cpds(*self.model1.cpds)
        self.data2 = self.model2.simulate(int(1e4), seed=42)

    def test_get_parameters(self):
        est = EM(self.model1, self.data1)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

        est = EM(self.model2, self.data2)
        cpds = est.get_parameters(seed=42, n_jobs=1, show_progress=False)
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model2.get_cpds(var)

            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

    def test_get_parameters_initial_cpds(self):
        # All observed. Specify initial CPDs.
        est = EM(self.model1, self.data1)
        smoker_initial = TabularCPD(
            "Smoker", 2, [[0.1], [0.9]], state_names={"Smoker": ["True", "False"]}
        )
        cpds = est.get_parameters(
            init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

        # With latents. Specify initial CPDs only for latent.
        est = EM(self.model2, self.data2)
        cpds = est.get_parameters(
            init_cpds={"Smoker": smoker_initial}, seed=42, n_jobs=1, show_progress=False
        )
        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            # The latent variable doesn't converge to the true value when
            # the initial CPD is specified.
            if orig_cpd.variables[0] == "Smoker":
                self.assertTrue(
                    np.allclose(
                        compat_fns.to_numpy(est_cpd.values),
                        np.array([0.123, 0.877]),
                        atol=0.01,
                    )
                )
            else:
                self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

        # With latents. Specify initial CPDs for both latents and observed.
        est = EM(self.model2, self.data2)
        xray_initial = TabularCPD(
            variable="Xray",
            variable_card=2,
            values=[[0.1, 0.8], [0.9, 0.2]],
            evidence=["Cancer"],
            evidence_card=[2],
            state_names={"Xray": ["positive", "negative"], "Cancer": ["True", "False"]},
        )
        cpds = est.get_parameters(
            init_cpds={"Smoker": smoker_initial, "Xray": xray_initial},
            seed=42,
            n_jobs=1,
            show_progress=False,
        )

        for est_cpd in cpds:
            var = est_cpd.variables[0]
            orig_cpd = self.model1.get_cpds(var)
            if "Smoker" in orig_cpd.variables:
                orig_cpd.state_names["Smoker"] = [1, 0]

            # The latent variable doesn't converge to the true value when
            # the initial CPD is specified.
            if orig_cpd.variables[0] == "Smoker":
                self.assertTrue(
                    np.allclose(
                        compat_fns.to_numpy(est_cpd.values),
                        np.array([0.123, 0.877]),
                        atol=0.01,
                    )
                )
            elif orig_cpd.variables[0] == "Xray":
                self.assertTrue(
                    np.allclose(
                        compat_fns.to_numpy(est_cpd.values),
                        np.array([[0.799, 0.093], [0.201, 0.907]]),
                        atol=0.01,
                    )
                )
            else:
                self.assertTrue(orig_cpd.__eq__(est_cpd, atol=0.1))

    def tearDown(self):
        del self.model1
        del self.model2
        del self.data1
        del self.data2

        get_reusable_executor().shutdown(wait=True)

        config.set_backend("numpy")
