import unittest

import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor

from pgmpy import config
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


class TestBayesianEstimator(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C")])
        self.model_latent = BayesianNetwork([("A", "C"), ("B", "C")], latents=["C"])
        self.d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
        self.d2 = pd.DataFrame(
            data={
                "A": [0, 0, 1, 0, 2, 0, 2, 1, 0, 2],
                "B": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
                "C": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        self.est1 = BayesianEstimator(self.m1, self.d1)
        self.est2 = BayesianEstimator(
            self.m1, self.d1, state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]}
        )
        self.est3 = BayesianEstimator(self.m1, self.d2)

    def test_error_latent_model(self):
        self.assertRaises(ValueError, BayesianEstimator, self.model_latent, self.d1)

    def test_estimate_cpd_dirichlet(self):
        cpd_A = self.est1.estimate_cpd(
            "A", prior_type="dirichlet", pseudo_counts=[[0], [1]]
        )
        cpd_A_exp = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"A": [0, 1]},
        )
        self.assertEqual(cpd_A, cpd_A_exp)

        # also test passing pseudo_counts as np.array
        pseudo_counts = np.array([[0], [1]])
        cpd_A = self.est1.estimate_cpd(
            "A", prior_type="dirichlet", pseudo_counts=pseudo_counts
        )
        self.assertEqual(cpd_A, cpd_A_exp)

        cpd_B = self.est1.estimate_cpd(
            "B", prior_type="dirichlet", pseudo_counts=[[9], [3]]
        )
        cpd_B_exp = TabularCPD(
            "B", 2, [[11.0 / 15], [4.0 / 15]], state_names={"B": [0, 1]}
        )
        self.assertEqual(cpd_B, cpd_B_exp)

        cpd_C = self.est1.estimate_cpd(
            "C",
            prior_type="dirichlet",
            pseudo_counts=[[0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]],
        )
        cpd_C_exp = TabularCPD(
            "C",
            2,
            [[0.2, 0.2, 0.7, 0.4], [0.8, 0.8, 0.3, 0.6]],
            evidence=["A", "B"],
            evidence_card=[2, 2],
            state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
        )
        self.assertEqual(cpd_C, cpd_C_exp)

    def test_estimate_cpd_improper_prior(self):
        cpd_C = self.est1.estimate_cpd(
            "C", prior_type="dirichlet", pseudo_counts=[[0, 0, 0, 0], [0, 0, 0, 0]]
        )
        cpd_C_correct = TabularCPD(
            "C",
            2,
            [[0.0, 0.0, 1.0, np.nan], [1.0, 1.0, 0.0, np.nan]],
            evidence=["A", "B"],
            evidence_card=[2, 2],
            state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
        )
        # manual comparison because np.nan != np.nan
        self.assertTrue(
            (
                (cpd_C.values == cpd_C_correct.values)
                | np.isnan(cpd_C.values) & np.isnan(cpd_C_correct.values)
            ).all()
        )

    def test_estimate_cpd_shortcuts(self):
        cpd_C1 = self.est2.estimate_cpd(
            "C", prior_type="BDeu", equivalent_sample_size=9
        )
        cpd_C1_correct = TabularCPD(
            "C",
            3,
            [
                [0.2, 0.2, 0.6, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                [0.6, 0.6, 0.2, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                [0.2, 0.2, 0.2, 1.0 / 3, 1.0 / 3, 1.0 / 3],
            ],
            evidence=["A", "B"],
            evidence_card=[3, 2],
            state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]},
        )
        self.assertEqual(cpd_C1, cpd_C1_correct)

        cpd_C2 = self.est3.estimate_cpd("C", prior_type="K2")
        cpd_C2_correct = TabularCPD(
            "C",
            2,
            [
                [0.5, 0.6, 1.0 / 3, 2.0 / 3, 0.75, 2.0 / 3],
                [0.5, 0.4, 2.0 / 3, 1.0 / 3, 0.25, 1.0 / 3],
            ],
            evidence=["A", "B"],
            evidence_card=[3, 2],
            state_names={"A": [0, 1, 2], "B": ["X", "Y"], "C": [0, 1]},
        )
        self.assertEqual(cpd_C2, cpd_C2_correct)

    def test_get_parameters(self):
        cpds = set(
            [
                self.est3.estimate_cpd("A"),
                self.est3.estimate_cpd("B"),
                self.est3.estimate_cpd("C"),
            ]
        )
        self.assertSetEqual(set(self.est3.get_parameters(n_jobs=1)), cpds)

    def test_get_parameters2(self):
        pseudo_counts = {
            "A": [[1], [2], [3]],
            "B": [[4], [5]],
            "C": [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7]],
        }
        cpds = set(
            [
                self.est3.estimate_cpd(
                    "A", prior_type="dirichlet", pseudo_counts=pseudo_counts["A"]
                ),
                self.est3.estimate_cpd(
                    "B", prior_type="dirichlet", pseudo_counts=pseudo_counts["B"]
                ),
                self.est3.estimate_cpd(
                    "C", prior_type="dirichlet", pseudo_counts=pseudo_counts["C"]
                ),
            ]
        )
        self.assertSetEqual(
            set(
                self.est3.get_parameters(
                    prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1
                )
            ),
            cpds,
        )

    def test_get_parameters3(self):
        pseudo_counts = 0.1
        cpds = set(
            [
                self.est3.estimate_cpd(
                    "A", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
                self.est3.estimate_cpd(
                    "B", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
                self.est3.estimate_cpd(
                    "C", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
            ]
        )
        self.assertSetEqual(
            set(
                self.est3.get_parameters(
                    prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1
                )
            ),
            cpds,
        )

    def tearDown(self):
        del self.m1
        del self.d1
        del self.d2
        del self.est1
        del self.est2
        get_reusable_executor().shutdown(wait=True)


class TestBayesianEstimatorTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.m1 = BayesianNetwork([("A", "C"), ("B", "C")])
        self.model_latent = BayesianNetwork([("A", "C"), ("B", "C")], latents=["C"])
        self.d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
        self.d2 = pd.DataFrame(
            data={
                "A": [0, 0, 1, 0, 2, 0, 2, 1, 0, 2],
                "B": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
                "C": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        self.est1 = BayesianEstimator(self.m1, self.d1)
        self.est2 = BayesianEstimator(
            self.m1, self.d1, state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]}
        )
        self.est3 = BayesianEstimator(self.m1, self.d2)

    def test_error_latent_model(self):
        self.assertRaises(ValueError, BayesianEstimator, self.model_latent, self.d1)

    def test_estimate_cpd_dirichlet(self):
        cpd_A = self.est1.estimate_cpd(
            "A", prior_type="dirichlet", pseudo_counts=[[0], [1]]
        )
        cpd_A_exp = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"A": [0, 1]},
        )
        self.assertEqual(cpd_A, cpd_A_exp)

        # also test passing pseudo_counts as np.array
        pseudo_counts = np.array([[0], [1]])
        cpd_A = self.est1.estimate_cpd(
            "A", prior_type="dirichlet", pseudo_counts=pseudo_counts
        )
        self.assertEqual(cpd_A, cpd_A_exp)

        cpd_B = self.est1.estimate_cpd(
            "B", prior_type="dirichlet", pseudo_counts=[[9], [3]]
        )
        cpd_B_exp = TabularCPD(
            "B", 2, [[11.0 / 15], [4.0 / 15]], state_names={"B": [0, 1]}
        )
        self.assertEqual(cpd_B, cpd_B_exp)

        cpd_C = self.est1.estimate_cpd(
            "C",
            prior_type="dirichlet",
            pseudo_counts=[[0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]],
        )
        cpd_C_exp = TabularCPD(
            "C",
            2,
            [[0.2, 0.2, 0.7, 0.4], [0.8, 0.8, 0.3, 0.6]],
            evidence=["A", "B"],
            evidence_card=[2, 2],
            state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
        )
        self.assertEqual(cpd_C, cpd_C_exp)

    def test_estimate_cpd_improper_prior(self):
        cpd_C = self.est1.estimate_cpd(
            "C", prior_type="dirichlet", pseudo_counts=[[0, 0, 0, 0], [0, 0, 0, 0]]
        )
        cpd_C_correct = TabularCPD(
            "C",
            2,
            [[0.0, 0.0, 1.0, np.nan], [1.0, 1.0, 0.0, np.nan]],
            evidence=["A", "B"],
            evidence_card=[2, 2],
            state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
        )
        # manual comparison because np.nan != np.nan
        self.assertTrue(
            (
                (cpd_C.values == cpd_C_correct.values)
                | config.get_compute_backend().isnan(cpd_C.values)
                & config.get_compute_backend().isnan(cpd_C_correct.values)
            ).all()
        )

    def test_estimate_cpd_shortcuts(self):
        cpd_C1 = self.est2.estimate_cpd(
            "C", prior_type="BDeu", equivalent_sample_size=9
        )
        cpd_C1_correct = TabularCPD(
            "C",
            3,
            [
                [0.2, 0.2, 0.6, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                [0.6, 0.6, 0.2, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                [0.2, 0.2, 0.2, 1.0 / 3, 1.0 / 3, 1.0 / 3],
            ],
            evidence=["A", "B"],
            evidence_card=[3, 2],
            state_names={"A": [0, 1, 2], "B": [0, 1], "C": [0, 1, 23]},
        )
        self.assertEqual(cpd_C1, cpd_C1_correct)

        cpd_C2 = self.est3.estimate_cpd("C", prior_type="K2")
        cpd_C2_correct = TabularCPD(
            "C",
            2,
            [
                [0.5, 0.6, 1.0 / 3, 2.0 / 3, 0.75, 2.0 / 3],
                [0.5, 0.4, 2.0 / 3, 1.0 / 3, 0.25, 1.0 / 3],
            ],
            evidence=["A", "B"],
            evidence_card=[3, 2],
            state_names={"A": [0, 1, 2], "B": ["X", "Y"], "C": [0, 1]},
        )
        self.assertEqual(cpd_C2, cpd_C2_correct)

    def test_get_parameters(self):
        cpds = [
            self.est3.estimate_cpd("A"),
            self.est3.estimate_cpd("B"),
            self.est3.estimate_cpd("C"),
        ]
        all_cpds = self.est3.get_parameters(n_jobs=1)
        self.assertListEqual(
            sorted(cpds, key=lambda t: t.variables[0]),
            sorted(all_cpds, key=lambda t: t.variables[0]),
        )

    def test_get_parameters2(self):
        pseudo_counts = {
            "A": [[1], [2], [3]],
            "B": [[4], [5]],
            "C": [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7]],
        }
        cpds = set(
            [
                self.est3.estimate_cpd(
                    "A", prior_type="dirichlet", pseudo_counts=pseudo_counts["A"]
                ),
                self.est3.estimate_cpd(
                    "B", prior_type="dirichlet", pseudo_counts=pseudo_counts["B"]
                ),
                self.est3.estimate_cpd(
                    "C", prior_type="dirichlet", pseudo_counts=pseudo_counts["C"]
                ),
            ]
        )
        all_cpds = self.est3.get_parameters(
            prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1
        )
        self.assertListEqual(
            sorted(cpds, key=lambda t: t.variables[0]),
            sorted(all_cpds, key=lambda t: t.variables[0]),
        )

    def test_get_parameters3(self):
        pseudo_counts = 0.1
        cpds = set(
            [
                self.est3.estimate_cpd(
                    "A", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
                self.est3.estimate_cpd(
                    "B", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
                self.est3.estimate_cpd(
                    "C", prior_type="dirichlet", pseudo_counts=pseudo_counts
                ),
            ]
        )
        all_cpds = self.est3.get_parameters(
            prior_type="dirichlet", pseudo_counts=pseudo_counts, n_jobs=1
        )
        self.assertListEqual(
            sorted(cpds, key=lambda t: t.variables[0]),
            sorted(all_cpds, key=lambda t: t.variables[0]),
        )

    def tearDown(self):
        del self.m1
        del self.d1
        del self.d2
        del self.est1
        del self.est2
        get_reusable_executor().shutdown(wait=True)

        config.set_backend("numpy")
