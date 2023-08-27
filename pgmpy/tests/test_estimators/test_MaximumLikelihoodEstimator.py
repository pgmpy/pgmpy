import unittest

import numpy as np
import pandas as pd

from pgmpy import config
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


class TestMLE(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C")])
        self.model_latents = BayesianNetwork([("A", "C"), ("B", "C")], latents=["C"])
        self.data_latents = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0]})

        self.d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
        self.d2 = pd.DataFrame(
            data={
                "A": [0, np.NaN, 1],
                "B": [0, 1, 0],
                "C": [1, 1, np.NaN],
                "D": [np.NaN, "Y", np.NaN],
            }
        )
        self.cpds = [
            TabularCPD("A", 2, [[2.0 / 3], [1.0 / 3]]),
            TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
            TabularCPD(
                "C",
                2,
                [[0.0, 0.0, 1.0, 0.5], [1.0, 1.0, 0.0, 0.5]],
                evidence=["A", "B"],
                evidence_card=[2, 2],
            ),
        ]
        self.mle1 = MaximumLikelihoodEstimator(self.m1, self.d1)

    def test_error_latent_model(self):
        self.assertRaises(
            ValueError,
            MaximumLikelihoodEstimator,
            self.model_latents,
            self.data_latents,
        )

    def test_get_parameters_incomplete_data(self):
        self.assertEqual(self.mle1.estimate_cpd("A"), self.cpds[0])
        self.assertEqual(self.mle1.estimate_cpd("B"), self.cpds[1])
        self.assertEqual(self.mle1.estimate_cpd("C"), self.cpds[2])
        self.assertEqual(len(self.mle1.get_parameters()), 3)

    def test_estimate_cpd(self):
        self.assertEqual(self.mle1.estimate_cpd("A"), self.cpds[0])
        self.assertEqual(self.mle1.estimate_cpd("B"), self.cpds[1])
        self.assertEqual(self.mle1.estimate_cpd("C"), self.cpds[2])

    def test_state_names1(self):
        m = BayesianNetwork([("A", "B")])
        d = pd.DataFrame(data={"A": [2, 3, 8, 8, 8], "B": ["X", "O", "X", "O", "X"]})
        cpd_b = TabularCPD(
            "B",
            2,
            [[0, 1, 1.0 / 3], [1, 0, 2.0 / 3]],
            evidence=["A"],
            evidence_card=[3],
            state_names={"A": [2, 3, 8], "B": ["O", "X"]},
        )
        mle2 = MaximumLikelihoodEstimator(m, d)
        self.assertEqual(mle2.estimate_cpd("B"), cpd_b)

    def test_state_names2(self):
        m = BayesianNetwork([("Light?", "Color"), ("Fruit", "Color")])
        d = pd.DataFrame(
            data={
                "Fruit": ["Apple", "Apple", "Apple", "Banana", "Banana"],
                "Light?": [True, True, False, False, True],
                "Color": ["red", "green", "black", "black", "yellow"],
            }
        )
        color_cpd = TabularCPD(
            "Color",
            4,
            [[1, 0, 1, 0], [0, 0.5, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 1]],
            evidence=["Fruit", "Light?"],
            evidence_card=[2, 2],
            state_names={
                "Color": ["black", "green", "red", "yellow"],
                "Light?": [False, True],
                "Fruit": ["Apple", "Banana"],
            },
        )
        mle2 = MaximumLikelihoodEstimator(m, d)
        self.assertEqual(mle2.estimate_cpd("Color"), color_cpd)

    def test_class_init(self):
        mle = MaximumLikelihoodEstimator(
            self.m1, self.d1, state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]}
        )
        self.assertEqual(mle.estimate_cpd("A"), self.cpds[0])
        self.assertEqual(mle.estimate_cpd("B"), self.cpds[1])
        self.assertEqual(mle.estimate_cpd("C"), self.cpds[2])
        self.assertEqual(len(mle.get_parameters()), 3)

    def test_nonoccurring_values(self):
        mle = MaximumLikelihoodEstimator(
            self.m1,
            self.d1,
            state_names={"A": [0, 1, 23], "B": [0, 1], "C": [0, 42, 1], 1: [2]},
        )
        cpds = [
            TabularCPD(
                "A", 3, [[2.0 / 3], [1.0 / 3], [0]], state_names={"A": [0, 1, 23]}
            ),
            TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]], state_names={"B": [0, 1]}),
            TabularCPD(
                "C",
                3,
                [
                    [0.0, 0.0, 1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                    [0.0, 0.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                    [1.0, 1.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                ],
                evidence=["A", "B"],
                evidence_card=[3, 2],
                state_names={"A": [0, 1, 23], "B": [0, 1], "C": [0, 42, 1]},
            ),
        ]
        self.assertEqual(mle.estimate_cpd("A"), cpds[0])
        self.assertEqual(mle.estimate_cpd("B"), cpds[1])
        self.assertEqual(mle.estimate_cpd("C"), cpds[2])
        self.assertEqual(len(mle.get_parameters()), 3)

    def test_missing_data(self):
        e1 = MaximumLikelihoodEstimator(
            self.m1, self.d2, state_names={"C": [0, 1]}, complete_samples_only=False
        )
        cpds1 = [
            TabularCPD("A", 2, [[0.5], [0.5]]),
            TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
            TabularCPD(
                "C",
                2,
                [[0, 0.5, 0.5, 0.5], [1, 0.5, 0.5, 0.5]],
                evidence=["A", "B"],
                evidence_card=[2, 2],
            ),
        ]
        self.assertEqual(e1.estimate_cpd("A"), cpds1[0])
        self.assertEqual(e1.estimate_cpd("B"), cpds1[1])
        self.assertEqual(e1.estimate_cpd("C"), cpds1[2])
        self.assertEqual(len(e1.get_parameters()), 3)

        e2 = MaximumLikelihoodEstimator(
            self.m1, self.d2, state_names={"C": [0, 1]}, complete_samples_only=True
        )
        cpds2 = [
            TabularCPD("A", 2, [[0.5], [0.5]]),
            TabularCPD("B", 2, [[0.5], [0.5]]),
            TabularCPD(
                "C",
                2,
                [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
                evidence=["A", "B"],
                evidence_card=[2, 2],
            ),
        ]
        self.assertEqual(e2.estimate_cpd("A"), cpds2[0])
        self.assertEqual(e2.estimate_cpd("B"), cpds2[1])
        self.assertEqual(e2.estimate_cpd("C"), cpds2[2])
        self.assertEqual(len(e2.get_parameters()), 3)

    def tearDown(self):
        del self.m1
        del self.d1
        del self.d2


class TestMLETorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.m1 = BayesianNetwork([("A", "C"), ("B", "C")])
        self.model_latents = BayesianNetwork([("A", "C"), ("B", "C")], latents=["C"])
        self.data_latents = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0]})

        self.d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
        self.d2 = pd.DataFrame(
            data={
                "A": [0, np.NaN, 1],
                "B": [0, 1, 0],
                "C": [1, 1, np.NaN],
                "D": [np.NaN, "Y", np.NaN],
            }
        )
        self.cpds = [
            TabularCPD("A", 2, [[2.0 / 3], [1.0 / 3]]),
            TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
            TabularCPD(
                "C",
                2,
                [[0.0, 0.0, 1.0, 0.5], [1.0, 1.0, 0.0, 0.5]],
                evidence=["A", "B"],
                evidence_card=[2, 2],
            ),
        ]
        self.mle1 = MaximumLikelihoodEstimator(self.m1, self.d1)

    def test_error_latent_model(self):
        self.assertRaises(
            ValueError,
            MaximumLikelihoodEstimator,
            self.model_latents,
            self.data_latents,
        )

    def test_get_parameters_incomplete_data(self):
        self.assertEqual(self.mle1.estimate_cpd("A"), self.cpds[0])
        self.assertEqual(self.mle1.estimate_cpd("B"), self.cpds[1])
        self.assertEqual(self.mle1.estimate_cpd("C"), self.cpds[2])
        self.assertEqual(len(self.mle1.get_parameters()), 3)

    def test_estimate_cpd(self):
        self.assertEqual(self.mle1.estimate_cpd("A"), self.cpds[0])
        self.assertEqual(self.mle1.estimate_cpd("B"), self.cpds[1])
        self.assertEqual(self.mle1.estimate_cpd("C"), self.cpds[2])

    def test_state_names1(self):
        m = BayesianNetwork([("A", "B")])
        d = pd.DataFrame(data={"A": [2, 3, 8, 8, 8], "B": ["X", "O", "X", "O", "X"]})
        cpd_b = TabularCPD(
            "B",
            2,
            [[0, 1, 1.0 / 3], [1, 0, 2.0 / 3]],
            evidence=["A"],
            evidence_card=[3],
            state_names={"A": [2, 3, 8], "B": ["O", "X"]},
        )
        mle2 = MaximumLikelihoodEstimator(m, d)
        self.assertEqual(mle2.estimate_cpd("B"), cpd_b)

    def test_state_names2(self):
        m = BayesianNetwork([("Light?", "Color"), ("Fruit", "Color")])
        d = pd.DataFrame(
            data={
                "Fruit": ["Apple", "Apple", "Apple", "Banana", "Banana"],
                "Light?": [True, True, False, False, True],
                "Color": ["red", "green", "black", "black", "yellow"],
            }
        )
        color_cpd = TabularCPD(
            "Color",
            4,
            [[1, 0, 1, 0], [0, 0.5, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 1]],
            evidence=["Fruit", "Light?"],
            evidence_card=[2, 2],
            state_names={
                "Color": ["black", "green", "red", "yellow"],
                "Light?": [False, True],
                "Fruit": ["Apple", "Banana"],
            },
        )
        mle2 = MaximumLikelihoodEstimator(m, d)
        self.assertEqual(mle2.estimate_cpd("Color"), color_cpd)

    def test_class_init(self):
        mle = MaximumLikelihoodEstimator(
            self.m1, self.d1, state_names={"A": [0, 1], "B": [0, 1], "C": [0, 1]}
        )
        self.assertEqual(mle.estimate_cpd("A"), self.cpds[0])
        self.assertEqual(mle.estimate_cpd("B"), self.cpds[1])
        self.assertEqual(mle.estimate_cpd("C"), self.cpds[2])
        self.assertEqual(len(mle.get_parameters()), 3)

    def test_nonoccurring_values(self):
        mle = MaximumLikelihoodEstimator(
            self.m1,
            self.d1,
            state_names={"A": [0, 1, 23], "B": [0, 1], "C": [0, 42, 1], 1: [2]},
        )
        cpds = [
            TabularCPD(
                "A", 3, [[2.0 / 3], [1.0 / 3], [0]], state_names={"A": [0, 1, 23]}
            ),
            TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]], state_names={"B": [0, 1]}),
            TabularCPD(
                "C",
                3,
                [
                    [0.0, 0.0, 1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                    [0.0, 0.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                    [1.0, 1.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
                ],
                evidence=["A", "B"],
                evidence_card=[3, 2],
                state_names={"A": [0, 1, 23], "B": [0, 1], "C": [0, 42, 1]},
            ),
        ]
        self.assertEqual(mle.estimate_cpd("A"), cpds[0])
        self.assertEqual(mle.estimate_cpd("B"), cpds[1])
        self.assertEqual(mle.estimate_cpd("C"), cpds[2])
        self.assertEqual(len(mle.get_parameters()), 3)

    def test_missing_data(self):
        e1 = MaximumLikelihoodEstimator(
            self.m1, self.d2, state_names={"C": [0, 1]}, complete_samples_only=False
        )
        cpds1 = [
            TabularCPD("A", 2, [[0.5], [0.5]]),
            TabularCPD("B", 2, [[2.0 / 3], [1.0 / 3]]),
            TabularCPD(
                "C",
                2,
                [[0, 0.5, 0.5, 0.5], [1, 0.5, 0.5, 0.5]],
                evidence=["A", "B"],
                evidence_card=[2, 2],
            ),
        ]
        self.assertEqual(e1.estimate_cpd("A"), cpds1[0])
        self.assertEqual(e1.estimate_cpd("B"), cpds1[1])
        self.assertEqual(e1.estimate_cpd("C"), cpds1[2])
        self.assertEqual(len(e1.get_parameters()), 3)

        e2 = MaximumLikelihoodEstimator(
            self.m1, self.d2, state_names={"C": [0, 1]}, complete_samples_only=True
        )
        cpds2 = [
            TabularCPD("A", 2, [[0.5], [0.5]]),
            TabularCPD("B", 2, [[0.5], [0.5]]),
            TabularCPD(
                "C",
                2,
                [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
                evidence=["A", "B"],
                evidence_card=[2, 2],
            ),
        ]
        self.assertEqual(e2.estimate_cpd("A"), cpds2[0])
        self.assertEqual(e2.estimate_cpd("B"), cpds2[1])
        self.assertEqual(e2.estimate_cpd("C"), cpds2[2])
        self.assertEqual(len(e2.get_parameters()), 3)

    def tearDown(self):
        del self.m1
        del self.d1
        del self.d2

        config.set_backend("numpy")
