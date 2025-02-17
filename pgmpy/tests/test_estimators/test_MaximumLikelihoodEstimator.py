import unittest

import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor

from pgmpy import config
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork, JunctionTree


class TestMLE(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C")])
        self.model_latents = BayesianNetwork([("A", "C"), ("B", "C")], latents=["C"])
        self.data_latents = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0]})
        self.m2 = JunctionTree()
        self.m2.add_nodes_from([("A", "B")])
        self.m3 = JunctionTree()
        self.m3.add_edges_from([(("A", "C"), ("B", "C"))])

        self.d1 = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
        self.d2 = pd.DataFrame(
            data={
                "A": [0, np.nan, 1],
                "B": [0, 1, 0],
                "C": [1, 1, np.nan],
                "D": [np.nan, "Y", np.nan],
            }
        )
        # Use Example from ML Machine Learning - A Probabilistic Perspective
        # Section 19.5.7.1.
        self.d3 = pd.DataFrame(
            data={
                "A": [0] * 43 + [0] * 9 + [1] * 44 + [1] * 4,
                "B": [0] * 43 + [1] * 9 + [0] * 44 + [1] * 4,
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
        self.potentials1 = FactorDict.from_dataframe(
            df=self.d3, marginals=self.m2.nodes
        )
        self.m2.clique_beliefs = self.potentials1

        self.potentials2 = FactorDict.from_dataframe(
            df=self.d1, marginals=self.m3.nodes
        )
        self.m3.clique_beliefs = self.potentials2

        self.mle1 = MaximumLikelihoodEstimator(self.m1, self.d1)
        self.mle2 = MaximumLikelihoodEstimator(model=self.m2, data=self.d3)
        self.mle3 = MaximumLikelihoodEstimator(model=self.m3, data=self.d1)

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
        self.assertEqual(len(self.mle1.get_parameters(n_jobs=1)), 3)

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
        self.assertEqual(len(mle.get_parameters(n_jobs=1)), 3)

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
        self.assertEqual(len(mle.get_parameters(n_jobs=1)), 3)

    def test_missing_data(self):
        e1 = MaximumLikelihoodEstimator(self.m1, self.d2, state_names={"C": [0, 1]})
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
        self.assertEqual(len(e1.get_parameters(n_jobs=1)), 3)

    def test_estimate_potentials_smoke_test(self):
        joint = self.mle3.estimate_potentials().product()
        self.assertEqual(
            joint.marginalize(variables=["B"], inplace=False),
            self.potentials2[("A", "C")].normalize(inplace=False),
        )
        self.assertEqual(
            joint.marginalize(variables=["A"], inplace=False),
            self.potentials2[("B", "C")].normalize(inplace=False),
        )

    def test_partition_function(self):
        model = self.m3.copy()
        model.clique_beliefs = self.mle3.estimate_potentials()
        self.assertEqual(model.get_partition_function(), 1.0)

    def test_estimate_potentials(self):
        self.assertEqual(
            self.mle2.estimate_potentials()[("A", "B")],
            self.potentials1[("A", "B")].normalize(inplace=False),
        )

    def tearDown(self):
        del self.m1
        del self.d1
        del self.d2

        get_reusable_executor().shutdown(wait=True)


class TestMLETorch(TestMLE):
    def setUp(self):
        config.set_backend("torch")
        super().setUp()

    def tearDown(self):
        super().tearDown()
        config.set_backend("numpy")
