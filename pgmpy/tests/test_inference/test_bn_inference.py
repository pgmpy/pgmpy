import unittest

import numpy as np
import pandas as pd

from pgmpy.models import BayesianNetwork
from pgmpy.metrics import BayesianModelProbability
from pgmpy.factors.discrete import TabularCPD


class TestBnInference(unittest.TestCase):
    def setUp(self) -> None:
        # construct a tree graph structure
        model = BayesianNetwork(
            [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]
        )

        # add CPD to each edge
        cpd_a = TabularCPD("A", 2, [[0.4], [0.6]])
        cpd_b = TabularCPD(
            "B",
            3,
            [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
            evidence=["A"],
            evidence_card=[2],
        )
        cpd_c = TabularCPD(
            "C", 2, [[0.3, 0.4], [0.7, 0.6]], evidence=["A"], evidence_card=[2]
        )
        cpd_d = TabularCPD(
            "D",
            3,
            [[0.5, 0.3, 0.1], [0.4, 0.4, 0.8], [0.1, 0.3, 0.1]],
            evidence=["B"],
            evidence_card=[3],
        )
        cpd_e = TabularCPD(
            "E",
            2,
            [[0.3, 0.5, 0.2], [0.7, 0.5, 0.8]],
            evidence=["B"],
            evidence_card=[3],
        )
        cpd_f = TabularCPD(
            "F",
            3,
            [[0.3, 0.6], [0.5, 0.2], [0.2, 0.2]],
            evidence=["C"],
            evidence_card=[2],
        )
        model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d, cpd_e, cpd_f)

        """ transposed conditional probabilities
        C {(0,): array([0.3, 0.7]), (1,): array([0.4, 0.6])}
        F {(0,): array([0.3, 0.5, 0.2]), (1,): array([0.6, 0.2, 0.2])}
        B {(0,): array([0.6, 0.3, 0.1]), (1,): array([0.2, 0.5, 0.3])}
        E {(0,): array([0.3, 0.7]), (1,): array([0.5, 0.5]), (2,): array([0.2, 0.8])}
        D {(0,): array([0.5, 0.4, 0.1]), (1,): array([0.3, 0.4, 0.3]), (2,): array([0.1, 0.8, 0.1])}
        """
        self.model = model
        self.inference = BayesianModelProbability(model)

    def test_bn_probability(self):
        ordering = ["A", "C", "F", "B", "E", "D"]

        x0 = np.array([[0, 0, 0, 0, 0, 0]])
        x1 = np.array([[1, 1, 1, 1, 1, 1]])
        x2 = np.array([[1, 1, 2, 2, 1, 2]])
        X = np.concatenate([x0, x1, x2], axis=0)

        p0 = 0.4 * 0.3 * 0.3 * 0.6 * 0.3 * 0.5
        p1 = 0.6 * 0.6 * 0.2 * 0.5 * 0.5 * 0.4
        p2 = 0.6 * 0.6 * 0.2 * 0.3 * 0.8 * 0.1

        logp = self.inference.log_probability(x0, ordering)
        p = np.exp(logp)
        np.testing.assert_almost_equal(p[0], p0)

        logp = self.inference.log_probability(x1, ordering)
        p = np.exp(logp)
        np.testing.assert_almost_equal(p[0], p1)

        logp = self.inference.log_probability(x2, ordering)
        p = np.exp(logp)
        np.testing.assert_almost_equal(p[0], p2)

        logp = self.inference.log_probability(X, ordering)
        p = np.exp(logp)
        np.testing.assert_array_almost_equal(p, [p0, p1, p2])

        data = pd.DataFrame(
            np.array(
                [
                    [1, 0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 2],
                    [1, 0, 0, 1, 0, 1],
                    [0, 1, 2, 0, 1, 1],
                    [1, 1, 1, 0, 1, 0],
                    [1, 1, 2, 1, 1, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 2, 0, 0, 1],
                    [0, 1, 0, 0, 1, 2],
                    [1, 0, 1, 2, 0, 1],
                ]
            ),
            columns=["A", "C", "F", "B", "E", "D"],
        )
        logp = self.inference.log_probability(data, ordering)
        p = np.exp(logp)

        p_check = np.array(
            [
                0.012,
                0.0054,
                0.0072,
                0.009408,
                0.00504,
                0.0054,
                0.03528,
                0.001728,
                0.007056,
                0.00576,
            ]
        )
        np.testing.assert_array_almost_equal(p, p_check)

        llsum = self.inference.score(data, ordering)
        np.testing.assert_almost_equal(llsum, -49.57170403869862)

        # use topological ordering of model to interpret data
        logp = self.inference.log_probability(data)
        p = np.exp(logp)
        np.testing.assert_array_almost_equal(p, p_check)

        llsum = self.inference.score(data)
        np.testing.assert_almost_equal(llsum, -49.57170403869862)
