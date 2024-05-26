import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestLGBNMethods(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        self.cpd1 = LinearGaussianCPD("x1", [1], 4)
        self.cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        self.cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])

    def test_cpds_simple(self):
        self.assertEqual("x1", self.cpd1.variable)
        self.assertEqual(4, self.cpd1.variance)
        self.assertEqual([1], self.cpd1.mean)

        self.model.add_cpds(self.cpd1)
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)
        self.assertEqual(cpd.variance, self.cpd1.variance)
        self.assertEqual(cpd.mean, self.cpd1.mean)

    def test_add_cpds(self):
        self.model.add_cpds(self.cpd1)
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)
        self.assertEqual(cpd.variance, self.cpd1.variance)

        self.model.add_cpds(self.cpd2)
        cpd = self.model.get_cpds("x2")
        self.assertEqual(cpd.variable, self.cpd2.variable)
        self.assertEqual(cpd.variance, self.cpd2.variance)
        self.assertEqual(cpd.evidence, self.cpd2.evidence)

        self.model.add_cpds(self.cpd3)
        cpd = self.model.get_cpds("x3")
        self.assertEqual(cpd.variable, self.cpd3.variable)
        self.assertEqual(cpd.variance, self.cpd3.variance)
        self.assertEqual(cpd.evidence, self.cpd3.evidence)

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

    @unittest.skip("TODO")
    def test_to_joint_gaussian(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        jgd = self.model.to_joint_gaussian()
        self.assertEqual(jgd.variables, ["x1", "x2", "x3"])
        np_test.assert_array_equal(jgd.mean, np.array([[1.0], [-4.5], [8.5]]))
        np_test.assert_array_equal(
            jgd.covariance,
            np.array([[4.0, 2.0, -2.0], [2.0, 5.0, -5.0], [-2.0, -5.0, 8.0]]),
        )

    def test_check_model(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        self.assertEqual(self.model.check_model(), True)

        self.model.add_edge("x1", "x4")
        cpd4 = LinearGaussianCPD("x4", [4, -1], 3, ["x2"])
        self.model.add_cpds(cpd4)

        self.assertRaises(ValueError, self.model.check_model)

    def test_not_implemented_methods(self):
        self.assertRaises(ValueError, self.model.get_cardinality, "x1")
        self.assertRaises(NotImplementedError, self.model.fit, [[1, 2, 3], [1, 5, 6]])
        self.assertRaises(
            NotImplementedError, self.model.predict, [[1, 2, 3], [1, 5, 6]]
        )
        self.assertRaises(NotImplementedError, self.model.to_markov_model)
        self.assertRaises(
            NotImplementedError, self.model.is_imap, [[1, 2, 3], [1, 5, 6]]
        )

    def test_simulate(self):
        model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        cpd_x1 = LinearGaussianCPD("x1", [1], 4)
        cpd_x2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        cpd_x3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        model.add_cpds(cpd_x1, cpd_x2, cpd_x3)

        df_cont = model.simulate(n=10000)

        x1 = 1 + np.random.normal(0, 2, 10000)
        x2 = -5 + 0.5 * x1 + np.random.normal(0, 2, 10000)
        x3 = 4 + -1 * x2 + np.random.normal(0, np.sqrt(3), 10000)
        df_equ = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
