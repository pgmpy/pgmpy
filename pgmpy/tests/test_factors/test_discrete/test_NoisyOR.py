import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.discrete import NoisyORCPD, TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


class TestNoisyORInit(unittest.TestCase):
    def test_class_init(self):
        cpd = NoisyORCPD(
            variable="Y", prob_values=[0.7, 0.6, 0.8], evidence=["X1", "X2", "X3"]
        )
        self.assertEqual(cpd.variables, ["Y", "X1", "X2", "X3"])
        np_test.assert_array_equal(cpd.cardinality, np.array([2, 2, 2, 2]))
        np_test.assert_array_equal(
            cpd.get_values().round(3),
            np.array(
                [
                    [0.976, 0.88, 0.94, 0.7, 0.92, 0.6, 0.8, 0.0],
                    [0.024, 0.12, 0.06, 0.3, 0.08, 0.4, 0.2, 1.0],
                ]
            ),
        )

        self.assertRaises(
            ValueError, NoisyORCPD, variable="X", prob_values=[0.7, 0.8], evidence=["Y"]
        )
        self.assertRaises(
            ValueError,
            NoisyORCPD,
            variable="X",
            prob_values=[0.7, 1.1],
            evidence=["Y", "Z"],
        )

    def test_inference(self):
        model = DiscreteBayesianNetwork([("A", "B"), ("C", "B"), ("B", "D")])

        cpd_a = TabularCPD("A", 2, [[0.2], [0.8]], state_names={"A": ["True", "False"]})
        cpd_c = TabularCPD("C", 2, [[0.1], [0.9]], state_names={"C": ["True", "False"]})
        cpd_b = NoisyORCPD("B", [0.4, 0.3], evidence=["A", "C"])
        cpd_d = NoisyORCPD("D", [0.8], evidence=["B"])

        model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)

        from pgmpy.inference import VariableElimination

        infer = VariableElimination(model)
        np.testing.assert_allclose(infer.query(["A"]).values, [0.2, 0.8])
        np.testing.assert_allclose(infer.query(["C"]).values, [0.1, 0.9])
        np.testing.assert_allclose(infer.query(["B"]).values, [0.1076, 0.8924])

        np.testing.assert_allclose(
            infer.query(["D"], evidence={"B": "True"}).values, [0.8, 0.2]
        )
        np.testing.assert_allclose(
            infer.query(["D"], evidence={"B": "False"}).values, [0, 1.0]
        )
