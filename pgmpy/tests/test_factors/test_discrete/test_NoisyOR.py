import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.discrete import NoisyORCPD


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
