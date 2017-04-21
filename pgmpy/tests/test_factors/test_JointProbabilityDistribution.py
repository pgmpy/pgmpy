import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors import JointProbabilityDistribution

class TestJointFactorDistribution(unittest.TestCase):
    def test_init(self):
        # If we give it an invalid distribution, it should raise an error
        with self.assertRaises(ValueError):
            JointProbabilityDistribution(['a'], [2], [[0.4], [0.4]])

        with self.assertRaises(ValueError):
            JointProbabilityDistribution(['a', 'b'], [2, 2], [[0.4, 0.4], [0.6, 0.6]])

        jpd = JointProbabilityDistribution(['a'], [2], [[0.4], [0.6]])
        self.assertEqual(jpd.variables, ['a'])
        np_test.assert_equal(jpd.values, np.array([0.4, 0.6]))


