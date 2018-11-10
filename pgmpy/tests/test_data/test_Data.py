import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pgmpy.data import Data


class TestData(unittest.TestCase):
    def setUp(self):
        self.random_values = np.random.randint(low=0, high=2, size=(10000, 5))
        self.random_data = Data(pd.DataFrame(self.random_values,
                                             columns=['A', 'B', 'C', 'D', 'E']))

    def test_init(self):
        data = Data(self.random_values, variables=['A', 'B', 'C', 'D', 'E'])
        self.assertListEqual(data.variables, ['A', 'B', 'C', 'D', 'E'])
        np_test.assert_array_equal(data.data.values, self.random_values)

    def test_test_independence(self):
        chi, p_value, dof = self.random_data.test_independence(var1='A', var2='B')
        self.assertGreater(chi, 0.1)
        self.assertGreater(p_value, 0.1)
        self.assertEqual(dof, 1)

        chi, p_value, dof = self.random_data.test_independence(var1='A', var2='B',
                                                               conditioned_vars=['C', 'D'])
        self.assertGreater(chi, 0.1)
        self.assertGreater(p_value, 0.1)
        self.assertEqual(dof, 4)

