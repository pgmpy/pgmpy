import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pgmpy.data import Data


@unittest.skip
class TestData(unittest.TestCase):
    def setUp(self):
        self.random_values = np.random.randint(low=0, high=2, size=(1000000, 5))
        self.random_df = pd.DataFrame(
            self.random_values, columns=["A", "B", "C", "D", "E"]
        )
        self.random_data = Data(self.random_df)

        self.dep_df = self.random_df.copy()
        self.dep_df["dep_col"] = self.dep_df.E * 2
        self.dep_data = Data(self.dep_df)

    def test_init(self):
        data = Data(self.random_values, variables=["A", "B", "C", "D", "E"])
        self.assertListEqual(data.variables, ["A", "B", "C", "D", "E"])
        np_test.assert_array_equal(data.data.values, self.random_values)

    def test_test_independence(self):
        chi, p_value, dof = self.random_data.test_independence(var1="A", var2="B")
        self.assertGreater(chi, 0.1)
        self.assertGreater(p_value, 0.1)
        self.assertEqual(dof, 1)

        chi, p_value, dof = self.random_data.test_independence(
            var1="A", var2="B", conditioned_vars=["C", "D"]
        )
        self.assertGreater(chi, 0.1)
        self.assertGreater(p_value, 0.1)
        self.assertEqual(dof, 4)

    def test_cov_matrix(self):
        cov_matrix_ind = self.random_data.cov_matrix()
        np_test.assert_allclose(
            cov_matrix_ind.values,
            np.array(
                [
                    [0.25, 0, 0, 0, 0],
                    [0, 0.25, 0, 0, 0],
                    [0, 0, 0.25, 0, 0],
                    [0, 0, 0, 0.25, 0],
                    [0, 0, 0, 0, 0.25],
                ]
            ),
            atol=0.01,
        )

        cov_matrix_dep = self.dep_data.cov_matrix()
        np_test.assert_allclose(
            cov_matrix_dep.values,
            np.array(
                [
                    [0.25, 0, 0, 0, 0, 0],
                    [0, 0.25, 0, 0, 0, 0],
                    [0, 0, 0.25, 0, 0, 0],
                    [0, 0, 0, 0.25, 0, 0],
                    [0, 0, 0, 0, 0.25, 0.5],
                    [0, 0, 0, 0, 0.5, 1],
                ]
            ),
            atol=0.01,
        )
