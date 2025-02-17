import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators import BaseEstimator


class TestBaseEstimator(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.d2 = pd.DataFrame(
            data={
                "A": [0, np.nan, 1],
                "B": [0, 1, 0],
                "C": [1, 1, np.nan],
                "D": [np.nan, "Y", np.nan],
            }
        )

        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )

    def test_state_count(self):
        e = BaseEstimator(self.d1)
        self.assertEqual(e.state_counts("A").values.tolist(), [[2], [1]])
        self.assertEqual(
            e.state_counts("C", ["A", "B"]).values.tolist(),
            [[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        )

    def test_missing_data(self):
        e = BaseEstimator(self.d2, state_names={"C": [0, 1]})
        self.assertEqual(e.state_counts("A").values.tolist(), [[1], [1]])
        self.assertEqual(
            e.state_counts("C", parents=["A", "B"]).values.tolist(),
            [[0, 0, 0, 0], [1, 0, 0, 0]],
        )

    def tearDown(self):
        del self.d1
