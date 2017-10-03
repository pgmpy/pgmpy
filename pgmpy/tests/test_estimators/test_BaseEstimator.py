import unittest

import pandas as pd
import numpy as np

from pgmpy.estimators import BaseEstimator


class TestBaseEstimator(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(data={'A': [0, 0, 1],
                                     'B': [0, 1, 0],
                                     'C': [1, 1, 0],
                                     'D': ['X', 'Y', 'Z']})
        self.d2 = pd.DataFrame(data={'A': [0, np.NaN, 1],
                                     'B': [0, 1, 0],
                                     'C': [1, 1, np.NaN],
                                     'D': [np.NaN, 'Y', np.NaN]})

        self.titanic_data = pd.read_csv('pgmpy/tests/test_estimators/testdata/titanic_train.csv')

    def test_state_count(self):
        e = BaseEstimator(self.d1)
        self.assertEqual(e.state_counts('A').values.tolist(), [[2], [1]])
        self.assertEqual(e.state_counts('C', ['A', 'B']).values.tolist(),
                         [[0., 0., 1., 0.], [1., 1., 0., 0.]])

    def test_missing_data(self):
        e = BaseEstimator(self.d2, state_names={'C': [0, 1]}, complete_samples_only=False)
        self.assertEqual(e.state_counts('A', complete_samples_only=True).values.tolist(), [[0], [0]])
        self.assertEqual(e.state_counts('A').values.tolist(), [[1], [1]])
        self.assertEqual(e.state_counts('C', parents=['A', 'B'], complete_samples_only=True).values.tolist(),
                         [[0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(e.state_counts('C', parents=['A', 'B']).values.tolist(),
                         [[0, 0, 0, 0], [1, 0, 0, 0]])

    def test_test_conditional_independence(self):
        data = pd.DataFrame(np.random.randint(0, 2, size=(1000, 4)), columns=list('ABCD'))
        data['E'] = data['A'] + data['B'] + data['C']
        est = BaseEstimator(data)

        self.assertGreater(est.test_conditional_independence('A', 'C')[1], 0.01)  # independent
        self.assertGreater(est.test_conditional_independence('A', 'B', 'D')[1], 0.01)  # independent
        self.assertLess(est.test_conditional_independence('A', 'B', ['D', 'E'])[1], 0.01)  # dependent

    def test_test_conditional_independence_titanic(self):
        est = BaseEstimator(self.titanic_data)

        np.testing.assert_almost_equal(est.test_conditional_independence('Embarked', 'Sex'),
                                       (13.355630515001746, 0.020264556044311655, True))
        np.testing.assert_almost_equal(est.test_conditional_independence('Pclass', 'Survived', ['Embarked']),
                                       (96.403283942888635, 4.1082315854166553e-13, True))
        np.testing.assert_almost_equal(est.test_conditional_independence('Embarked', 'Survived', ["Sex", "Pclass"]),
                                       (21.537481934494085, 0.96380273702382602, True))
        # insufficient data test commented out, because generates warning
        # self.assertEqual(est.test_conditional_independence('Sex', 'Survived', ["Age", "Embarked"]),
        #                 (235.51133052530713, 0.99999999683394869, False))

    def tearDown(self):
        del self.d1
