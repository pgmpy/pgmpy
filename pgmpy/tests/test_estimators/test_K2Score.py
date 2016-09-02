import unittest

import pandas as pd

from pgmpy.models import BayesianModel
from pgmpy.estimators import K2Score


class TestK2Score(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0], 'D': ['X', 'Y', 'Z']})
        self.m1 = BayesianModel([('A', 'C'), ('B', 'C'), ('D', 'B')])
        self.m2 = BayesianModel([('C', 'A'), ('C', 'B'), ('A', 'D')])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv('pgmpy/tests/test_estimators/testdata/titanic_train.csv')
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(K2Score(self.d1).score(self.m1), -10.73813429536977)
        self.assertEqual(K2Score(self.d1).score(BayesianModel()), 0)

    def test_score_titanic(self):
        scorer = K2Score(self.titanic_data2)
        titanic = BayesianModel([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1891.0630673606006)
        titanic2 = BayesianModel([("Pclass", "Sex"), ])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2
