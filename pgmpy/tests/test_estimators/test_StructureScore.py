import unittest

import pandas as pd

from pgmpy.estimators import (
    AICScore,
    AICScoreGauss,
    BDeuScore,
    BDsScore,
    BicScore,
    BicScoreGauss,
    K2Score,
)
from pgmpy.models import BayesianNetwork

# Score values in the tests are compared to R package bnlearn


class TestBDeuScore(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = BayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(BDeuScore(self.d1).score(self.m1), -9.907103407446435)
        self.assertAlmostEqual(BDeuScore(self.d1).score(self.m2), -9.839964104608821)
        self.assertEqual(BDeuScore(self.d1).score(BayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = BDeuScore(self.titanic_data2, equivalent_sample_size=25)
        titanic = BayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1892.7383393910427)
        titanic2 = BayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


class TestBDsScore(unittest.TestCase):
    def setUp(self):
        """Example taken from https://arxiv.org/pdf/1708.00689.pdf"""
        self.d1 = pd.DataFrame(
            data={
                "X": [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                "Z": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                "W": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            }
        )
        self.m1 = BayesianNetwork([("W", "X"), ("Z", "X")])
        self.m1.add_node("Y")
        self.m2 = BayesianNetwork([("W", "X"), ("Z", "X"), ("Y", "X")])

    def test_score(self):
        self.assertAlmostEqual(
            BDsScore(self.d1, equivalent_sample_size=1).score(self.m1),
            -36.82311976667139,
        )
        self.assertAlmostEqual(
            BDsScore(self.d1, equivalent_sample_size=1).score(self.m2),
            -45.788991276221964,
        )

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2


class TestBicScore(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = BayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(BicScore(self.d1).score(self.m1), -10.698440814229318)
        self.assertAlmostEqual(BicScore(self.d1).score(self.m2), -9.625886526130714)
        self.assertEqual(BicScore(self.d1).score(BayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = BicScore(self.titanic_data2)
        titanic = BayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1896.7250012840179)
        titanic2 = BayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


class TestBicScoreGauss(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
        self.score_fn = BicScoreGauss(data)

        self.m1 = BayesianNetwork([("A", "C"), ("B", "C")])
        self.m2 = BayesianNetwork([("A", "B"), ("B", "C")])

    def test_score(self):
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C", parents=["A", "B"]),
            -87.5918,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=[]), -124.3254, places=3
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B", parents=[]), -261.6093, places=3
        )

        self.assertAlmostEqual(self.score_fn.score(self.m1), -473.5265, places=3)
        self.assertAlmostEqual(self.score_fn.score(self.m2), -587.8711, places=3)


class TestK2Score(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = BayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(K2Score(self.d1).score(self.m1), -10.73813429536977)
        self.assertAlmostEqual(K2Score(self.d1).score(self.m2), -10.345091707260167)
        self.assertEqual(K2Score(self.d1).score(BayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = K2Score(self.titanic_data2)
        titanic = BayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1891.0630673606006)
        titanic2 = BayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


class TestAICScore(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = BayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(AICScore(self.d1).score(self.m1), -15.205379370888767)
        self.assertAlmostEqual(AICScore(self.d1).score(self.m2), -13.68213122712422)
        self.assertEqual(AICScore(self.d1).score(BayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = AICScore(self.titanic_data2)
        titanic = BayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1875.1594513603993)
        titanic2 = BayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


class TestAICScoreGauss(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
        self.score_fn = AICScoreGauss(data)

        self.m1 = BayesianNetwork([("A", "C"), ("B", "C")])
        self.m2 = BayesianNetwork([("A", "B"), ("B", "C")])

    def test_score(self):
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C", parents=["A", "B"]),
            -82.3815,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=[]), -121.7202, places=3
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B", parents=[]), -259.0042, places=3
        )

        self.assertAlmostEqual(self.score_fn.score(self.m1), -463.1059, places=3)
        self.assertAlmostEqual(self.score_fn.score(self.m2), -577.4505, places=3)
