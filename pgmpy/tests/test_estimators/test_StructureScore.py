import unittest

import pandas as pd

from pgmpy.estimators import (
    AIC,
    BIC,
    K2,
    AICCondGauss,
    AICGauss,
    BDeu,
    BDs,
    BICCondGauss,
    BICGauss,
    LogLikelihoodCondGauss,
    LogLikelihoodGauss,
)
from pgmpy.models import DiscreteBayesianNetwork

# Score values in the tests are compared to R package bnlearn


class TestBDeu(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = DiscreteBayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(BDeu(self.d1).score(self.m1), -9.907103407446435)
        self.assertAlmostEqual(BDeu(self.d1).score(self.m2), -9.839964104608821)
        self.assertEqual(BDeu(self.d1).score(DiscreteBayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = BDeu(self.titanic_data2, equivalent_sample_size=25)
        titanic = DiscreteBayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1892.7383393910427)
        titanic2 = DiscreteBayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


class TestBDs(unittest.TestCase):
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
        self.m1 = DiscreteBayesianNetwork([("W", "X"), ("Z", "X")])
        self.m1.add_node("Y")
        self.m2 = DiscreteBayesianNetwork([("W", "X"), ("Z", "X"), ("Y", "X")])

    def test_score(self):
        self.assertAlmostEqual(
            BDs(self.d1, equivalent_sample_size=1).score(self.m1),
            -36.82311976667139,
        )
        self.assertAlmostEqual(
            BDs(self.d1, equivalent_sample_size=1).score(self.m2),
            -45.788991276221964,
        )

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2


class TestBIC(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = DiscreteBayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(BIC(self.d1).score(self.m1), -10.698440814229318)
        self.assertAlmostEqual(BIC(self.d1).score(self.m2), -9.625886526130714)
        self.assertEqual(BIC(self.d1).score(DiscreteBayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = BIC(self.titanic_data2)
        titanic = DiscreteBayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1896.7250012840179)
        titanic2 = DiscreteBayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


class TestK2(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = DiscreteBayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(K2(self.d1).score(self.m1), -10.73813429536977)
        self.assertAlmostEqual(K2(self.d1).score(self.m2), -10.345091707260167)
        self.assertEqual(K2(self.d1).score(DiscreteBayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = K2(self.titanic_data2)
        titanic = DiscreteBayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1891.0630673606006)
        titanic2 = DiscreteBayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


class TestAIC(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = DiscreteBayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(AIC(self.d1).score(self.m1), -15.205379370888767)
        self.assertAlmostEqual(AIC(self.d1).score(self.m2), -13.68213122712422)
        self.assertEqual(AIC(self.d1).score(DiscreteBayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = AIC(self.titanic_data2)
        titanic = DiscreteBayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1875.1594513603993)
        titanic2 = DiscreteBayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


class TestLogLikeGauss(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
        self.score_fn = LogLikelihoodGauss(data)

        self.m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        self.m2 = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])

    def test_score(self):
        # score(model2network("[A]"), df[c('A')], type='loglik-g') -> -119.7228
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=[]), -119.7202, places=3
        )

        # score(model2network("[B]"), df[c('B')], type='loglik-g') -> -257.0067
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B", parents=[]), -257.0042, places=3
        )

        # score(model2network("[C]"), df[c('C')], type='loglik-g')
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C", parents=[]), -328.2361, places=3
        )

        # score(model2network("[A][B][C|A:B]"), df[c('A', 'B', 'C')], type='loglik-g') -> -455.1339
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C", parents=["A", "B"]),
            -78.3815,
            places=3,
        )

        # score(model2network("[A][B][C][D|A:B:C]"), df[c('A', 'B', 'C', 'D')], type='loglik-g') -> -732.2027
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="D", parents=["A", "B", "C"]),
            -27.1936,
            places=3,
        )

        self.assertAlmostEqual(self.score_fn.score(self.m1), -455.1058, places=3)
        self.assertAlmostEqual(self.score_fn.score(self.m2), -569.4505, places=3)


class TestAICGauss(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
        self.score_fn = AICGauss(data)

        self.m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        self.m2 = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])

    def test_score(self):
        # score(model2network("[A]"), df_cont[c('A')], type='aic-g') -> -121.7228
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=[]), -121.7202, places=3
        )

        # score(model2network("[B]"), df_cont[c('B')], type='aic-g') -> -259.0067
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B", parents=[]), -259.0042, places=3
        )

        # score(model2network("[C]"), df_cont[c('C')], type='aic-g') -> -330.2386
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C", parents=[]), -330.2361, places=3
        )

        # score(model2network("[A][B][C|A:B]"), df_cont[c('A', 'B', 'C')], type='aic-g') -> -463.1339
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C", parents=["A", "B"]),
            -82.3815,
            places=3,
        )

        # score(model2network("[A][B][C][D|A:B:C]"), df_cont[c('A', 'B', 'C', 'D')], type='aic-g')
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="D", parents=["A", "B", "C"]),
            -32.1936,
            places=3,
        )

        self.assertAlmostEqual(self.score_fn.score(self.m1), -463.1059, places=3)
        self.assertAlmostEqual(self.score_fn.score(self.m2), -577.4505, places=3)


class TestBICGauss(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
        self.score_fn = BICGauss(data)

        self.m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        self.m2 = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])

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


class TestLogLikelihoodCondGauss(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0
        )
        self.score_fn = LogLikelihoodCondGauss(data)
        self.score_fn_manual = LogLikelihoodCondGauss(data.iloc[:2, :])

    def test_score_manual(self):
        self.assertAlmostEqual(
            self.score_fn_manual.local_score(variable="A", parents=["B_cat"]),
            -1.8378,
            places=3,
        )

        self.assertAlmostEqual(
            self.score_fn_manual.local_score(variable="A", parents=["B_cat", "B"]),
            -1.8379,
            places=3,
        )

        self.assertAlmostEqual(
            self.score_fn_manual.local_score(variable="A_cat", parents=["B"]),
            2.9024,
            places=3,
        )

        self.assertAlmostEqual(
            self.score_fn_manual.local_score(variable="A_cat", parents=["B_cat", "A"]),
            0,
            places=3,
        )

        self.assertAlmostEqual(
            self.score_fn_manual.local_score(
                variable="A_cat", parents=["B", "B_cat", "C", "C_cat"]
            ),
            0,
            places=3,
        )

    def test_score_bnlearn(self):
        # Values and code from/for bnlearn.
        # score(model2network("[A]"), d[c('A')], type='loglik-g') -> -119.7228
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=[]), -119.7228, places=3
        )

        # score(model2network("[B][A|B]"), d[c('A', 'B')], type='loglik-g') -> 376.5078
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=["B"]), -119.4935, places=3
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B", parents=[]), -257.0067, places=3
        )

        # score(model2network("[B_cat][A|B_cat]"), d[c('A', 'B_cat')], type='loglik-cg') -> 200.2201
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=["B_cat"]),
            -118.5250,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B_cat", parents=[]), -81.6952, places=3
        )

        # score(model2network("[B][B_cat][A|B:B_cat]"), d[c('A', 'B', 'B_cat')], type='loglik-cg') -> 452.0991
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=["B_cat", "B"]),
            -113.2371,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B", parents=[]),
            -257.0067,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B_cat", parents=[]),
            -81.6952,
            places=3,
        )

        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), d[c('A', 'B', 'B_cat', 'C', 'C_cat')], type='loglik-cg') -> -Inf
        self.assertAlmostEqual(
            self.score_fn.local_score(
                variable="A", parents=["B_cat", "B", "C_cat", "C"]
            ),
            19.1557,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C", parents=[]),
            -328.2386,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C_cat", parents=[]),
            -130.1208,
            places=3,
        )

        # score(model2network("[A_cat]"), d[c('A_cat')], type='loglik') -> -121.527
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=[]), -121.527, places=3
        )

        #  score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=["B_cat"]),
            -117.6219,
            places=3,
        )

        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B_cat", parents=[]),
            -81.6952,
            places=3,
        )

        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=["B"]),
            -116.7104,
            places=3,
        )

        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=["B_cat", "A"]),
            -6.1599,
            places=3,
        )

        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        self.assertAlmostEqual(
            self.score_fn.local_score(
                variable="A_cat", parents=["B", "B_cat", "C", "C_cat"]
            ),
            41.9122,
            places=3,
        )


class TestAICCondGauss(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0
        )
        self.score_fn = AICCondGauss(data)

    def test_score_bnlearn(self):
        # Values and code from/for bnlearn.

        # score(model2network("[B_cat][A|B_cat]"), d[c('A', 'B_cat')], type='aic-cg') -> 208.2201
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=["B_cat"]),
            -124.525,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B_cat", parents=[]), -83.6952, places=3
        )

        # score(model2network("[B][B_cat][A|B:B_cat]"), d[c('A', 'B', 'B_cat')], type='loglik-cg') -> 465.0991
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=["B_cat", "B"]),
            -122.2372,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B", parents=[]),
            -259.0067,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B_cat", parents=[]),
            -83.6952,
            places=3,
        )

        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), d[c('A', 'B', 'B_cat', 'C', 'C_cat')], type='loglik-cg') -> -Inf
        self.assertAlmostEqual(
            self.score_fn.local_score(
                variable="A", parents=["B_cat", "B", "C_cat", "C"]
            ),
            -40.8443,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C", parents=[]),
            -330.2386,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C_cat", parents=[]),
            -134.1208,
            places=3,
        )

        # score(model2network("[A_cat]"), d[c('A_cat')], type='loglik') -> -121.527
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=[]), -124.527, places=3
        )

        #  score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=["B_cat"]),
            -126.6219,
            places=3,
        )

        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B_cat", parents=[]),
            -83.6952,
            places=3,
        )

        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=["B"]),
            -125.7104,
            places=3,
        )

        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=["B_cat", "A"]),
            -33.1599,
            places=3,
        )

        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        self.assertAlmostEqual(
            self.score_fn.local_score(
                variable="A_cat", parents=["B", "B_cat", "C", "C_cat"]
            ),
            -138.0878,
            places=3,
        )


class TestBICCondGauss(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0
        )
        self.score_fn = BICCondGauss(data)

    def test_score_bnlearn(self):
        # Values and code from/for bnlearn.

        # score(model2network("[B_cat][A|B_cat]"), d[c('A', 'B_cat')], type='bic-cg') -> 218.6408
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=["B_cat"]),
            -132.3405,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B_cat", parents=[]), -86.3004, places=3
        )

        # score(model2network("[B][B_cat][A|B:B_cat]"), d[c('A', 'B', 'B_cat')], type='loglik-cg') -> 482.0327
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A", parents=["B_cat", "B"]),
            -133.9605,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B", parents=[]),
            -261.6119,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B_cat", parents=[]),
            -86.3004,
            places=3,
        )

        # score(model2network("[B][B_cat][C][C_cat][A|B:B_cat:C:C_cat]"), d[c('A', 'B', 'B_cat', 'C', 'C_cat')], type='loglik-cg') -> -Inf
        self.assertAlmostEqual(
            self.score_fn.local_score(
                variable="A", parents=["B_cat", "B", "C_cat", "C"]
            ),
            -118.9994,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C", parents=[]),
            -332.8438,
            places=3,
        )
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="C_cat", parents=[]),
            -139.3311,
            places=3,
        )

        # score(model2network("[A_cat]"), d[c('A_cat')], type='loglik') -> -121.527
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=[]), -128.4347, places=3
        )

        #  score(model2network("[B_cat][A_cat|B_cat]"), d[c('A_cat', 'B_cat')], type='loglik') -> -199.3171
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=["B_cat"]),
            -138.3452,
            places=3,
        )

        self.assertAlmostEqual(
            self.score_fn.local_score(variable="B_cat", parents=[]),
            -86.3004,
            places=3,
        )

        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=["B"]),
            -137.4337,
            places=3,
        )

        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        self.assertAlmostEqual(
            self.score_fn.local_score(variable="A_cat", parents=["B_cat", "A"]),
            -68.3297,
            places=3,
        )

        # bnlearn doesn't work. Can not have edge from continuous to categorical variable.
        self.assertAlmostEqual(
            self.score_fn.local_score(
                variable="A_cat", parents=["B", "B_cat", "C", "C_cat"]
            ),
            -372.5531,
            places=3,
        )
