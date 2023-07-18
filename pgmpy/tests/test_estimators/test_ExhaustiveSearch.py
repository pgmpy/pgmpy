import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators import BDeuScore, BicScore, ExhaustiveSearch


class TestBaseEstimator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.rand_data = pd.DataFrame(
            np.random.randint(0, 5, size=(5000, 2)), columns=list("AB")
        )
        self.rand_data["C"] = self.rand_data["B"]
        self.est_rand = ExhaustiveSearch(self.rand_data)
        self.est_rand_bdeu = ExhaustiveSearch(
            self.rand_data, scoring_method=BDeuScore(self.rand_data)
        )
        self.est_rand_bic = ExhaustiveSearch(
            self.rand_data, scoring_method=BicScore(self.rand_data)
        )

        # link to dataset: "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]
        self.est_titanic = ExhaustiveSearch(self.titanic_data2)

    def test_all_dags(self):
        self.assertEqual(len(list(self.est_rand.all_dags(["A", "B", "C", "D"]))), 543)
        # self.assertEqual(len(list(self.est_rand.all_dags(nodes=range(5)))), 29281)  # takes ~30s

        abc_dags = set(
            map(tuple, [sorted(dag.edges()) for dag in self.est_rand.all_dags()])
        )
        abc_dags_ref = set(
            [
                (("A", "B"), ("C", "A"), ("C", "B")),
                (("A", "C"), ("B", "C")),
                (("B", "A"), ("B", "C")),
                (("C", "B"),),
                (("A", "C"), ("B", "A")),
                (("B", "C"), ("C", "A")),
                (("A", "B"), ("B", "C")),
                (("A", "C"), ("B", "A"), ("B", "C")),
                (("A", "B"),),
                (("A", "B"), ("C", "A")),
                (("B", "A"), ("C", "A"), ("C", "B")),
                (("A", "C"), ("C", "B")),
                (("A", "B"), ("A", "C"), ("C", "B")),
                (("B", "A"), ("C", "B")),
                (("A", "B"), ("A", "C")),
                (("C", "A"), ("C", "B")),
                (("A", "B"), ("A", "C"), ("B", "C")),
                (("C", "A"),),
                (("B", "A"), ("B", "C"), ("C", "A")),
                (("B", "A"),),
                (("A", "B"), ("C", "B")),
                (),
                (("B", "A"), ("C", "A")),
                (("A", "C"),),
                (("B", "C"),),
            ]
        )
        self.assertSetEqual(abc_dags, abc_dags_ref)

    def test_estimate_rand(self):
        est = self.est_rand.estimate()
        self.assertSetEqual(set(est.nodes()), set(["A", "B", "C"]))
        self.assertEqual(set(est.edges()), {("B", "A"), ("B", "C"), ("C", "A")})

        est_bdeu = self.est_rand.estimate()
        self.assertEqual(set(est.edges()), {("B", "A"), ("B", "C"), ("C", "A")})

        est_bic = self.est_rand.estimate()
        self.assertEqual(set(est_bic.edges()), {("B", "A"), ("B", "C"), ("C", "A")})

    def test_estimate_titanic(self):
        e1 = self.est_titanic.estimate()
        self.assertSetEqual(
            set(e1.edges()),
            set([("Survived", "Pclass"), ("Sex", "Pclass"), ("Sex", "Survived")]),
        )

    def test_all_scores(self):
        scores = self.est_titanic.all_scores()
        scores_ref = [
            (-2072.9132364404695, []),
            (-2069.071694164769, [("Pclass", "Sex")]),
            (-2069.0144197068785, [("Sex", "Pclass")]),
            (-2025.869489762676, [("Survived", "Pclass")]),
            (-2025.8559302273054, [("Pclass", "Survived")]),
            (-2022.0279474869753, [("Pclass", "Sex"), ("Survived", "Pclass")]),
            (-2022.0143879516047, [("Pclass", "Sex"), ("Pclass", "Survived")]),
            (-2021.9571134937144, [("Pclass", "Survived"), ("Sex", "Pclass")]),
            (-2017.5258065853768, [("Sex", "Pclass"), ("Survived", "Pclass")]),
            (-1941.3075053892837, [("Survived", "Sex")]),
            (-1941.2720031713893, [("Sex", "Survived")]),
            (-1937.4304608956886, [("Pclass", "Sex"), ("Sex", "Survived")]),
            (-1937.4086886556927, [("Sex", "Pclass"), ("Survived", "Sex")]),
            (-1937.3731864377983, [("Sex", "Pclass"), ("Sex", "Survived")]),
            (-1934.1344850608882, [("Pclass", "Sex"), ("Survived", "Sex")]),
            (-1894.2637587114903, [("Survived", "Pclass"), ("Survived", "Sex")]),
            (-1894.2501991761198, [("Pclass", "Survived"), ("Survived", "Sex")]),
            (-1894.2282564935958, [("Sex", "Survived"), ("Survived", "Pclass")]),
            (-1891.0630673606006, [("Pclass", "Survived"), ("Sex", "Survived")]),
            (
                -1887.2215250849,
                [("Pclass", "Sex"), ("Pclass", "Survived"), ("Sex", "Survived")],
            ),
            (
                -1887.1642506270096,
                [("Pclass", "Survived"), ("Sex", "Pclass"), ("Sex", "Survived")],
            ),
            (
                -1887.0907383830947,
                [("Pclass", "Sex"), ("Survived", "Pclass"), ("Survived", "Sex")],
            ),
            (
                -1887.0771788477243,
                [("Pclass", "Sex"), ("Pclass", "Survived"), ("Survived", "Sex")],
            ),
            (
                -1885.9200755341915,
                [("Sex", "Pclass"), ("Survived", "Pclass"), ("Survived", "Sex")],
            ),
            (
                -1885.884573316297,
                [("Sex", "Pclass"), ("Sex", "Survived"), ("Survived", "Pclass")],
            ),
        ]

        self.assertEqual(
            [sorted(model.edges()) for score, model in scores],
            [edges for score, edges in scores_ref],
        )
        # use assertAlmostEqual point wise to avoid rounding issues
        map(
            lambda x, y: self.assertAlmostEqual(x, y),
            [score for score, model in scores],
            [score for score, edges in scores_ref],
        )

    def tearDown(self):
        del self.rand_data
        del self.est_rand
        del self.est_rand_bdeu
        del self.est_rand_bic
        del self.titanic_data
        del self.est_titanic
