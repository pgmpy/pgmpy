import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators import GES, ExpertKnowledge
from pgmpy.models import DiscreteBayesianNetwork


class TestGESDiscrete(unittest.TestCase):
    def setUp(self):
        self.rand_data = pd.DataFrame(
            np.random.randint(0, 5, size=(int(1e4), 2)), columns=list("AB")
        )
        self.rand_data["C"] = self.rand_data["B"]
        self.est_rand = GES(self.rand_data, use_cache=False)

        self.model1 = DiscreteBayesianNetwork()
        self.model1.add_nodes_from(["A", "B", "C"])
        self.model1_possible_edges = set(
            [(u, v) for u in self.model1.nodes() for v in self.model1.nodes()]
        )

        self.model2 = self.model1.copy()
        self.model2.add_edge("A", "B")
        self.model2_possible_edges = set(
            [(u, v) for u in self.model2.nodes() for v in self.model2.nodes()]
        )

        # link to dataset: "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/titanic_train.csv"
        )
        self.titanic_data1 = self.titanic_data[
            ["Survived", "Sex", "Pclass", "Age", "Embarked"]
        ]
        self.est_titanic1 = GES(self.titanic_data1, use_cache=False)

        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]
        self.est_titanic2 = GES(self.titanic_data2, use_cache=False)

    def test_estimate(self):
        dag = self.est_rand.estimate()
        dag = self.est_titanic1.estimate()

        temporal_knowledge = ExpertKnowledge(
            temporal_order=[["Pclass", "Sex"], ["Survived"]]
        )
        dag2 = self.est_titanic2.estimate(
            expert_knowledge=temporal_knowledge, scoring_method="k2"
        )

        self.assertSetEqual(
            set([("Sex", "Survived"), ("Sex", "Pclass"), ("Pclass", "Survived")]),
            set(dag2.edges()),
        )


class TestGESGauss(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv", index_col=0
        )

    def test_estimate(self):
        est = GES(self.data)
        for score in ["aic-g", "bic-g"]:
            dag = est.estimate(scoring_method=score, debug=True)


class TestGESMixed(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0
        )
        self.data["A_cat"] = self.data.A_cat.astype("category")
        self.data["B_cat"] = self.data.B_cat.astype("category")
        self.data["C_cat"] = self.data.C_cat.astype("category")
        self.data["B_int"] = self.data.B_int.astype("category")

    def test_estimate(self):
        est = GES(self.data)
        dag = est.estimate(scoring_method="ll-cg")
