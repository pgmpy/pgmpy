import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators import GES
from pgmpy.models import BayesianNetwork


class TestGESDiscrete(unittest.TestCase):
    def setUp(self):
        self.rand_data = pd.DataFrame(
            np.random.randint(0, 5, size=(int(1e4), 2)), columns=list("AB")
        )
        self.rand_data["C"] = self.rand_data["B"]
        self.est_rand = GES(self.rand_data, use_cache=False)

        self.model1 = BayesianNetwork()
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
        dag = self.est_titanic2.estimate()
