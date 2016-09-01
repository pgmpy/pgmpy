import unittest

import pandas as pd
import numpy as np

from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.models import BayesianModel


class TestBaseEstimator(unittest.TestCase):
    def setUp(self):
        self.rand_data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 2)), columns=list('AB'))
        self.rand_data['C'] = self.rand_data['B']
        self.est_rand = HillClimbSearch(self.rand_data, scoring_method=K2Score(self.rand_data))
        self.model1 = BayesianModel()
        self.model1.add_nodes_from(['A', 'B', 'C'])
        self.model2 = self.model1.copy()
        self.model2.add_edge('A', 'B')

        # link to dataset: "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv('pgmpy/tests/test_estimators/testdata/titanic_train.csv')
        self.titanic_data1 = self.titanic_data[["Survived", "Sex", "Pclass", "Age", "Embarked"]]
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]
        self.est_titanic1 = HillClimbSearch(self.titanic_data1)
        self.est_titanic2 = HillClimbSearch(self.titanic_data2)

    def test_legal_operations(self):
        model2_legal_ops = list(self.est_rand._legal_operations(self.model2))
        model2_legal_ops_ref = [(('+', ('C', 'A')), -28.15602208305154),
                                (('+', ('A', 'C')), -28.155467430966382),
                                (('+', ('C', 'B')), 7636.947544933631),
                                (('+', ('B', 'C')), 7937.805375579936),
                                (('-', ('A', 'B')), 28.155467430966382),
                                (('flip', ('A', 'B')), -0.0005546520851567038)]
        self.assertSetEqual(set([op for op, score in model2_legal_ops]),
                            set([op for op, score in model2_legal_ops_ref]))

    def test_legal_operations_titanic(self):
        est = self.est_titanic1
        start_model = BayesianModel([("Survived", "Sex"),
                                     ("Pclass", "Age"),
                                     ("Pclass", "Embarked")])

        legal_ops = est._legal_operations(start_model)
        self.assertEqual(len(list(legal_ops)), 20)

        tabu_list = [('-', ("Survived", "Sex")),
                     ('-', ("Survived", "Pclass")),
                     ('flip', ("Age", "Pclass"))]
        legal_ops_tabu = est._legal_operations(start_model, tabu_list=tabu_list)
        self.assertEqual(len(list(legal_ops_tabu)), 18)

        legal_ops_indegree = est._legal_operations(start_model, max_indegree=1)
        self.assertEqual(len(list(legal_ops_indegree)), 11)

        legal_ops_both = est._legal_operations(start_model, tabu_list=tabu_list, max_indegree=1)
        legal_ops_both_ref = [(('+', ('Embarked', 'Survived')), 10.050632580087608),
                              (('+', ('Survived', 'Pclass')), 41.88868046549101),
                              (('+', ('Age', 'Survived')), -23.635716036430836),
                              (('+', ('Pclass', 'Survived')), 41.81314459373226),
                              (('+', ('Sex', 'Pclass')), 4.772261678792802),
                              (('-', ('Pclass', 'Age')), 11.546515590731815),
                              (('-', ('Pclass', 'Embarked')), -32.171482832532774),
                              (('flip', ('Pclass', 'Embarked')), 3.3563814191281836),
                              (('flip', ('Survived', 'Sex')), 0.039737027979640516)]
        self.assertSetEqual(set(legal_ops_both), set(legal_ops_both_ref))

    def test_estimate_rand(self):
        est1 = self.est_rand.estimate()
        self.assertSetEqual(set(est1.nodes()), set(['A', 'B', 'C']))
        self.assertTrue(est1.edges() == [('B', 'C')] or est1.edges() == [('C', 'B')])

        est2 = self.est_rand.estimate(start=BayesianModel([('A', 'B'), ('A', 'C')]))
        self.assertTrue(est2.edges() == [('B', 'C')] or est2.edges() == [('C', 'B')])

    def test_estimate_titanic(self):
        self.assertSetEqual(set(self.est_titanic2.estimate().edges()),
                            set([('Survived', 'Pclass'), ('Sex', 'Pclass'), ('Sex', 'Survived')]))

    def tearDown(self):
        del self.rand_data
        del self.est_rand
        del self.model1
        del self.titanic_data
        del self.titanic_data1
        del self.titanic_data2
        del self.est_titanic1
        del self.est_titanic2
