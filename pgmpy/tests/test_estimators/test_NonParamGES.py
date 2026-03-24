import unittest
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators.NonParamGES import (
    conditional_log_likelihood, complexity_penalty, NonParamGESScore, NonParamGES
)

class TestNonParamGES(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a tiny continuous dataset: X1 -> X2
        self.X1 = np.random.normal(0, 1, 100)
        self.X2 = self.X1 + np.random.normal(0, 0.5, 100)
        self.data = pd.DataFrame({'X1': self.X1, 'X2': self.X2})
        
    def test_conditional_log_likelihood(self):
        # For X2, conditioning on X1 should increase the likelihood
        # because they are highly correlated.
        ll_cond = conditional_log_likelihood(self.data, 'X2', ['X1'], cv=2)
        ll_marg = conditional_log_likelihood(self.data, 'X2', [], cv=2)
        
        self.assertGreater(ll_cond, ll_marg)
        
    def test_complexity_penalty(self):
        pen1 = complexity_penalty(1, 100)
        pen2 = complexity_penalty(2, 100)
        self.assertEqual(pen2, 2 * pen1)
        self.assertTrue(np.isclose(pen1, np.log(100) / 100))
        
    def test_score_decomposability(self):
        score = NonParamGESScore(self.data, cv=2)
        G = DAG()
        G.add_nodes_from(['X1', 'X2'])
        
        G_prime = DAG()
        G_prime.add_nodes_from(['X1', 'X2'])
        G_prime.add_edge('X1', 'X2')
        
        res = score.test(self.data, G_prime, G, lambda_threshold=1.0)
        self.assertIsInstance(res, bool)
        
    def test_estimator(self):
        est = NonParamGES(self.data, cv=2)
        model = est.estimate()
        # NonParamGES should find the edge between X1 and X2 (could be either direction 
        # depending on likelihood values since both are bivariate normal).
        self.assertEqual(len(model.edges()), 1)

if __name__ == '__main__':
    unittest.main()
