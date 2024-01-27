import unittest
import numpy as np

import pandas as pd
from pgmpy.estimators import MarginalEstimator, MirrorDescentEstimator
from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph, MarkovNetwork


class TestMarginalEstimator(unittest.TestCase):
    def setUp(self):
        self.m1 = MarkovNetwork([("A", "B"), ("B", "C")])
        self.df = pd.DataFrame({"A": np.repeat([0, 1], 50)})
        self.m2 = FactorGraph()
        self.m2.add_node("A")
        self.factor = DiscreteFactor(
            variables=["A"], cardinality=[2], values=np.zeros(2)
        )
        self.m2.add_factors(self.factor)
        self.m2.add_edges_from([("A", self.factor)])
        self.m2.check_model()

    def test_class_init(self):
        marginal_estimator = MarginalEstimator(
            MarkovNetwork([("A", "B"), ("B", "C")]), pd.DataFrame()
        )
        self.assert_(marginal_estimator)

    def test_marginal_loss(self):
        marginal_estimator = MarginalEstimator(self.m2, data=self.df)
        factor_dict = FactorDict.from_dataframe(df=self.df, marginals=[("A",)])
        clique_to_marginal = marginal_estimator._clique_to_marginal(
            marginals=factor_dict
        )
        loss, _ = marginal_estimator._marginal_loss(
            marginals=marginal_estimator.belief_propagation.junction_tree.factor_dict,
            clique_to_marginal=clique_to_marginal,
            metric="L1",
        )
        self.assertEqual(loss, 100)

    def test_mirror_descent_estimator(self):
        mirror_descent_estimator = MirrorDescentEstimator(self.m2, data=self.df)
        tree = mirror_descent_estimator.estimate(
            marginals=[("A",)], metric="L2", iterations=2, alpha=1
        )
        marginal = FactorDict.from_dataframe(df=self.df, marginals=[("A",)])[("A",)]
        diff = tree.factors[0].values.flatten() - marginal.values.flatten()
        self.assertEqual(diff.sum(), 0.0)
