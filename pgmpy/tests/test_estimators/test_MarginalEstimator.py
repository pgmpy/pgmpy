import unittest
import numpy as np
import pandas as pd

from pgmpy import config
from pgmpy.estimators import MarginalEstimator
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
        self.assertTrue(marginal_estimator)

    def test_marginal_loss(self):
        marginal_estimator = MarginalEstimator(self.m2, data=self.df)
        factor_dict = FactorDict.from_dataframe(df=self.df, marginals=[("A",)])
        clique_to_marginal = marginal_estimator._clique_to_marginal(
            marginals=factor_dict,
            clique_nodes=marginal_estimator.belief_propagation.junction_tree.nodes(),
        )
        loss, _ = marginal_estimator._marginal_loss(
            marginals=marginal_estimator.belief_propagation.junction_tree.clique_beliefs,
            clique_to_marginal=clique_to_marginal,
            metric="L1",
        )
        self.assertEqual(loss, 100)

    def test_clique_to_marginal(self):
        marginals = FactorDict(
            {
                variable: FactorDict(
                    {
                        variable: DiscreteFactor(
                            [variable], cardinality=[1], values=np.ones(1)
                        )
                    }
                )
                for variable in {"A", "B", "C"}
            }
        )
        clique_to_marginal = MarginalEstimator._clique_to_marginal(
            marginals=marginals,
            clique_nodes=[("A", "B", "C"), ("A",), ("B",), ("C",)],
        )
        self.assertEqual(len(clique_to_marginal[("A", "B", "C")]), 3)
        self.assertEqual(len(clique_to_marginal[("A",)]), 0)
        self.assertEqual(len(clique_to_marginal[("B",)]), 0)
        self.assertEqual(len(clique_to_marginal[("C",)]), 0)
        self.assertEqual(
            clique_to_marginal[("A", "B", "C")],
            [{k: v[k]} for k, v in marginals.items()],
        )

    def test_clique_to_marginal_no_matching_cliques(self):
        marginals = FactorDict(
            {
                variable: FactorDict(
                    {
                        variable: DiscreteFactor(
                            [variable], cardinality=[1], values=np.ones(1)
                        )
                    }
                )
                for variable in {"A", "B", "C"}
            }
        )
        self.assertRaises(
            ValueError,
            MarginalEstimator._clique_to_marginal,
            marginals,
            [("D",)],
        )

    def tearDown(self):
        del self.m1
        del self.m2
        del self.df


class TestMarginalEstimatorTorch(TestMarginalEstimator):
    def setUp(self):
        config.set_backend("torch")
        super().setUp()

    def tearDown(self):
        super().tearDown()
        config.set_backend("numpy")
