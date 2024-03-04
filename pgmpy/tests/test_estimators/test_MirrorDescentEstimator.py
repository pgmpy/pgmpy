import unittest
import numpy as np

import pandas as pd
from pgmpy.estimators import MirrorDescentEstimator
from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph, JunctionTree


class TestMarginalEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({"A": np.repeat([0, 1], 50)})
        self.m2 = FactorGraph()
        self.m2.add_node("A")
        self.factor = DiscreteFactor(
            variables=["A"], cardinality=[2], values=np.zeros(2)
        )
        self.m2.add_factors(self.factor)
        self.m2.add_edges_from([("A", self.factor)])
        self.m2.check_model()

    def estimate_example_smoke_test(self):
        data = pd.DataFrame(data={"a": [0, 0, 1, 1, 1], "b": [0, 1, 0, 1, 1]})
        model = FactorGraph()
        model.add_nodes_from(["a", "b"])
        phi1 = DiscreteFactor(["a", "b"], [2, 2], np.zeros(4))
        model.add_factors(phi1)
        model.add_edges_from([("a", phi1), ("b", phi1)])
        tree1 = MirrorDescentEstimator(model=model, data=data).estimate(
            marginals=[("a", "b")]
        )
        self.assertTrue(np.all(tree1.factors[0].values, np.array([1.0, 1.0, 1.0, 2.0])))
        tree2 = MirrorDescentEstimator(model=model, data=data).estimate(
            marginals=[("a",)]
        )
        self.assertTrue(np.all(tree2.factors[0].values, np.array([1.0, 1.0, 1.5, 1.5])))

    def test_mirror_descent_estimator_l2(self):
        mirror_descent_estimator = MirrorDescentEstimator(self.m2, data=self.df)
        tree = mirror_descent_estimator.estimate(
            marginals=[("A",)], metric="L2", iterations=2, stepsize=1
        )
        marginal = FactorDict.from_dataframe(df=self.df, marginals=[("A",)])[("A",)]
        diff = tree.factors[0].values.flatten() - marginal.values.flatten()
        self.assertAlmostEqual(diff.sum(), 0.0)

    def test_mirror_descent_estimator_l1(self):
        mirror_descent_estimator = MirrorDescentEstimator(self.m2, data=self.df)
        tree = mirror_descent_estimator.estimate(
            marginals=[("A",)], metric="L1", iterations=2, stepsize=1
        )
        marginal = FactorDict.from_dataframe(df=self.df, marginals=[("A",)])[("A",)]
        diff = tree.factors[0].values.flatten() - marginal.values.flatten()
        self.assertAlmostEqual(diff.sum(), 0.0)

    def test_mirror_descent_warm_start(self):
        df = pd.DataFrame({"A": np.repeat([0, 1], 50), "B": np.repeat([1, 0], 50)})
        model = JunctionTree()
        model.add_node(node=["A", "B"])
        model.add_factors(
            DiscreteFactor(variables=["A", "B"], cardinality=[2, 2], values=np.zeros(4))
        )
        mirror_descent_estimator = MirrorDescentEstimator(model, data=df)
        clique_to_marginal = mirror_descent_estimator._clique_to_marginal(
            marginals=FactorDict.from_dataframe(
                df,
                marginals=[("A", "B")],
            ),
            clique_nodes=[("A", "B")],
        )
        loss_1, _ = mirror_descent_estimator._marginal_loss(
            marginals=mirror_descent_estimator.estimate(
                marginals=[("A", "B")], metric="L2", iterations=5, stepsize=0.1
            ).clique_beliefs,
            clique_to_marginal=clique_to_marginal,
            metric="L2",
        )
        loss_2, _ = mirror_descent_estimator._marginal_loss(
            marginals=mirror_descent_estimator.estimate(
                marginals=[("A", "B")], metric="L2", iterations=2, stepsize=0.1
            ).clique_beliefs,
            clique_to_marginal=clique_to_marginal,
            metric="L2",
        )
        self.assertTrue(loss_2 < loss_1)
