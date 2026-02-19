import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.estimators import MarginalEstimator
from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import DiscreteMarkovNetwork, FactorGraph


@pytest.fixture
def setup_marginal_estimator():
    m1 = DiscreteMarkovNetwork([("A", "B"), ("B", "C")])
    df = pd.DataFrame({"A": np.repeat([0, 1], 50)})
    m2 = FactorGraph()
    m2.add_node("A")
    factor = DiscreteFactor(variables=["A"], cardinality=[2], values=np.zeros(2))
    m2.add_factors(factor)
    m2.add_edges_from([("A", factor)])
    m2.check_model()

    return {"m1": m1, "df": df, "m2": m2, "factor": factor}


class TestMarginalEstimator:
    def test_class_init(self, setup_marginal_estimator):
        m1 = setup_marginal_estimator["m1"]
        df = setup_marginal_estimator["df"]
        marginal_estimator = MarginalEstimator(m1, df)
        assert marginal_estimator

    def test_marginal_loss(self, setup_marginal_estimator):
        m2 = setup_marginal_estimator["m2"]
        df = setup_marginal_estimator["df"]
        marginal_estimator = MarginalEstimator(m2, data=df)
        factor_dict = FactorDict.from_dataframe(df=df, marginals=[("A",)])
        clique_to_marginal = marginal_estimator._clique_to_marginal(
            marginals=factor_dict,
            clique_nodes=marginal_estimator.belief_propagation.junction_tree.nodes(),
        )
        loss, _ = marginal_estimator._marginal_loss(
            marginals=marginal_estimator.belief_propagation.junction_tree.clique_beliefs,
            clique_to_marginal=clique_to_marginal,
            metric="L1",
        )
        assert loss == 100

    def test_clique_to_marginal(self, setup_marginal_estimator):
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
        assert len(clique_to_marginal[("A", "B", "C")]) == 3
        assert len(clique_to_marginal[("A",)]) == 0
        assert len(clique_to_marginal[("B",)]) == 0
        assert len(clique_to_marginal[("C",)]) == 0
        assert clique_to_marginal[("A", "B", "C")] == [
            {k: v[k]} for k, v in marginals.items()
        ]

    def test_clique_to_marginal_no_matching_cliques(self, setup_marginal_estimator):
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
        with pytest.raises(ValueError):
            MarginalEstimator._clique_to_marginal(
                marginals,
                [("D",)],
            )


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestMarginalEstimatorTorch:
    def setup_method(self):
        config.set_backend("torch")

    def teardown_method(self):
        config.set_backend("numpy")

    def test_class_init(self, setup_marginal_estimator):
        m1 = setup_marginal_estimator["m1"]
        df = setup_marginal_estimator["df"]
        marginal_estimator = MarginalEstimator(m1, df)
        assert marginal_estimator

    def test_marginal_loss(self, setup_marginal_estimator):
        m2 = setup_marginal_estimator["m2"]
        df = setup_marginal_estimator["df"]
        marginal_estimator = MarginalEstimator(m2, data=df)
        factor_dict = FactorDict.from_dataframe(df=df, marginals=[("A",)])
        clique_to_marginal = marginal_estimator._clique_to_marginal(
            marginals=factor_dict,
            clique_nodes=marginal_estimator.belief_propagation.junction_tree.nodes(),
        )
        loss, _ = marginal_estimator._marginal_loss(
            marginals=marginal_estimator.belief_propagation.junction_tree.clique_beliefs,
            clique_to_marginal=clique_to_marginal,
            metric="L1",
        )
        assert loss == 100
