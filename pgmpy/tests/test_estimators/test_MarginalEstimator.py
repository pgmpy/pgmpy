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
def models():
    m1 = DiscreteMarkovNetwork([("A", "B"), ("B", "C")])
    df = pd.DataFrame({"A": np.repeat([0, 1], 50)})
    m2 = FactorGraph()
    m2.add_node("A")
    factor = DiscreteFactor(variables=["A"], cardinality=[2], values=np.zeros(2))
    m2.add_factors(factor)
    m2.add_edges_from([("A", factor)])
    m2.check_model()
    return {"m1": m1, "m2": m2, "df": df, "factor": factor}


@pytest.fixture
def torch_models():
    config.set_backend("torch")
    m1 = DiscreteMarkovNetwork([("A", "B"), ("B", "C")])
    df = pd.DataFrame({"A": np.repeat([0, 1], 50)})
    m2 = FactorGraph()
    m2.add_node("A")
    factor = DiscreteFactor(variables=["A"], cardinality=[2], values=np.zeros(2))
    m2.add_factors(factor)
    m2.add_edges_from([("A", factor)])
    m2.check_model()
    yield {"m1": m1, "m2": m2, "df": df, "factor": factor}
    config.set_backend("numpy")


requires_torch = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)


def test_class_init():
    marginal_estimator = MarginalEstimator(
        DiscreteMarkovNetwork([("A", "B"), ("B", "C")]), pd.DataFrame()
    )
    assert marginal_estimator


def test_marginal_loss(models):
    marginal_estimator = MarginalEstimator(models["m2"], data=models["df"])
    factor_dict = FactorDict.from_dataframe(df=models["df"], marginals=[("A",)])
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


def test_clique_to_marginal():
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


def test_clique_to_marginal_no_matching_cliques():
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
        MarginalEstimator._clique_to_marginal(marginals, [("D",)])


@requires_torch
def test_torch_class_init():
    marginal_estimator = MarginalEstimator(
        DiscreteMarkovNetwork([("A", "B"), ("B", "C")]), pd.DataFrame()
    )
    assert marginal_estimator


@requires_torch
def test_torch_marginal_loss(torch_models):
    marginal_estimator = MarginalEstimator(torch_models["m2"], data=torch_models["df"])
    factor_dict = FactorDict.from_dataframe(df=torch_models["df"], marginals=[("A",)])
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


@requires_torch
def test_torch_clique_to_marginal():
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


@requires_torch
def test_torch_clique_to_marginal_no_matching_cliques():
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
        MarginalEstimator._clique_to_marginal(marginals, [("D",)])
