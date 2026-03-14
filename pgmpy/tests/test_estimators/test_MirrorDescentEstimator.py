import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.estimators import MirrorDescentEstimator
from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph, JunctionTree


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("torch not installed")
        request.node.add_marker(pytest.mark.torch)
        config.set_backend("torch")
    yield request.param
    config.set_backend("numpy")


@pytest.fixture
def simple_factor_graph():
    df = pd.DataFrame({"A": np.repeat([0, 1], 50)})
    m2 = FactorGraph()
    m2.add_node("A")
    factor = DiscreteFactor(variables=["A"], cardinality=[2], values=np.zeros(2))
    m2.add_factors(factor)
    m2.add_edges_from([("A", factor)])
    m2.check_model()

    return df, m2


def test_estimate_example_smoke_test(backend):
    data = pd.DataFrame(data={"a": [0, 0, 1, 1, 1], "b": [0, 1, 0, 1, 1]})
    model = FactorGraph()
    model.add_nodes_from(["a", "b"])
    phi1 = DiscreteFactor(["a", "b"], [2, 2], np.zeros(4))
    model.add_factors(phi1)
    model.add_edges_from([("a", phi1), ("b", phi1)])
    tree1 = MirrorDescentEstimator(model=model, data=data).estimate(
        marginals=[("a", "b")]
    )
    np.testing.assert_array_almost_equal(
        tree1.factors[0].values, [[1.0, 1.0], [1.0, 2.0]]
    )
    tree2 = MirrorDescentEstimator(model=model, data=data).estimate(marginals=[("a",)])
    assert tree2.factors[0].get_value(a=0, b=0) == tree2.factors[0].get_value(a=0, b=1)

    assert tree2.factors[0].get_value(a=1, b=0) == tree2.factors[0].get_value(a=1, b=1)

    assert pytest.approx(tree2.factors[0].get_value(a=0, b=0)) == 1.0
    assert pytest.approx(tree2.factors[0].get_value(a=1, b=0)) == 1.5


def test_mirror_descent_estimator_l2(backend, simple_factor_graph):
    df, m2 = simple_factor_graph
    mirror_descent_estimator = MirrorDescentEstimator(m2, data=df)
    tree = mirror_descent_estimator.estimate(
        marginals=[("A",)], metric="L2", iterations=2, stepsize=1
    )
    marginal = FactorDict.from_dataframe(df=df, marginals=[("A",)])[("A",)]
    diff = tree.factors[0].values.flatten() - marginal.values.flatten()
    assert pytest.approx(diff.sum()) == 0.0


def test_mirror_descent_estimator_l1(backend, simple_factor_graph):
    df, m2 = simple_factor_graph
    mirror_descent_estimator = MirrorDescentEstimator(m2, data=df)
    tree = mirror_descent_estimator.estimate(
        marginals=[("A",)], metric="L1", iterations=2, stepsize=1
    )
    marginal = FactorDict.from_dataframe(df=df, marginals=[("A",)])[("A",)]
    diff = tree.factors[0].values.flatten() - marginal.values.flatten()
    assert pytest.approx(diff.sum()) == 0.0


def test_mirror_descent_warm_start(backend):
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
    assert loss_2 < loss_1


def test_multi_clique_tree(backend):
    df = pd.DataFrame(
        data={
            "a": [1, 0, 0, 1, 1],
            "b": [0, 1, 0, 1, 1],
            "c": [1, 1, 0, 0, 1],
            "d": [1, 0, 0, 1, 1],
            "e": [0, 0, 0, 1, 1],
        }
    )
    model = JunctionTree()
    model.add_edges_from(
        [
            (("a", "b"), ("b", "c")),
            (("b", "c"), ("c", "d")),
            (("b", "c"), ("c", "e")),
        ]
    )
    assert len(model.nodes) > 1
    for node in model.nodes():
        model.add_factors(
            DiscreteFactor(
                variables=node,
                cardinality=[2 for _ in node],
                values=np.ones(tuple(2 for _ in node)),
            )
        )
    tree = MirrorDescentEstimator(model=model, data=df).estimate(marginals=model.nodes)
    empirical_marginals = FactorDict.from_dataframe(df=df, marginals=list(model.nodes))
    for clique, belief in tree.clique_beliefs.items():
        diff = empirical_marginals[clique] + -1 * belief
        np.testing.assert_allclose(diff.values, 0.0)
