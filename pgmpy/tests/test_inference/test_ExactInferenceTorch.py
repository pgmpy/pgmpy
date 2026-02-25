import itertools

import numpy as np
import numpy.testing as np_test
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork, JunctionTree
from pgmpy.utils import compat_fns


@pytest.fixture
def torch_backend():
    config.set_backend("torch")
    yield
    config.set_backend("numpy")


@pytest.fixture
def bayesian_model(torch_backend):
    bayesian_model = DiscreteBayesianNetwork(
        [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")]
    )
    cpd_a = TabularCPD("A", 2, values=[[0.2], [0.8]])
    cpd_r = TabularCPD("R", 2, values=[[0.4], [0.6]])
    cpd_j = TabularCPD(
        "J",
        2,
        values=[[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
        evidence=["A", "R"],
        evidence_card=[2, 2],
    )
    cpd_q = TabularCPD(
        "Q", 2, values=[[0.9, 0.2], [0.1, 0.8]], evidence=["J"], evidence_card=[2]
    )
    cpd_l = TabularCPD(
        "L",
        2,
        values=[[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        evidence=["J", "G"],
        evidence_card=[2, 2],
    )
    cpd_g = TabularCPD("G", 2, values=[[0.6], [0.4]])
    bayesian_model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
    return bayesian_model


@pytest.fixture
def bayesian_inference(bayesian_model):
    bayesian_inference = VariableElimination(bayesian_model)
    return bayesian_inference


@pytest.fixture
def test_duplicated_factors_markov_inference(torch_backend):
    markov_model = DiscreteMarkovNetwork([("A", "B"), ("A", "C")])
    f1 = DiscreteFactor(variables=["A", "B"], cardinality=[2, 2], values=np.eye(2) * 2)
    f2 = DiscreteFactor(variables=["A", "C"], cardinality=[2, 2], values=np.eye(2) * 2)
    markov_model.add_factors(f1, f2)
    markov_inference = VariableElimination(markov_model)
    return markov_inference


@pytest.fixture
def snow_network_torch_model(torch_backend):
    model = DiscreteBayesianNetwork(
        [
            ("Snow", "Risk"),
            ("Snow", "Traffic"),
            ("Traffic", "Late"),
            ("Risk", "Late"),
        ]
    )

    cpd_snow = TabularCPD(
        "Snow", 2, [[0.4], [0.6]], state_names={"Snow": ["yes", "no"]}
    )
    cpd_risk = TabularCPD(
        "Risk",
        2,
        [[0.8, 0.4], [0.2, 0.6]],
        evidence=["Snow"],
        evidence_card=[2],
        state_names={"Snow": ["yes", "no"], "Risk": ["yes", "no"]},
    )
    cpd_traffic = TabularCPD(
        "Traffic",
        2,
        [[0.4, 0.65], [0.6, 0.35]],
        evidence=["Snow"],
        evidence_card=[2],
        state_names={"Traffic": ["normal", "slow"], "Snow": ["yes", "no"]},
    )
    cpd_late = TabularCPD(
        "Late",
        2,
        [[0.45, 0.85, 0.1, 0.7], [0.55, 0.15, 0.90, 0.30]],
        evidence=["Risk", "Traffic"],
        evidence_card=[2, 2],
        state_names={
            "Late": ["yes", "no"],
            "Traffic": ["normal", "slow"],
            "Risk": ["yes", "no"],
        },
    )
    model.add_cpds(cpd_snow, cpd_risk, cpd_traffic, cpd_late)
    return model


@pytest.fixture
def variable_elimination_markov_inference(torch_backend):
    # It is just a moralised version of the above Bayesian network so all the results are same. Only factors
    # are under consideration for inference so this should be fine.
    markov_model = DiscreteMarkovNetwork(
        [
            ("A", "J"),
            ("R", "J"),
            ("J", "Q"),
            ("J", "L"),
            ("G", "L"),
            ("A", "R"),
            ("J", "G"),
        ]
    )

    factor_a = TabularCPD("A", 2, values=[[0.2], [0.8]]).to_factor()
    factor_r = TabularCPD("R", 2, values=[[0.4], [0.6]]).to_factor()
    factor_j = TabularCPD(
        "J",
        2,
        values=[[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
        evidence=["A", "R"],
        evidence_card=[2, 2],
    ).to_factor()
    factor_q = TabularCPD(
        "Q", 2, values=[[0.9, 0.2], [0.1, 0.8]], evidence=["J"], evidence_card=[2]
    ).to_factor()
    factor_l = TabularCPD(
        "L",
        2,
        values=[[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        evidence=["J", "G"],
        evidence_card=[2, 2],
    ).to_factor()
    factor_g = TabularCPD("G", 2, [[0.6], [0.4]]).to_factor()

    markov_model.add_factors(factor_a, factor_r, factor_j, factor_q, factor_l, factor_g)
    markov_inference = VariableElimination(markov_model)
    return markov_inference


@pytest.fixture
def junction_tree(torch_backend):
    junction_tree = JunctionTree([(("A", "B"), ("B", "C")), (("B", "C"), ("C", "D"))])
    phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
    phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
    phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))
    junction_tree.add_factors(phi1, phi2, phi3)
    return junction_tree


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestVariableEliminationTorch:
    # All the values that are used for comparison in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self, bayesian_inference):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = bayesian_inference.query(
                ["J"], elimination_order=order, show_progress=False
            )
            assert query_result == DiscreteFactor(
                variables=["J"], cardinality=[2], values=[0.416, 0.584]
            )

    def test_query_multiple_variable(self, bayesian_inference):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = bayesian_inference.query(
                ["Q", "J"], elimination_order=order, show_progress=False
            )
            assert query_result == DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.3744, 0.0416], [0.1168, 0.4672]]),
            )

    def test_query_single_variable_with_evidence(self, bayesian_inference):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = bayesian_inference.query(
                variables=["J"],
                evidence={"A": 0, "R": 1},
                elimination_order=order,
                show_progress=False,
            )
            assert query_result == DiscreteFactor(
                variables=["J"], cardinality=[2], values=[0.6, 0.4]
            )

    def test_query_multiple_variable_with_evidence(self, bayesian_inference):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = bayesian_inference.query(
                variables=["J", "Q"],
                evidence={"A": 0, "R": 0, "G": 0, "L": 1},
                elimination_order=order,
                show_progress=False,
            )
            assert query_result == DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.73636364, 0.08181818], [0.03636364, 0.14545455]]),
            )

    def test_query_multiple_times(self, bayesian_inference):
        # This just tests that the models are not getting modified while querying them
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = bayesian_inference.query(
                ["J"], elimination_order=order, show_progress=False
            )
            query_result = bayesian_inference.query(
                ["J"], elimination_order=order, show_progress=False
            )
            assert query_result == DiscreteFactor(
                variables=["J"], cardinality=[2], values=np.array([0.416, 0.584])
            )
            query_result = bayesian_inference.query(
                ["Q", "J"], elimination_order=order, show_progress=False
            )
            query_result = bayesian_inference.query(
                ["Q", "J"], elimination_order=order, show_progress=False
            )
            assert query_result == DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.3744, 0.0416], [0.1168, 0.4672]]),
            )

            query_result = bayesian_inference.query(
                variables=["J"],
                evidence={"A": 0, "R": 1},
                elimination_order=order,
                show_progress=False,
            )
            query_result = bayesian_inference.query(
                variables=["J"],
                evidence={"A": 0, "R": 1},
                elimination_order=order,
                show_progress=False,
            )
            assert query_result == DiscreteFactor(
                variables=["J"], cardinality=[2], values=[0.6, 0.4]
            )

            query_result = bayesian_inference.query(
                variables=["J", "Q"],
                evidence={"A": 0, "R": 0, "G": 0, "L": 1},
                elimination_order=order,
                show_progress=False,
            )
            query_result = bayesian_inference.query(
                variables=["J", "Q"],
                evidence={"A": 0, "R": 0, "G": 0, "L": 1},
                elimination_order=order,
                show_progress=False,
            )
            assert query_result == DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.73636364, 0.08181818], [0.03636364, 0.14545455]]),
            )

    def test_query_common_var(self, bayesian_inference):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            with pytest.raises(ValueError):
                bayesian_inference.query(
                    variables=["J"],
                    evidence=["J"],
                    elimination_order=order,
                )

    def test_max_marginal(self, bayesian_inference):
        np_test.assert_almost_equal(
            float(bayesian_inference.max_marginal()), 0.1659, decimal=4
        )

    def test_max_marginal_var(self, bayesian_inference):
        np_test.assert_almost_equal(
            float(bayesian_inference.max_marginal(["G"])), 0.6, decimal=4
        )

    def test_max_marginal_var1(self, bayesian_inference):
        np_test.assert_almost_equal(
            float(bayesian_inference.max_marginal(["G", "R"])), 0.36, decimal=4
        )

    def test_max_marginal_var2(self, bayesian_inference):
        np_test.assert_almost_equal(
            float(bayesian_inference.max_marginal(["G", "R", "A"])),
            0.288,
            decimal=4,
        )

    def test_max_marginal_common_var(self, bayesian_inference):
        with pytest.raises(ValueError):
            bayesian_inference.max_marginal(
                variables=["J"],
                evidence=["J"],
            )

    def test_map_query(self, bayesian_inference):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            map_query = bayesian_inference.map_query(
                elimination_order=order, show_progress=False
            )
            assert map_query == {"A": 1, "R": 1, "J": 1, "Q": 1, "G": 0, "L": 0}

    def test_map_query_with_evidence(self, bayesian_inference):
        map_query = bayesian_inference.map_query(
            ["A", "R", "L"], {"J": 0, "Q": 1, "G": 0}, show_progress=False
        )
        assert map_query == {"A": 1, "R": 0, "L": 0}

    def test_map_query_common_var(self, bayesian_inference):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            with pytest.raises(ValueError):
                bayesian_inference.map_query(
                    variables=["J"],
                    evidence=["J"],
                    elimination_order=order,
                )

    def test_elimination_order(self, bayesian_inference):
        # Check all the heuristics give the same results.
        for elimination_order in [
            "WeightedMinFill",
            "MinNeighbors",
            "MinWeight",
            "MinFill",
        ]:
            query_result = bayesian_inference.query(
                ["J"], elimination_order=elimination_order, show_progress=False
            )
            assert query_result == DiscreteFactor(
                variables=["J"], cardinality=[2], values=[0.416, 0.584]
            )

            query_result = bayesian_inference.query(
                variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
            )
            assert query_result == DiscreteFactor(
                variables=["J"], cardinality=[2], values=[0.6, 0.4]
            )

        # Check when elimination order has extra variables. Because of pruning.
        query_result = bayesian_inference.query(
            ["J"], elimination_order=["A", "R", "L", "Q", "G"], show_progress=False
        )
        assert query_result == DiscreteFactor(
            variables=["J"], cardinality=[2], values=[0.416, 0.584]
        )

        # Check for when elimination order doesn't have all the variables
        with pytest.raises(ValueError):
            bayesian_inference.query(
                variables=["J"],
                elimination_order=["A"],
            )

    def test_induced_graph(self, bayesian_inference):
        induced_graph = bayesian_inference.induced_graph(["G", "Q", "A", "J", "L", "R"])
        result_edges = sorted([sorted(x) for x in induced_graph.edges()])
        assert result_edges == [
            ["A", "J"],
            ["A", "R"],
            ["G", "J"],
            ["G", "L"],
            ["J", "L"],
            ["J", "Q"],
            ["J", "R"],
            ["L", "R"],
        ]

    def test_induced_width(self, bayesian_inference):
        result_width = bayesian_inference.induced_width(["G", "Q", "A", "J", "L", "R"])
        assert 2 == result_width


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestSnowNetworkTorch:
    def test_queries(self, snow_network_torch_model):
        for algo in [VariableElimination, BeliefPropagation]:
            infer = algo(snow_network_torch_model)
            query1 = infer.query(
                ["Snow"], evidence={"Traffic": "slow"}, show_progress=False
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query1.values), [0.533333, 0.466667]
            )

            query2 = infer.query(
                ["Risk"], evidence={"Traffic": "slow"}, show_progress=False
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query2.values), [0.613333, 0.386667]
            )

            query3 = infer.query(
                ["Late"], evidence={"Traffic": "slow"}, show_progress=False
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query3.values), [0.7920, 0.2080]
            )

            with pytest.raises(ValueError):
                infer.query(
                    variables=["Traffic"],
                    evidence={"Traffic": "slow"},
                )

    def test_elimination_order(self, snow_network_torch_model):
        infer = VariableElimination(snow_network_torch_model)
        for order in ["MinFill", "MinNeighbors", "MinWeight", "WeightedMinFill"]:
            computed_order = infer._get_elimination_order(
                variables=["Traffic"], evidence={}, elimination_order=order
            )
            assert set(computed_order) == {"Risk", "Late", "Snow"}

        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query1 = infer.query(
                ["Snow"],
                evidence={"Traffic": "slow"},
                elimination_order=order,
                show_progress=False,
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query1.values), [0.533333, 0.466667]
            )

            query2 = infer.query(
                ["Risk"],
                evidence={"Traffic": "slow"},
                elimination_order=order,
                show_progress=False,
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query2.values), [0.613333, 0.386667]
            )

            query3 = infer.query(
                ["Late"],
                evidence={"Traffic": "slow"},
                elimination_order=order,
                show_progress=False,
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query3.values), [0.7920, 0.2080]
            )

    def test_joint_distribution(self, snow_network_torch_model):
        infer = VariableElimination(snow_network_torch_model)
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_expected = {}
            query_expected["Snow"] = infer.query(
                ["Snow"], elimination_order=order, show_progress=False
            )
            query_expected["Risk"] = infer.query(
                ["Risk"], elimination_order=order, show_progress=False
            )

            query_joint = infer.query(
                ["Snow", "Risk"], elimination_order=order, joint=False
            )
            for var in ["Snow", "Risk"]:
                assert query_joint[var] == query_expected[var]

    def test_virt_evidence(self, snow_network_torch_model):
        virt_evidence = TabularCPD(
            "Traffic", 2, [[0.3], [0.7]], state_names={"Traffic": ["normal", "slow"]}
        )
        for algo in [VariableElimination, BeliefPropagation]:
            infer = algo(snow_network_torch_model)
            query1 = infer.query(
                ["Snow"], virtual_evidence=[virt_evidence], show_progress=False
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query1.values), [0.45, 0.55]
            )

            map1 = infer.map_query(
                ["Snow"], virtual_evidence=[virt_evidence], show_progress=False
            )
            assert map1 == {"Snow": "no"}

            query2 = infer.query(
                ["Risk"], virtual_evidence=[virt_evidence], show_progress=False
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query2.values), [0.58, 0.42]
            )

            map2 = infer.map_query(
                ["Risk"], virtual_evidence=[virt_evidence], show_progress=False
            )
            assert map2 == {"Risk": "yes"}

            query3 = infer.query(
                ["Late"], virtual_evidence=[virt_evidence], show_progress=False
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query3.values), [0.61625, 0.38375]
            )

            map3 = infer.map_query(
                ["Late"], virtual_evidence=[virt_evidence], show_progress=False
            )
            assert map3 == {"Late": "yes"}

            query4 = infer.query(
                ["Traffic"], virtual_evidence=[virt_evidence], show_progress=False
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query4.values), [0.34375, 0.65625]
            )

            # TODO: State name should be returned here.
            map4 = infer.map_query(
                ["Traffic"], virtual_evidence=[virt_evidence], show_progress=False
            )
            assert map4 in [{"Traffic": "slow"}, {"Traffic": 1}]

        virt_evidence1 = TabularCPD(
            "Risk", 2, [[0.7], [0.3]], state_names={"Risk": ["yes", "no"]}
        )
        for algo in [VariableElimination, BeliefPropagation]:
            infer = algo(snow_network_torch_model)
            query1 = infer.query(
                ["Snow"],
                virtual_evidence=[virt_evidence, virt_evidence1],
                show_progress=False,
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query1.values), [0.52443609, 0.47556391]
            )

            map1 = infer.map_query(
                ["Snow"],
                virtual_evidence=[virt_evidence, virt_evidence1],
                show_progress=False,
            )
            assert map1 == {"Snow": "yes"}

            query2 = infer.query(
                ["Risk"],
                virtual_evidence=[virt_evidence, virt_evidence1],
                show_progress=False,
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query2.values), [0.76315789, 0.23684211]
            )
            map2 = infer.map_query(
                ["Risk"],
                virtual_evidence=[virt_evidence, virt_evidence1],
                show_progress=False,
            )
            assert map2 in [{"Risk": 0}, {"Risk": "yes"}]

            query3 = infer.query(
                ["Traffic"],
                virtual_evidence=[virt_evidence, virt_evidence1],
                show_progress=False,
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query3.values), [0.32730263, 0.67269737]
            )
            map3 = infer.map_query(
                ["Traffic"],
                virtual_evidence=[virt_evidence, virt_evidence1],
                show_progress=False,
            )
            assert map3 in [{"Traffic": "slow"}, {"Traffic": 1}]

            query4 = infer.query(
                ["Late"],
                virtual_evidence=[virt_evidence, virt_evidence1],
                show_progress=False,
            )
            np_test.assert_array_almost_equal(
                compat_fns.to_numpy(query4.values), [0.66480263, 0.33519737]
            )
            map4 = infer.map_query(
                ["Late"],
                virtual_evidence=[virt_evidence, virt_evidence1],
                show_progress=False,
            )
            assert map4 == {"Late": "yes"}


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestVariableEliminationDuplicatedFactors:
    def test_duplicated_factors(self, test_duplicated_factors_markov_inference):
        query_result = test_duplicated_factors_markov_inference.query(
            ["A"], show_progress=False
        )
        assert query_result == DiscreteFactor(
            variables=["A"], cardinality=[2], values=np.array([4, 4])
        )


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestVariableEliminationMarkov:

    # All the values that are used for comparison in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self, variable_elimination_markov_inference):
        query_result = variable_elimination_markov_inference.query(
            ["J"], show_progress=False
        )
        assert query_result == DiscreteFactor(
            variables=["J"], cardinality=[2], values=np.array([0.416, 0.584])
        )

    def test_query_multiple_variable(self, variable_elimination_markov_inference):
        query_result = variable_elimination_markov_inference.query(
            ["Q", "J"], show_progress=False
        )
        assert query_result == DiscreteFactor(
            variables=["Q", "J"],
            cardinality=[2, 2],
            values=np.array([[0.3744, 0.1168], [0.0416, 0.4672]]),
        )

    def test_query_single_variable_with_evidence(
        self, variable_elimination_markov_inference
    ):
        query_result = variable_elimination_markov_inference.query(
            variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
        )
        assert query_result == DiscreteFactor(
            variables=["J"], cardinality=[2], values=[0.072, 0.048]
        )

    def test_query_multiple_variable_with_evidence(
        self, variable_elimination_markov_inference
    ):
        query_result = variable_elimination_markov_inference.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        assert query_result == DiscreteFactor(
            variables=["J", "Q"],
            cardinality=[2, 2],
            values=np.array([[0.003888, 0.000432], [0.000192, 0.000768]]),
        )

    def test_query_multiple_times(self, variable_elimination_markov_inference):
        # This just tests that the models are not getting modified while querying them
        query_result = variable_elimination_markov_inference.query(
            ["J"], show_progress=False
        )
        query_result = variable_elimination_markov_inference.query(
            ["J"], show_progress=False
        )
        assert query_result == DiscreteFactor(
            variables=["J"], cardinality=[2], values=np.array([0.416, 0.584])
        )

        query_result = variable_elimination_markov_inference.query(
            ["Q", "J"], show_progress=False
        )
        query_result = variable_elimination_markov_inference.query(
            ["Q", "J"], show_progress=False
        )
        assert query_result == DiscreteFactor(
            variables=["Q", "J"],
            cardinality=[2, 2],
            values=np.array([[0.3744, 0.1168], [0.0416, 0.4672]]),
        )

        query_result = variable_elimination_markov_inference.query(
            variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
        )
        query_result = variable_elimination_markov_inference.query(
            variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
        )
        assert query_result == DiscreteFactor(
            variables=["J"], cardinality=[2], values=[0.072, 0.048]
        )

        query_result = variable_elimination_markov_inference.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        query_result = variable_elimination_markov_inference.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        assert query_result == DiscreteFactor(
            variables=["J", "Q"],
            cardinality=[2, 2],
            values=np.array([[0.003888, 0.000432], [0.000192, 0.000768]]),
        )

    def test_max_marginal(self, variable_elimination_markov_inference):
        np_test.assert_almost_equal(
            float(variable_elimination_markov_inference.max_marginal()),
            0.1659,
            decimal=4,
        )

    def test_max_marginal_var(self, variable_elimination_markov_inference):
        np_test.assert_almost_equal(
            float(variable_elimination_markov_inference.max_marginal(["G"])),
            0.1659,
            decimal=4,
        )

    def test_max_marginal_var1(self, variable_elimination_markov_inference):
        np_test.assert_almost_equal(
            float(variable_elimination_markov_inference.max_marginal(["G", "R"])),
            0.1659,
            decimal=4,
        )

    def test_max_marginal_var2(self, variable_elimination_markov_inference):
        np_test.assert_almost_equal(
            float(variable_elimination_markov_inference.max_marginal(["G", "R", "A"])),
            0.1659,
            decimal=4,
        )

    def test_map_query(self, variable_elimination_markov_inference):
        map_query = variable_elimination_markov_inference.map_query(show_progress=False)
        assert map_query == {"A": 1, "R": 1, "J": 1, "Q": 1, "G": 0, "L": 0}

    def test_map_query_with_evidence(self, variable_elimination_markov_inference):
        map_query = variable_elimination_markov_inference.map_query(
            ["A", "R", "L"], {"J": 0, "Q": 1, "G": 0}, show_progress=False
        )
        assert map_query == {"A": 1, "R": 0, "L": 0}

    def test_induced_graph(self, variable_elimination_markov_inference):
        induced_graph = variable_elimination_markov_inference.induced_graph(
            ["G", "Q", "A", "J", "L", "R"]
        )
        result_edges = sorted([sorted(x) for x in induced_graph.edges()])
        assert result_edges == [
            ["A", "J"],
            ["A", "R"],
            ["G", "J"],
            ["G", "L"],
            ["J", "L"],
            ["J", "Q"],
            ["J", "R"],
            ["L", "R"],
        ]

    def test_induced_width(self, variable_elimination_markov_inference):
        result_width = variable_elimination_markov_inference.induced_width(
            ["G", "Q", "A", "J", "L", "R"]
        )
        assert 2 == result_width

    def test_issue_1421(self):
        model = DiscreteBayesianNetwork([("X", "Y"), ("Z", "X"), ("W", "Y")])
        cpd_z = TabularCPD(variable="Z", variable_card=2, values=[[0.5], [0.5]])

        cpd_x = TabularCPD(
            variable="X",
            variable_card=2,
            values=[[0.25, 0.75], [0.75, 0.25]],
            evidence=["Z"],
            evidence_card=[2],
        )

        cpd_w = TabularCPD(variable="W", variable_card=2, values=[[0.5], [0.5]])
        cpd_y = TabularCPD(
            variable="Y",
            variable_card=2,
            values=[[0.3, 0.4, 0.7, 0.8], [0.7, 0.6, 0.3, 0.2]],
            evidence=["X", "W"],
            evidence_card=[2, 2],
        )

        model.add_cpds(cpd_z, cpd_x, cpd_w, cpd_y)

        infer = VariableElimination(model)
        np_test.assert_array_almost_equal(
            compat_fns.to_numpy(
                infer.query(["Y"], evidence={"X": 0}, show_progress=False).values
            ),
            [0.35, 0.65],
        )


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestBeliefPropagation:
    def test_calibrate_clique_belief(self, junction_tree):
        belief_propagation = BeliefPropagation(junction_tree)
        belief_propagation.calibrate()
        clique_belief = belief_propagation.get_clique_beliefs()

        phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
        phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
        phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))

        b_A_B = phi1 * (phi3.marginalize(["D"], inplace=False) * phi2).marginalize(
            ["C"], inplace=False
        )
        b_B_C = phi2 * (
            phi1.marginalize(["A"], inplace=False)
            * phi3.marginalize(["D"], inplace=False)
        )
        b_C_D = phi3 * (phi1.marginalize(["A"], inplace=False) * phi2).marginalize(
            ["B"], inplace=False
        )

        assert clique_belief[("A", "B")] == b_A_B
        assert clique_belief[("B", "C")] == b_B_C
        assert clique_belief[("C", "D")] == b_C_D

    def test_calibrate_sepset_belief(self, junction_tree):
        belief_propagation = BeliefPropagation(junction_tree)
        belief_propagation.calibrate()
        sepset_belief = belief_propagation.get_sepset_beliefs()

        phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
        phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
        phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))

        b_B = (
            phi1
            * (phi3.marginalize(["D"], inplace=False) * phi2).marginalize(
                ["C"], inplace=False
            )
        ).marginalize(["A"], inplace=False)

        b_C = (
            phi2
            * (
                phi1.marginalize(["A"], inplace=False)
                * phi3.marginalize(["D"], inplace=False)
            )
        ).marginalize(["B"], inplace=False)

        assert all(
            sepset_belief[frozenset((("A", "B"), ("B", "C")))].values == b_B.values
        )
        assert all(
            sepset_belief[frozenset((("B", "C"), ("C", "D")))].values == b_C.values
        )

    def test_max_calibrate_clique_belief(self, junction_tree):
        belief_propagation = BeliefPropagation(junction_tree)
        belief_propagation.max_calibrate()
        clique_belief = belief_propagation.get_clique_beliefs()

        phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
        phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
        phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))

        b_A_B = phi1 * (phi3.maximize(["D"], inplace=False) * phi2).maximize(
            ["C"], inplace=False
        )
        b_B_C = phi2 * (
            phi1.maximize(["A"], inplace=False) * phi3.maximize(["D"], inplace=False)
        )
        b_C_D = phi3 * (phi1.maximize(["A"], inplace=False) * phi2).maximize(
            ["B"], inplace=False
        )

        assert clique_belief[("A", "B")] == b_A_B
        assert clique_belief[("B", "C")] == b_B_C
        assert clique_belief[("C", "D")] == b_C_D

    def test_max_calibrate_sepset_belief(self, junction_tree):
        belief_propagation = BeliefPropagation(junction_tree)
        belief_propagation.max_calibrate()
        sepset_belief = belief_propagation.get_sepset_beliefs()

        phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
        phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
        phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))

        b_B = (
            phi1
            * (phi3.maximize(["D"], inplace=False) * phi2).maximize(
                ["C"], inplace=False
            )
        ).maximize(["A"], inplace=False)

        b_C = (
            phi2
            * (
                phi1.maximize(["A"], inplace=False)
                * phi3.maximize(["D"], inplace=False)
            )
        ).maximize(["B"], inplace=False)

        assert all(
            sepset_belief[frozenset((("A", "B"), ("B", "C")))].values == b_B.values
        )
        assert all(
            sepset_belief[frozenset((("B", "C"), ("C", "D")))].values == b_C.values
        )

    # All the values that are used for comparison in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self, bayesian_model):
        belief_propagation = BeliefPropagation(bayesian_model)
        query_result = belief_propagation.query(["J"], show_progress=False)
        assert query_result == DiscreteFactor(
            variables=["J"], cardinality=[2], values=[0.416, 0.584]
        )

    def test_query_multiple_variable(self, bayesian_model):
        belief_propagation = BeliefPropagation(bayesian_model)
        query_result = belief_propagation.query(["Q", "J"], show_progress=False)
        assert query_result == DiscreteFactor(
            variables=["J", "Q"],
            cardinality=[2, 2],
            values=np.array([[0.3744, 0.0416], [0.1168, 0.4672]]),
        )

    def test_query_single_variable_with_evidence(self, bayesian_model):
        belief_propagation = BeliefPropagation(bayesian_model)
        query_result = belief_propagation.query(
            variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
        )
        assert query_result == DiscreteFactor(
            variables=["J"], cardinality=[2], values=np.array([0.6, 0.4])
        )

    def test_query_multiple_variable_with_evidence(self, bayesian_model):
        belief_propagation = BeliefPropagation(bayesian_model)
        query_result = belief_propagation.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        assert query_result == DiscreteFactor(
            variables=["J", "Q"],
            cardinality=[2, 2],
            values=np.array([[0.73636364, 0.08181818], [0.03636364, 0.14545455]]),
        )

    def test_query_common_var(self, bayesian_model):
        belief_propagation = BeliefPropagation(bayesian_model)
        with pytest.raises(ValueError):
            belief_propagation.query(variables=["J"], evidence=["J"])

    def test_map_query(self, bayesian_model):
        belief_propagation = BeliefPropagation(bayesian_model)
        map_query = belief_propagation.map_query(show_progress=False)
        assert map_query == {"A": 1, "R": 1, "J": 1, "Q": 1, "G": 0, "L": 0}

    def test_map_query_with_evidence(self, bayesian_model):
        belief_propagation = BeliefPropagation(bayesian_model)
        map_query = belief_propagation.map_query(
            ["A", "R", "L"], {"J": 0, "Q": 1, "G": 0}, show_progress=False
        )
        assert map_query == {"A": 1, "R": 0, "L": 0}

    def test_map_query_common_var(self, bayesian_model):
        belief_propagation = BeliefPropagation(bayesian_model)
        with pytest.raises(ValueError):
            belief_propagation.map_query(variables=["J"], evidence=["J"])

    def test_issue_1048(self):
        model = DiscreteBayesianNetwork()

        # Nodes
        parents = ["parent"]
        children = [f"child_{i}" for i in range(10)]

        # Add nodes and edges
        model.add_nodes_from(parents + children)
        model.add_edges_from(itertools.product(parents, children))

        # Add cpds
        model.add_cpds(TabularCPD(parents[0], 2, [[0.5], [0.5]]))
        for c in children:
            model.add_cpds(
                TabularCPD(
                    c, 2, [[0.9, 0.1], [0.1, 0.9]], evidence=parents, evidence_card=[2]
                )
            )

        # Infer
        inf = BeliefPropagation(model)
        inf.calibrate()
        evidence = {}

        expected_evidences = [
            {},
            {"child_0": 1},
            {"child_0": 1, "child_1": 1},
            {"child_0": 1, "child_1": 1, "child_2": 1},
        ]
        expected_values = [
            np.array([0.5, 0.5]),
            np.array([0.1, 0.9]),
            np.array([0.0122, 0.9878]),
            np.array([0.0014, 0.9987]),
        ]
        for i, c in enumerate(children[:4]):
            assert evidence == expected_evidences[i]
            np_test.assert_almost_equal(
                compat_fns.to_numpy(
                    inf.query(["parent"], evidence, show_progress=False)
                    .normalize(inplace=False)
                    .values
                ),
                expected_values[i],
                decimal=2,
            )
            evidence.update({c: 1})
