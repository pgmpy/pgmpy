import numpy as np
import numpy.testing as np_test
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork, FunctionalBayesianNetwork


def build_bayesian_model():
    model = DiscreteBayesianNetwork([("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")])
    cpd_a = TabularCPD("A", 2, values=[[0.2], [0.8]])
    cpd_r = TabularCPD("R", 2, values=[[0.4], [0.6]])
    cpd_j = TabularCPD(
        "J",
        2,
        values=[[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
        evidence=["A", "R"],
        evidence_card=[2, 2],
    )
    cpd_q = TabularCPD("Q", 2, values=[[0.9, 0.2], [0.1, 0.8]], evidence=["J"], evidence_card=[2])
    cpd_l = TabularCPD(
        "L",
        2,
        values=[[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        evidence=["J", "G"],
        evidence_card=[2, 2],
    )
    cpd_g = TabularCPD("G", 2, values=[[0.6], [0.4]])
    model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
    return model


def build_markov_model():
    model = DiscreteMarkovNetwork(
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
    factor_q = TabularCPD("Q", 2, values=[[0.9, 0.2], [0.1, 0.8]], evidence=["J"], evidence_card=[2]).to_factor()
    factor_l = TabularCPD(
        "L",
        2,
        values=[[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        evidence=["J", "G"],
        evidence_card=[2, 2],
    ).to_factor()
    factor_g = TabularCPD("G", 2, [[0.6], [0.4]]).to_factor()

    model.add_factors(factor_a, factor_r, factor_j, factor_q, factor_l, factor_g)
    return model


def build_snow_model():
    model = DiscreteBayesianNetwork(
        [
            ("Snow", "Risk"),
            ("Snow", "Traffic"),
            ("Traffic", "Late"),
            ("Risk", "Late"),
        ]
    )

    cpd_snow = TabularCPD("Snow", 2, [[0.4], [0.6]], state_names={"Snow": ["yes", "no"]})
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
def bayesian_ve():
    model = build_bayesian_model()
    return model, VariableElimination(model)


@pytest.fixture
def torch_models():
    from pgmpy.example_models import load_model

    config.set_backend("torch")
    yield {
        "lgbm": load_model("bnlearn/ecoli70"),
        "fbn": FunctionalBayesianNetwork([("X", "Y")]),
    }
    config.set_backend("numpy")


@pytest.fixture
def snow_model():
    return build_snow_model()


@pytest.fixture
def duplicated_markov_ve():
    markov_model = DiscreteMarkovNetwork([("A", "B"), ("A", "C")])
    f1 = DiscreteFactor(variables=["A", "B"], cardinality=[2, 2], values=np.eye(2) * 2)
    f2 = DiscreteFactor(variables=["A", "C"], cardinality=[2, 2], values=np.eye(2) * 2)
    markov_model.add_factors(f1, f2)
    return markov_model, VariableElimination(markov_model)


@pytest.fixture
def markov_ve():
    model = build_markov_model()
    return model, VariableElimination(model)


requires_torch = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)

ELIMINATION_ORDERS = ["greedy", "MinFill", "MinNeighbors", "MinWeight", "WeightedMinFill"]


class TestVariableElimination:
    def test_query_raises_for_empty_variables(self):
        model = load_model("bnlearn/earthquake")
        infer = VariableElimination(model)

        with pytest.raises(ValueError) as exc_info:
            infer.query(variables=[], evidence={"A": 1})

        assert "must contain at least one variable" in str(exc_info.value)

    def test_query_single_variable(self, bayesian_ve):
        _, infer = bayesian_ve
        for order in ELIMINATION_ORDERS:
            query_result = infer.query(["J"], elimination_order=order, show_progress=False)
            assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=[0.416, 0.584])

    def test_query_multiple_variable(self, bayesian_ve):
        _, infer = bayesian_ve
        for order in ELIMINATION_ORDERS:
            query_result = infer.query(["Q", "J"], elimination_order=order, show_progress=False)
            assert query_result == DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.3744, 0.0416], [0.1168, 0.4672]]),
            )

    def test_query_single_variable_with_evidence(self, bayesian_ve):
        _, infer = bayesian_ve
        for order in ELIMINATION_ORDERS:
            query_result = infer.query(
                variables=["J"],
                evidence={"A": 0, "R": 1},
                elimination_order=order,
                show_progress=False,
            )
            assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=[0.6, 0.4])

    def test_query_multiple_variable_with_evidence(self, bayesian_ve):
        _, infer = bayesian_ve
        for order in ELIMINATION_ORDERS:
            query_result = infer.query(
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

    def test_query_multiple_times(self, bayesian_ve):
        _, infer = bayesian_ve
        for order in ELIMINATION_ORDERS:
            query_result = infer.query(["J"], elimination_order=order, show_progress=False)
            query_result = infer.query(["J"], elimination_order=order, show_progress=False)
            assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=np.array([0.416, 0.584]))
            query_result = infer.query(["Q", "J"], elimination_order=order, show_progress=False)
            query_result = infer.query(["Q", "J"], elimination_order=order, show_progress=False)
            assert query_result == DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.3744, 0.0416], [0.1168, 0.4672]]),
            )

            query_result = infer.query(
                variables=["J"],
                evidence={"A": 0, "R": 1},
                elimination_order=order,
                show_progress=False,
            )
            query_result = infer.query(
                variables=["J"],
                evidence={"A": 0, "R": 1},
                elimination_order=order,
                show_progress=False,
            )
            assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=[0.6, 0.4])

            query_result = infer.query(
                variables=["J", "Q"],
                evidence={"A": 0, "R": 0, "G": 0, "L": 1},
                elimination_order=order,
                show_progress=False,
            )
            query_result = infer.query(
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

    def test_query_common_var(self, bayesian_ve):
        _, infer = bayesian_ve
        for order in ELIMINATION_ORDERS:
            with pytest.raises(ValueError):
                infer.query(variables=["J"], evidence=["J"], elimination_order=order)

    def test_max_marginal(self, bayesian_ve):
        _, infer = bayesian_ve
        np_test.assert_almost_equal(infer.max_marginal(), 0.1659, decimal=4)

    def test_max_marginal_var(self, bayesian_ve):
        _, infer = bayesian_ve
        np_test.assert_almost_equal(infer.max_marginal(["G"]), 0.6, decimal=4)

    def test_max_marginal_var1(self, bayesian_ve):
        _, infer = bayesian_ve
        np_test.assert_almost_equal(infer.max_marginal(["G", "R"]), 0.36, decimal=4)

    def test_max_marginal_var2(self, bayesian_ve):
        _, infer = bayesian_ve
        np_test.assert_almost_equal(infer.max_marginal(["G", "R", "A"]), 0.288, decimal=4)

    def test_max_marginal_common_var(self, bayesian_ve):
        _, infer = bayesian_ve
        with pytest.raises(ValueError):
            infer.max_marginal(variables=["J"], evidence=["J"])

    def test_map_query(self, bayesian_ve):
        _, infer = bayesian_ve
        for order in ELIMINATION_ORDERS:
            map_query = infer.map_query(elimination_order=order, show_progress=False)
            assert map_query == {"A": 1, "R": 1, "J": 1, "Q": 1, "G": 0, "L": 0}

    def test_map_query_with_evidence(self, bayesian_ve):
        _, infer = bayesian_ve
        map_query = infer.map_query(["A", "R", "L"], {"J": 0, "Q": 1, "G": 0}, show_progress=False)
        assert map_query == {"A": 1, "R": 0, "L": 0}

    def test_map_query_common_var(self, bayesian_ve):
        _, infer = bayesian_ve
        for order in ELIMINATION_ORDERS:
            with pytest.raises(ValueError):
                infer.map_query(variables=["J"], evidence=["J"], elimination_order=order)

    def test_elimination_order(self, bayesian_ve):
        _, infer = bayesian_ve
        for elimination_order in ["WeightedMinFill", "MinNeighbors", "MinWeight", "MinFill"]:
            query_result = infer.query(["J"], elimination_order=elimination_order, show_progress=False)
            assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=[0.416, 0.584])

            query_result = infer.query(variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False)
            assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=[0.6, 0.4])

        query_result = infer.query(["J"], elimination_order=["A", "R", "L", "Q", "G"], show_progress=False)
        assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=[0.416, 0.584])

        with pytest.raises(ValueError):
            infer.query(variables=["J"], elimination_order=["A"])

    def test_induced_graph(self, bayesian_ve):
        _, infer = bayesian_ve
        induced_graph = infer.induced_graph(["G", "Q", "A", "J", "L", "R"])
        result_edges = sorted([sorted(x) for x in induced_graph.edges()])
        assert [
            ["A", "J"],
            ["A", "R"],
            ["G", "J"],
            ["G", "L"],
            ["J", "L"],
            ["J", "Q"],
            ["J", "R"],
            ["L", "R"],
        ] == result_edges

    def test_induced_width(self, bayesian_ve):
        _, infer = bayesian_ve
        result_width = infer.induced_width(["G", "Q", "A", "J", "L", "R"])
        assert 2 == result_width

    def test_invalid_state_name(self, bayesian_ve):
        _, infer = bayesian_ve
        with pytest.raises(KeyError):
            infer.query(variables=["J"], evidence={"A": -1}, show_progress=False)

    def test_invalid_variable_name(self, bayesian_ve):
        _, infer = bayesian_ve
        with pytest.raises(ValueError):
            infer.query(variables=["J"], evidence={"wrong_variable": 0}, show_progress=False)


class TestSnowNetwork:
    def test_queries(self, snow_model):
        for algo in [VariableElimination, BeliefPropagation]:
            infer = algo(snow_model)
            query1 = infer.query(["Snow"], evidence={"Traffic": "slow"}, show_progress=False)
            np_test.assert_array_almost_equal(query1.values, [0.533333, 0.466667])

            query2 = infer.query(["Risk"], evidence={"Traffic": "slow"}, show_progress=False)
            np_test.assert_array_almost_equal(query2.values, [0.613333, 0.386667])

            query3 = infer.query(["Late"], evidence={"Traffic": "slow"}, show_progress=False)
            np_test.assert_array_almost_equal(query3.values, [0.7920, 0.2080])

            with pytest.raises(ValueError):
                infer.query(variables=["Traffic"], evidence={"Traffic": "slow"})

    def test_elimination_order(self, snow_model):
        infer = VariableElimination(snow_model)
        for order in ["MinFill", "MinNeighbors", "MinWeight", "WeightedMinFill"]:
            computed_order = infer._get_elimination_order(variables=["Traffic"], evidence={}, elimination_order=order)
            assert set(computed_order) == {"Risk", "Late", "Snow"}

        for order in ["greedy", "MinFill", "MinNeighbors", "MinWeight", "WeightedMinFill"]:
            query1 = infer.query(
                ["Snow"],
                evidence={"Traffic": "slow"},
                elimination_order=order,
                show_progress=False,
            )
            np_test.assert_array_almost_equal(query1.values, [0.533333, 0.466667])

            query2 = infer.query(
                ["Risk"],
                evidence={"Traffic": "slow"},
                elimination_order=order,
                show_progress=False,
            )
            np_test.assert_array_almost_equal(query2.values, [0.613333, 0.386667])

            query3 = infer.query(
                ["Late"],
                evidence={"Traffic": "slow"},
                elimination_order=order,
                show_progress=False,
            )
            np_test.assert_array_almost_equal(query3.values, [0.7920, 0.2080])

    def test_joint_distribution(self, snow_model):
        infer = VariableElimination(snow_model)
        for order in ELIMINATION_ORDERS:
            query_expected = {}
            query_expected["Snow"] = infer.query(["Snow"], elimination_order=order, show_progress=False)
            query_expected["Risk"] = infer.query(["Risk"], elimination_order=order, show_progress=False)

            query_joint = infer.query(["Snow", "Risk"], elimination_order=order, joint=False)
            for var in ["Snow", "Risk"]:
                assert query_joint[var] == query_expected[var]

    def test_virt_evidence(self, snow_model):
        virt_evidence_cpd = TabularCPD("Traffic", 2, [[0.3], [0.7]], state_names={"Traffic": ["normal", "slow"]})
        virt_evidence_factor = DiscreteFactor(["Traffic"], [2], [0.3, 0.7], state_names={"Traffic": ["normal", "slow"]})
        for virt_evidence in [virt_evidence_cpd, virt_evidence_factor]:
            for algo in [VariableElimination, BeliefPropagation]:
                infer = algo(snow_model)
                query1 = infer.query(["Snow"], virtual_evidence=[virt_evidence], show_progress=False)
                np_test.assert_array_almost_equal(query1.values, [0.45, 0.55])

                map1 = infer.map_query(["Snow"], virtual_evidence=[virt_evidence], show_progress=False)
                assert map1 == {"Snow": "no"}

                query2 = infer.query(["Risk"], virtual_evidence=[virt_evidence], show_progress=False)
                np_test.assert_array_almost_equal(query2.values, [0.58, 0.42])

                map2 = infer.map_query(["Risk"], virtual_evidence=[virt_evidence], show_progress=False)
                assert map2 == {"Risk": "yes"}

                query3 = infer.query(["Late"], virtual_evidence=[virt_evidence], show_progress=False)
                np_test.assert_array_almost_equal(query3.values, [0.61625, 0.38375])

                map3 = infer.map_query(["Late"], virtual_evidence=[virt_evidence], show_progress=False)
                assert map3 == {"Late": "yes"}

                query4 = infer.query(["Traffic"], virtual_evidence=[virt_evidence], show_progress=False)
                np_test.assert_array_almost_equal(query4.values, [0.34375, 0.65625])

                map4 = infer.map_query(["Traffic"], virtual_evidence=[virt_evidence], show_progress=False)
                assert map4 in [{"Traffic": "slow"}, {"Traffic": 1}]

        virt_evidence1_cpd = TabularCPD("Risk", 2, [[0.7], [0.3]], state_names={"Risk": ["yes", "no"]})
        virt_evidence1_factor = DiscreteFactor(["Risk"], [2], [0.7, 0.3], state_names={"Risk": ["yes", "no"]})
        for virt_evidence in [virt_evidence_cpd, virt_evidence_factor]:
            for virt_evidence1 in [virt_evidence1_cpd, virt_evidence1_factor]:
                for algo in [VariableElimination, BeliefPropagation]:
                    infer = algo(snow_model)
                    query1 = infer.query(
                        ["Snow"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    np_test.assert_array_almost_equal(query1.values, [0.52443609, 0.47556391])

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
                    np_test.assert_array_almost_equal(query2.values, [0.76315789, 0.23684211])
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
                    np_test.assert_array_almost_equal(query3.values, [0.32730263, 0.67269737])
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
                    np_test.assert_array_almost_equal(query4.values, [0.66480263, 0.33519737])
                    map4 = infer.map_query(
                        ["Late"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    assert map4 == {"Late": "yes"}


class TestVariableEliminationDuplicatedFactors:
    def test_duplicated_factors(self, duplicated_markov_ve):
        _, infer = duplicated_markov_ve
        query_result = infer.query(["A"], show_progress=False)
        assert query_result == DiscreteFactor(variables=["A"], cardinality=[2], values=np.array([4, 4]))


class TestVariableEliminationMarkov:
    def test_query_single_variable(self, markov_ve):
        _, infer = markov_ve
        query_result = infer.query(["J"], show_progress=False)
        assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=np.array([0.416, 0.584]))

    def test_query_multiple_variable(self, markov_ve):
        _, infer = markov_ve
        query_result = infer.query(["Q", "J"], show_progress=False)
        assert query_result == DiscreteFactor(
            variables=["Q", "J"],
            cardinality=[2, 2],
            values=np.array([[0.3744, 0.1168], [0.0416, 0.4672]]),
        )

    def test_query_single_variable_with_evidence(self, markov_ve):
        _, infer = markov_ve
        query_result = infer.query(variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False)
        assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=[0.072, 0.048])

    def test_query_multiple_variable_with_evidence(self, markov_ve):
        _, infer = markov_ve
        query_result = infer.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        assert query_result == DiscreteFactor(
            variables=["J", "Q"],
            cardinality=[2, 2],
            values=np.array([[0.003888, 0.000432], [0.000192, 0.000768]]),
        )

    def test_query_multiple_times(self, markov_ve):
        _, infer = markov_ve
        query_result = infer.query(["J"], show_progress=False)
        query_result = infer.query(["J"], show_progress=False)
        assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=np.array([0.416, 0.584]))

        query_result = infer.query(["Q", "J"], show_progress=False)
        query_result = infer.query(["Q", "J"], show_progress=False)
        assert query_result == DiscreteFactor(
            variables=["Q", "J"],
            cardinality=[2, 2],
            values=np.array([[0.3744, 0.1168], [0.0416, 0.4672]]),
        )

        query_result = infer.query(variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False)
        query_result = infer.query(variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False)
        assert query_result == DiscreteFactor(variables=["J"], cardinality=[2], values=[0.072, 0.048])

        query_result = infer.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        query_result = infer.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        assert query_result == DiscreteFactor(
            variables=["J", "Q"],
            cardinality=[2, 2],
            values=np.array([[0.003888, 0.000432], [0.000192, 0.000768]]),
        )

    def test_max_marginal(self, markov_ve):
        _, infer = markov_ve
        np_test.assert_almost_equal(infer.max_marginal(), 0.1659, decimal=4)

    def test_max_marginal_var(self, markov_ve):
        _, infer = markov_ve
        np_test.assert_almost_equal(infer.max_marginal(["G"]), 0.1659, decimal=4)

    def test_max_marginal_var1(self, markov_ve):
        _, infer = markov_ve
        np_test.assert_almost_equal(infer.max_marginal(["G", "R"]), 0.1659, decimal=4)

    def test_max_marginal_var2(self, markov_ve):
        _, infer = markov_ve
        np_test.assert_almost_equal(infer.max_marginal(["G", "R", "A"]), 0.1659, decimal=4)

    def test_map_query(self, markov_ve):
        _, infer = markov_ve
        map_query = infer.map_query(show_progress=False)
        assert map_query == {"A": 1, "R": 1, "J": 1, "Q": 1, "G": 0, "L": 0}

    def test_map_query_with_evidence(self, markov_ve):
        _, infer = markov_ve
        map_query = infer.map_query(["A", "R", "L"], {"J": 0, "Q": 1, "G": 0}, show_progress=False)
        assert map_query == {"A": 1, "R": 0, "L": 0}

    def test_induced_graph(self, markov_ve):
        _, infer = markov_ve
        induced_graph = infer.induced_graph(["G", "Q", "A", "J", "L", "R"])
        result_edges = sorted([sorted(x) for x in induced_graph.edges()])
        assert [
            ["A", "J"],
            ["A", "R"],
            ["G", "J"],
            ["G", "L"],
            ["J", "L"],
            ["J", "Q"],
            ["J", "R"],
            ["L", "R"],
        ] == result_edges

    def test_induced_width(self, markov_ve):
        _, infer = markov_ve
        result_width = infer.induced_width(["G", "Q", "A", "J", "L", "R"])
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
            infer.query(["Y"], evidence={"X": 0}, show_progress=False).values,
            [0.35, 0.65],
        )


@requires_torch
class TestVariableEliminationLinearGaussianAndFunctionalBayesian:
    def test_query_linear_gaussian(self, torch_models):
        inference = VariableElimination(torch_models["lgbm"])
        with pytest.raises(
            NotImplementedError,
            match="Variable Elimination is not supported for LinearGaussianBayesianNetwork.",
        ):
            inference.query(["Y"])

    def test_query_functional_bayesian(self, torch_models):
        inference = VariableElimination(torch_models["fbn"])
        with pytest.raises(
            NotImplementedError,
            match="Variable Elimination is not supported for FunctionalBayesianNetwork.",
        ):
            inference.query(["Y"])
