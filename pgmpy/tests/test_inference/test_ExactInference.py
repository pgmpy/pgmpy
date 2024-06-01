import itertools
import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.inference.ExactInference import BeliefPropagationWithMessagePassing
from pgmpy.models import BayesianNetwork, FactorGraph, JunctionTree, MarkovNetwork


class TestVariableElimination(unittest.TestCase):
    def setUp(self):
        self.bayesian_model = BayesianNetwork(
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
        self.bayesian_model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)

        self.bayesian_inference = VariableElimination(self.bayesian_model)

    # All the values that are used for comparison in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = self.bayesian_inference.query(
                ["J"], elimination_order=order, show_progress=False
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(variables=["J"], cardinality=[2], values=[0.416, 0.584]),
            )

    def test_query_multiple_variable(self):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = self.bayesian_inference.query(
                ["Q", "J"], elimination_order=order, show_progress=False
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(
                    variables=["J", "Q"],
                    cardinality=[2, 2],
                    values=np.array([[0.3744, 0.0416], [0.1168, 0.4672]]),
                ),
            )

    def test_query_single_variable_with_evidence(self):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = self.bayesian_inference.query(
                variables=["J"],
                evidence={"A": 0, "R": 1},
                elimination_order=order,
                show_progress=False,
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(variables=["J"], cardinality=[2], values=[0.6, 0.4]),
            )

    def test_query_multiple_variable_with_evidence(self):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = self.bayesian_inference.query(
                variables=["J", "Q"],
                evidence={"A": 0, "R": 0, "G": 0, "L": 1},
                elimination_order=order,
                show_progress=False,
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(
                    variables=["J", "Q"],
                    cardinality=[2, 2],
                    values=np.array(
                        [[0.73636364, 0.08181818], [0.03636364, 0.14545455]]
                    ),
                ),
            )

    def test_query_multiple_times(self):
        # This just tests that the models are not getting modified while querying them
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            query_result = self.bayesian_inference.query(
                ["J"], elimination_order=order, show_progress=False
            )
            query_result = self.bayesian_inference.query(
                ["J"], elimination_order=order, show_progress=False
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(
                    variables=["J"], cardinality=[2], values=np.array([0.416, 0.584])
                ),
            )
            query_result = self.bayesian_inference.query(
                ["Q", "J"], elimination_order=order, show_progress=False
            )
            query_result = self.bayesian_inference.query(
                ["Q", "J"], elimination_order=order, show_progress=False
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(
                    variables=["J", "Q"],
                    cardinality=[2, 2],
                    values=np.array([[0.3744, 0.0416], [0.1168, 0.4672]]),
                ),
            )

            query_result = self.bayesian_inference.query(
                variables=["J"],
                evidence={"A": 0, "R": 1},
                elimination_order=order,
                show_progress=False,
            )
            query_result = self.bayesian_inference.query(
                variables=["J"],
                evidence={"A": 0, "R": 1},
                elimination_order=order,
                show_progress=False,
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(variables=["J"], cardinality=[2], values=[0.6, 0.4]),
            )

            query_result = self.bayesian_inference.query(
                variables=["J", "Q"],
                evidence={"A": 0, "R": 0, "G": 0, "L": 1},
                elimination_order=order,
                show_progress=False,
            )
            query_result = self.bayesian_inference.query(
                variables=["J", "Q"],
                evidence={"A": 0, "R": 0, "G": 0, "L": 1},
                elimination_order=order,
                show_progress=False,
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(
                    variables=["J", "Q"],
                    cardinality=[2, 2],
                    values=np.array(
                        [[0.73636364, 0.08181818], [0.03636364, 0.14545455]]
                    ),
                ),
            )

    def test_query_common_var(self):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            self.assertRaises(
                ValueError,
                self.bayesian_inference.query,
                variables=["J"],
                evidence=["J"],
                elimination_order=order,
            )

    def test_max_marginal(self):
        np_test.assert_almost_equal(
            self.bayesian_inference.max_marginal(), 0.1659, decimal=4
        )

    def test_max_marginal_var(self):
        np_test.assert_almost_equal(
            self.bayesian_inference.max_marginal(["G"]), 0.6, decimal=4
        )

    def test_max_marginal_var1(self):
        np_test.assert_almost_equal(
            self.bayesian_inference.max_marginal(["G", "R"]), 0.36, decimal=4
        )

    def test_max_marginal_var2(self):
        np_test.assert_almost_equal(
            self.bayesian_inference.max_marginal(["G", "R", "A"]), 0.288, decimal=4
        )

    def test_max_marginal_common_var(self):
        self.assertRaises(
            ValueError,
            self.bayesian_inference.max_marginal,
            variables=["J"],
            evidence=["J"],
        )

    def test_map_query(self):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            map_query = self.bayesian_inference.map_query(
                elimination_order=order, show_progress=False
            )
            self.assertDictEqual(
                map_query, {"A": 1, "R": 1, "J": 1, "Q": 1, "G": 0, "L": 0}
            )

    def test_map_query_with_evidence(self):
        map_query = self.bayesian_inference.map_query(
            ["A", "R", "L"], {"J": 0, "Q": 1, "G": 0}, show_progress=False
        )
        self.assertDictEqual(map_query, {"A": 1, "R": 0, "L": 0})

    def test_map_query_common_var(self):
        for order in [
            "greedy",
            "MinFill",
            "MinNeighbors",
            "MinWeight",
            "WeightedMinFill",
        ]:
            self.assertRaises(
                ValueError,
                self.bayesian_inference.map_query,
                variables=["J"],
                evidence=["J"],
                elimination_order=order,
            )

    def test_elimination_order(self):
        # Check all the heuristics give the same results.
        for elimination_order in [
            "WeightedMinFill",
            "MinNeighbors",
            "MinWeight",
            "MinFill",
        ]:
            query_result = self.bayesian_inference.query(
                ["J"], elimination_order=elimination_order, show_progress=False
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(variables=["J"], cardinality=[2], values=[0.416, 0.584]),
            )

            query_result = self.bayesian_inference.query(
                variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
            )
            self.assertEqual(
                query_result,
                DiscreteFactor(variables=["J"], cardinality=[2], values=[0.6, 0.4]),
            )

        # Check when elimination order has extra variables. Because of pruning.
        query_result = self.bayesian_inference.query(
            ["J"], elimination_order=["A", "R", "L", "Q", "G"], show_progress=False
        )
        self.assertEqual(
            query_result,
            DiscreteFactor(variables=["J"], cardinality=[2], values=[0.416, 0.584]),
        )

        # Check for when elimination order doesn't have all the variables
        self.assertRaises(
            ValueError,
            self.bayesian_inference.query,
            variables=["J"],
            elimination_order=["A"],
        )

    def test_induced_graph(self):
        induced_graph = self.bayesian_inference.induced_graph(
            ["G", "Q", "A", "J", "L", "R"]
        )
        result_edges = sorted([sorted(x) for x in induced_graph.edges()])
        self.assertEqual(
            [
                ["A", "J"],
                ["A", "R"],
                ["G", "J"],
                ["G", "L"],
                ["J", "L"],
                ["J", "Q"],
                ["J", "R"],
                ["L", "R"],
            ],
            result_edges,
        )

    def test_induced_width(self):
        result_width = self.bayesian_inference.induced_width(
            ["G", "Q", "A", "J", "L", "R"]
        )
        self.assertEqual(2, result_width)

    def tearDown(self):
        del self.bayesian_inference
        del self.bayesian_model


class TestSnowNetwork(unittest.TestCase):
    def setUp(self):
        self.model = BayesianNetwork(
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
        self.model.add_cpds(cpd_snow, cpd_risk, cpd_traffic, cpd_late)

    def test_queries(self):
        for algo in [VariableElimination, BeliefPropagation]:
            infer = algo(self.model)
            query1 = infer.query(
                ["Snow"], evidence={"Traffic": "slow"}, show_progress=False
            )
            np_test.assert_array_almost_equal(query1.values, [0.533333, 0.466667])

            query2 = infer.query(
                ["Risk"], evidence={"Traffic": "slow"}, show_progress=False
            )
            np_test.assert_array_almost_equal(query2.values, [0.613333, 0.386667])

            query3 = infer.query(
                ["Late"], evidence={"Traffic": "slow"}, show_progress=False
            )
            np_test.assert_array_almost_equal(query3.values, [0.7920, 0.2080])

            self.assertRaises(
                ValueError,
                infer.query,
                variables=["Traffic"],
                evidence={"Traffic": "slow"},
            )

    def test_elimination_order(self):
        infer = VariableElimination(self.model)
        for order in ["MinFill", "MinNeighbors", "MinWeight", "WeightedMinFill"]:
            computed_order = infer._get_elimination_order(
                variables=["Traffic"], evidence={}, elimination_order=order
            )
            self.assertEqual(set(computed_order), set({"Risk", "Late", "Snow"}))

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

    def test_joint_distribution(self):
        infer = VariableElimination(self.model)
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
                self.assertEqual(query_joint[var], query_expected[var])

    def test_virt_evidence(self):
        virt_evidence_cpd = TabularCPD(
            "Traffic", 2, [[0.3], [0.7]], state_names={"Traffic": ["normal", "slow"]}
        )
        virt_evidence_factor = DiscreteFactor(
            ["Traffic"], [2], [0.3, 0.7], state_names={"Traffic": ["normal", "slow"]}
        )
        for virt_evidence in [virt_evidence_cpd, virt_evidence_factor]:
            for algo in [VariableElimination, BeliefPropagation]:
                infer = algo(self.model)
                query1 = infer.query(
                    ["Snow"], virtual_evidence=[virt_evidence], show_progress=False
                )
                np_test.assert_array_almost_equal(query1.values, [0.45, 0.55])

                map1 = infer.map_query(
                    ["Snow"], virtual_evidence=[virt_evidence], show_progress=False
                )
                self.assertEqual(map1, {"Snow": "no"})

                query2 = infer.query(
                    ["Risk"], virtual_evidence=[virt_evidence], show_progress=False
                )
                np_test.assert_array_almost_equal(query2.values, [0.58, 0.42])

                map2 = infer.map_query(
                    ["Risk"], virtual_evidence=[virt_evidence], show_progress=False
                )
                self.assertEqual(map2, {"Risk": "yes"})

                query3 = infer.query(
                    ["Late"], virtual_evidence=[virt_evidence], show_progress=False
                )
                np_test.assert_array_almost_equal(query3.values, [0.61625, 0.38375])

                map3 = infer.map_query(
                    ["Late"], virtual_evidence=[virt_evidence], show_progress=False
                )
                self.assertEqual(map3, {"Late": "yes"})

                query4 = infer.query(
                    ["Traffic"], virtual_evidence=[virt_evidence], show_progress=False
                )
                np_test.assert_array_almost_equal(query4.values, [0.34375, 0.65625])

                # TODO: State name should be returned here.
                map4 = infer.map_query(
                    ["Traffic"], virtual_evidence=[virt_evidence], show_progress=False
                )
                self.assertTrue(map4 in [{"Traffic": "slow"}, {"Traffic": 1}])

        virt_evidence1_cpd = TabularCPD(
            "Risk", 2, [[0.7], [0.3]], state_names={"Risk": ["yes", "no"]}
        )
        virt_evidence1_factor = DiscreteFactor(
            ["Risk"], [2], [0.7, 0.3], state_names={"Risk": ["yes", "no"]}
        )
        for virt_evidence in [virt_evidence_cpd, virt_evidence_factor]:
            for virt_evidence1 in [virt_evidence1_cpd, virt_evidence1_factor]:
                for algo in [VariableElimination, BeliefPropagation]:
                    infer = algo(self.model)
                    query1 = infer.query(
                        ["Snow"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    np_test.assert_array_almost_equal(
                        query1.values, [0.52443609, 0.47556391]
                    )

                    map1 = infer.map_query(
                        ["Snow"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    self.assertEqual(map1, {"Snow": "yes"})

                    query2 = infer.query(
                        ["Risk"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    np_test.assert_array_almost_equal(
                        query2.values, [0.76315789, 0.23684211]
                    )
                    map2 = infer.map_query(
                        ["Risk"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    self.assertTrue(map2 in [{"Risk": 0}, {"Risk": "yes"}])

                    query3 = infer.query(
                        ["Traffic"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    np_test.assert_array_almost_equal(
                        query3.values, [0.32730263, 0.67269737]
                    )
                    map3 = infer.map_query(
                        ["Traffic"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    self.assertTrue(map3 in [{"Traffic": "slow"}, {"Traffic": 1}])

                    query4 = infer.query(
                        ["Late"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    np_test.assert_array_almost_equal(
                        query4.values, [0.66480263, 0.33519737]
                    )
                    map4 = infer.map_query(
                        ["Late"],
                        virtual_evidence=[virt_evidence, virt_evidence1],
                        show_progress=False,
                    )
                    self.assertEqual(map4, {"Late": "yes"})


class TestVariableEliminationDuplicatedFactors(unittest.TestCase):
    def setUp(self):
        self.markov_model = MarkovNetwork([("A", "B"), ("A", "C")])
        f1 = DiscreteFactor(
            variables=["A", "B"], cardinality=[2, 2], values=np.eye(2) * 2
        )
        f2 = DiscreteFactor(
            variables=["A", "C"], cardinality=[2, 2], values=np.eye(2) * 2
        )
        self.markov_model.add_factors(f1, f2)
        self.markov_inference = VariableElimination(self.markov_model)

    def test_duplicated_factors(self):
        query_result = self.markov_inference.query(["A"], show_progress=False)
        self.assertEqual(
            query_result,
            DiscreteFactor(variables=["A"], cardinality=[2], values=np.array([4, 4])),
        )


class TestVariableEliminationMarkov(unittest.TestCase):
    def setUp(self):
        # It is just a moralised version of the above Bayesian network so all the results are same. Only factors
        # are under consideration for inference so this should be fine.
        self.markov_model = MarkovNetwork(
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

        self.markov_model.add_factors(
            factor_a, factor_r, factor_j, factor_q, factor_l, factor_g
        )
        self.markov_inference = VariableElimination(self.markov_model)

    # All the values that are used for comparison in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self):
        query_result = self.markov_inference.query(["J"], show_progress=False)
        self.assertEqual(
            query_result,
            DiscreteFactor(
                variables=["J"], cardinality=[2], values=np.array([0.416, 0.584])
            ),
        )

    def test_query_multiple_variable(self):
        query_result = self.markov_inference.query(["Q", "J"], show_progress=False)
        self.assertEqual(
            query_result,
            DiscreteFactor(
                variables=["Q", "J"],
                cardinality=[2, 2],
                values=np.array([[0.3744, 0.1168], [0.0416, 0.4672]]),
            ),
        )

    def test_query_single_variable_with_evidence(self):
        query_result = self.markov_inference.query(
            variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
        )
        self.assertEqual(
            query_result,
            DiscreteFactor(variables=["J"], cardinality=[2], values=[0.072, 0.048]),
        )

    def test_query_multiple_variable_with_evidence(self):
        query_result = self.markov_inference.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        self.assertEqual(
            query_result,
            DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.003888, 0.000432], [0.000192, 0.000768]]),
            ),
        )

    def test_query_multiple_times(self):
        # This just tests that the models are not getting modified while querying them
        query_result = self.markov_inference.query(["J"], show_progress=False)
        query_result = self.markov_inference.query(["J"], show_progress=False)
        self.assertEqual(
            query_result,
            DiscreteFactor(
                variables=["J"], cardinality=[2], values=np.array([0.416, 0.584])
            ),
        )

        query_result = self.markov_inference.query(["Q", "J"], show_progress=False)
        query_result = self.markov_inference.query(["Q", "J"], show_progress=False)
        self.assertEqual(
            query_result,
            DiscreteFactor(
                variables=["Q", "J"],
                cardinality=[2, 2],
                values=np.array([[0.3744, 0.1168], [0.0416, 0.4672]]),
            ),
        )

        query_result = self.markov_inference.query(
            variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
        )
        query_result = self.markov_inference.query(
            variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
        )
        self.assertEqual(
            query_result,
            DiscreteFactor(variables=["J"], cardinality=[2], values=[0.072, 0.048]),
        )

        query_result = self.markov_inference.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        query_result = self.markov_inference.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        self.assertEqual(
            query_result,
            DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.003888, 0.000432], [0.000192, 0.000768]]),
            ),
        )

    def test_max_marginal(self):
        np_test.assert_almost_equal(
            self.markov_inference.max_marginal(), 0.1659, decimal=4
        )

    def test_max_marginal_var(self):
        np_test.assert_almost_equal(
            self.markov_inference.max_marginal(["G"]), 0.1659, decimal=4
        )

    def test_max_marginal_var1(self):
        np_test.assert_almost_equal(
            self.markov_inference.max_marginal(["G", "R"]), 0.1659, decimal=4
        )

    def test_max_marginal_var2(self):
        np_test.assert_almost_equal(
            self.markov_inference.max_marginal(["G", "R", "A"]), 0.1659, decimal=4
        )

    def test_map_query(self):
        map_query = self.markov_inference.map_query(show_progress=False)
        self.assertDictEqual(
            map_query, {"A": 1, "R": 1, "J": 1, "Q": 1, "G": 0, "L": 0}
        )

    def test_map_query_with_evidence(self):
        map_query = self.markov_inference.map_query(
            ["A", "R", "L"], {"J": 0, "Q": 1, "G": 0}, show_progress=False
        )
        self.assertDictEqual(map_query, {"A": 1, "R": 0, "L": 0})

    def test_induced_graph(self):
        induced_graph = self.markov_inference.induced_graph(
            ["G", "Q", "A", "J", "L", "R"]
        )
        result_edges = sorted([sorted(x) for x in induced_graph.edges()])
        self.assertEqual(
            [
                ["A", "J"],
                ["A", "R"],
                ["G", "J"],
                ["G", "L"],
                ["J", "L"],
                ["J", "Q"],
                ["J", "R"],
                ["L", "R"],
            ],
            result_edges,
        )

    def test_induced_width(self):
        result_width = self.markov_inference.induced_width(
            ["G", "Q", "A", "J", "L", "R"]
        )
        self.assertEqual(2, result_width)

    def test_issue_1421(self):
        model = BayesianNetwork([("X", "Y"), ("Z", "X"), ("W", "Y")])
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

    def tearDown(self):
        del self.markov_inference
        del self.markov_model


class TestBeliefPropagation(unittest.TestCase):
    def setUp(self):
        self.junction_tree = JunctionTree(
            [(("A", "B"), ("B", "C")), (("B", "C"), ("C", "D"))]
        )
        phi1 = DiscreteFactor(["A", "B"], [2, 3], range(6))
        phi2 = DiscreteFactor(["B", "C"], [3, 2], range(6))
        phi3 = DiscreteFactor(["C", "D"], [2, 2], range(4))
        self.junction_tree.add_factors(phi1, phi2, phi3)

        self.bayesian_model = BayesianNetwork(
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
        self.bayesian_model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)

    def test_calibrate_clique_belief(self):
        belief_propagation = BeliefPropagation(self.junction_tree)
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

        self.assertEqual(clique_belief[("A", "B")], b_A_B)
        self.assertEqual(clique_belief[("B", "C")], b_B_C)
        self.assertEqual(clique_belief[("C", "D")], b_C_D)

    def test_calibrate_sepset_belief(self):
        belief_propagation = BeliefPropagation(self.junction_tree)
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

        np_test.assert_array_almost_equal(
            sepset_belief[frozenset((("A", "B"), ("B", "C")))].values, b_B.values
        )
        np_test.assert_array_almost_equal(
            sepset_belief[frozenset((("B", "C"), ("C", "D")))].values, b_C.values
        )

    def test_max_calibrate_clique_belief(self):
        belief_propagation = BeliefPropagation(self.junction_tree)
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

        self.assertEqual(clique_belief[("A", "B")], b_A_B)
        self.assertEqual(clique_belief[("B", "C")], b_B_C)
        self.assertEqual(clique_belief[("C", "D")], b_C_D)

    def test_max_calibrate_sepset_belief(self):
        belief_propagation = BeliefPropagation(self.junction_tree)
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

        np_test.assert_array_almost_equal(
            sepset_belief[frozenset((("A", "B"), ("B", "C")))].values, b_B.values
        )
        np_test.assert_array_almost_equal(
            sepset_belief[frozenset((("B", "C"), ("C", "D")))].values, b_C.values
        )

    # All the values that are used for comparison in the all the tests are
    # found using SAMIAM (assuming that it is correct ;))

    def test_query_single_variable(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        query_result = belief_propagation.query(["J"], show_progress=False)
        self.assertEqual(
            query_result,
            DiscreteFactor(variables=["J"], cardinality=[2], values=[0.416, 0.584]),
        )

    def test_query_multiple_variable(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        query_result = belief_propagation.query(["Q", "J"], show_progress=False)
        self.assertEqual(
            query_result,
            DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.3744, 0.0416], [0.1168, 0.4672]]),
            ),
        )

    def test_query_single_variable_with_evidence(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        query_result = belief_propagation.query(
            variables=["J"], evidence={"A": 0, "R": 1}, show_progress=False
        )
        self.assertEqual(
            query_result,
            DiscreteFactor(
                variables=["J"], cardinality=[2], values=np.array([0.6, 0.4])
            ),
        )

    def test_query_multiple_variable_with_evidence(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        query_result = belief_propagation.query(
            variables=["J", "Q"],
            evidence={"A": 0, "R": 0, "G": 0, "L": 1},
            show_progress=False,
        )
        self.assertEqual(
            query_result,
            DiscreteFactor(
                variables=["J", "Q"],
                cardinality=[2, 2],
                values=np.array([[0.73636364, 0.08181818], [0.03636364, 0.14545455]]),
            ),
        )

    def test_query_common_var(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        self.assertRaises(
            ValueError, belief_propagation.query, variables=["J"], evidence=["J"]
        )

    def test_map_query(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        map_query = belief_propagation.map_query(show_progress=False)
        self.assertDictEqual(
            map_query, {"A": 1, "R": 1, "J": 1, "Q": 1, "G": 0, "L": 0}
        )

    def test_map_query_with_evidence(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        map_query = belief_propagation.map_query(
            ["A", "R", "L"], {"J": 0, "Q": 1, "G": 0}, show_progress=False
        )
        self.assertDictEqual(map_query, {"A": 1, "R": 0, "L": 0})

    def test_map_query_common_var(self):
        belief_propagation = BeliefPropagation(self.bayesian_model)
        self.assertRaises(
            ValueError, belief_propagation.map_query, variables=["J"], evidence=["J"]
        )

    def test_issue_1048(self):
        model = BayesianNetwork()

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
            self.assertEqual(evidence, expected_evidences[i])
            np_test.assert_almost_equal(
                inf.query(["parent"], evidence, show_progress=False)
                .normalize(inplace=False)
                .values,
                expected_values[i],
                decimal=2,
            )
            evidence.update({c: 1})

    def tearDown(self):
        del self.junction_tree
        del self.bayesian_model


class TestBeliefPropagationWithMessagePassing(unittest.TestCase):
    def setUp(self):
        self.factor_graph = FactorGraph()
        self.factor_graph.add_nodes_from(["A", "B", "C", "D"])

        phi1 = DiscreteFactor(["A"], [2], [0.4, 0.6])
        phi2 = DiscreteFactor(
            ["B", "A"], [3, 2], [[0.2, 0.05], [0.3, 0.15], [0.5, 0.8]]
        )
        phi3 = DiscreteFactor(["C", "B"], [2, 3], [[0.4, 0.5, 0.1], [0.6, 0.5, 0.9]])
        phi4 = DiscreteFactor(
            ["D", "B"], [3, 3], [[0.1, 0.1, 0.2], [0.3, 0.2, 0.1], [0.6, 0.7, 0.7]]
        )

        self.factor_graph.add_factors(phi1, phi2, phi3, phi4)

        self.factor_graph.add_edges_from(
            [
                (phi1, "A"),
                ("A", phi2),
                (phi2, "B"),
                ("B", phi3),
                (phi3, "C"),
                ("B", phi4),
                (phi4, "D"),
            ]
        )

        self.belief_propagation = BeliefPropagationWithMessagePassing(self.factor_graph)

    def test_query_single_variable(self):
        res = self.belief_propagation.query(["C"])
        assert np.allclose(res["C"].values, np.array([0.217, 0.783]), atol=1e-20)

    def test_query_multiple_variable(self):
        res = self.belief_propagation.query(["A", "B", "C", "D"])
        assert np.allclose(res["A"].values, np.array([0.4, 0.6]), atol=1e-20)
        assert np.allclose(res["B"].values, np.array([0.11, 0.21, 0.68]), atol=1e-20)
        assert np.allclose(res["C"].values, np.array([0.217, 0.783]), atol=1e-20)
        assert np.allclose(res["D"].values, np.array([0.168, 0.143, 0.689]), atol=1e-20)

    def test_query_single_variable_with_evidence(self):
        res = self.belief_propagation.query(["B", "C"], {"A": 1, "D": 0})
        assert np.allclose(
            res["B"].values, np.array([0.02777778, 0.08333333, 0.88888889]), atol=1e-20
        )
        assert np.allclose(
            res["C"].values, np.array([0.14166667, 0.85833333]), atol=1e-20
        )

    def test_query_multiple_variable_with_evidence(self):
        res = self.belief_propagation.query(["B", "C"], {"A": 1, "D": 0})
        assert np.allclose(
            res["B"].values, np.array([0.02777778, 0.08333333, 0.88888889]), atol=1e-20
        )
        assert np.allclose(
            res["C"].values, np.array([0.14166667, 0.85833333]), atol=1e-20
        )

    def test_query_single_variable_with_virtual_evidence(self):
        ve = [TabularCPD("A", 2, [[0.1], [0.9]])]
        res = self.belief_propagation.query(["B"], virtual_evidence=ve)
        assert np.allclose(
            res["B"].values, np.array([0.06034483, 0.16034483, 0.77931034]), atol=1e-20
        )

    def test_query_multiple_variable_with_multiple_evidence_and_virtual_evidence(self):
        ve = [
            TabularCPD("A", 2, [[0.027], [0.972]]),
            TabularCPD("B", 3, [[0.3], [0.6], [0.1]]),
        ]
        res = self.belief_propagation.query(
            ["B", "C"], evidence={"D": 0}, virtual_evidence=ve
        )
        assert np.allclose(
            res["B"].values, np.array([0.05938567, 0.3440273, 0.59658703]), atol=1e-20
        )
        assert np.allclose(
            res["C"].values, np.array([0.25542662, 0.74457338]), atol=1e-20
        )

    def test_query_allows_multiple_virtual_evidence_per_variable(self):
        ve1 = [
            TabularCPD("A", 2, [[0.1], [0.9]]),
            TabularCPD("A", 2, [[0.3], [0.7]]),
        ]
        res1 = self.belief_propagation.query(["B"], virtual_evidence=ve1)
        cpd = TabularCPD("A", 2, [[0.1 * 0.3], [0.9 * 0.7]])
        cpd.normalize()
        res2 = self.belief_propagation.query(["B"], virtual_evidence=[cpd])
        assert np.allclose(res1["B"].values, res2["B"].values, atol=1e-20)
        assert np.allclose(
            res2["B"].values, np.array([0.05461538, 0.15461538, 0.79076923]), atol=1e-20
        )

    def test_query_error_obs_var_has_evidence(self):
        with self.assertRaises(
            ValueError,
            msg="Can't have the same variables in both `evidence` and `virtual_evidence`. Found in both: {'A'}",
        ):
            self.belief_propagation.query(
                ["B"], evidence={"A": 1}, virtual_evidence={"A": [np.array([0.1, 0.9])]}
            )

    def test_query_single_variable_can_return_all_computed_messages(self):
        res, messages = self.belief_propagation.query(["B"], get_messages=True)
        assert np.allclose(res["B"].values, np.array([0.11, 0.21, 0.68]), atol=1e-20)
        # Assert on messages values
        assert np.allclose(messages["['A'] -> A"], np.array([0.4, 0.6]), atol=1e-20)
        assert np.allclose(
            messages["['B', 'A'] -> B"], np.array([0.11, 0.21, 0.68]), atol=1e-20
        )
        assert np.allclose(
            messages["['C', 'B'] -> B"],
            np.array([0.33333333, 0.33333333, 0.33333333]),
            atol=1e-20,
        )
        assert np.allclose(
            messages["['D', 'B'] -> B"],
            np.array([0.33333333, 0.33333333, 0.33333333]),
            atol=1e-20,
        )

    def test_query_multiple_variable_returns_each_message_once(self):
        res, messages = self.belief_propagation.query(["C", "B"], get_messages=True)
        assert np.allclose(res["B"].values, np.array([0.11, 0.21, 0.68]), atol=1e-20)
        assert np.allclose(res["C"].values, np.array([0.217, 0.783]), atol=1e-20)

        # Message common to both B and C
        assert np.allclose(messages["['A'] -> A"], np.array([0.4, 0.6]), atol=1e-20)
        assert np.allclose(
            messages["['B', 'A'] -> B"], np.array([0.11, 0.21, 0.68]), atol=1e-20
        )

        # Message specific to B
        assert np.allclose(
            messages["['C', 'B'] -> B"],
            np.array([0.33333333, 0.33333333, 0.33333333]),
            atol=1e-20,
        )
        assert np.allclose(
            messages["['D', 'B'] -> B"],
            np.array([0.33333333, 0.33333333, 0.33333333]),
            atol=1e-20,
        )

        # Messages specific to C
        assert np.allclose(
            messages["['C', 'B'] -> C"], np.array([0.217, 0.783]), atol=1e-20
        )
