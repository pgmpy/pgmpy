import unittest

import numpy as np
from mock import MagicMock, patch

from pgmpy.factors.discrete import DiscreteFactor, State, TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork, MarkovNetwork
from pgmpy.sampling import BayesianModelSampling, GibbsSampling
from pgmpy.sampling.base import BayesianModelInference


class TestBayesianModelSampling(unittest.TestCase):
    def setUp(self):
        # Bayesian Model without state names
        self.bayesian_model = BayesianNetwork(
            [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")]
        )
        cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
        cpd_r = TabularCPD("R", 2, [[0.4], [0.6]])
        cpd_j = TabularCPD(
            "J", 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], ["R", "A"], [2, 2]
        )
        cpd_q = TabularCPD("Q", 2, [[0.9, 0.2], [0.1, 0.8]], ["J"], [2])
        cpd_l = TabularCPD(
            "L", 2, [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]], ["G", "J"], [2, 2]
        )
        cpd_g = TabularCPD("G", 2, [[0.6], [0.4]])
        self.bayesian_model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
        self.sampling_inference = BayesianModelSampling(self.bayesian_model)
        self.forward_marginals = VariableElimination(self.bayesian_model).query(
            self.bayesian_model.nodes(), joint=False, show_progress=False
        )

        # Bayesian Model without state names and with latent variables
        self.bayesian_model_lat = BayesianNetwork(
            [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")],
            latents=["R", "Q"],
        )
        cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
        cpd_r = TabularCPD("R", 2, [[0.4], [0.6]])
        cpd_j = TabularCPD(
            "J", 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], ["R", "A"], [2, 2]
        )
        cpd_q = TabularCPD("Q", 2, [[0.9, 0.2], [0.1, 0.8]], ["J"], [2])
        cpd_l = TabularCPD(
            "L", 2, [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]], ["G", "J"], [2, 2]
        )
        cpd_g = TabularCPD("G", 2, [[0.6], [0.4]])
        self.bayesian_model_lat.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
        self.sampling_inference_lat = BayesianModelSampling(self.bayesian_model_lat)

        # Bayesian Model with state names
        self.bayesian_model_names = BayesianNetwork(
            [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")]
        )
        cpd_a_names = TabularCPD(
            "A", 2, [[0.2], [0.8]], state_names={"A": ["a0", "a1"]}
        )
        cpd_r_names = TabularCPD(
            "R", 2, [[0.4], [0.6]], state_names={"R": ["r0", "r1"]}
        )
        cpd_j_names = TabularCPD(
            "J",
            2,
            [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
            ["R", "A"],
            [2, 2],
            state_names={"J": ["j0", "j1"], "R": ["r0", "r1"], "A": ["a0", "a1"]},
        )
        cpd_q_names = TabularCPD(
            "Q",
            2,
            [[0.9, 0.2], [0.1, 0.8]],
            ["J"],
            [2],
            state_names={"Q": ["q0", "q1"], "J": ["j0", "j1"]},
        )
        cpd_l_names = TabularCPD(
            "L",
            2,
            [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
            ["G", "J"],
            [2, 2],
            state_names={"L": ["l0", "l1"], "G": ["g0", "g1"], "J": ["j0", "j1"]},
        )
        cpd_g_names = TabularCPD(
            "G", 2, [[0.6], [0.4]], state_names={"G": ["g0", "g1"]}
        )
        self.bayesian_model_names.add_cpds(
            cpd_a_names, cpd_g_names, cpd_j_names, cpd_l_names, cpd_q_names, cpd_r_names
        )

        self.sampling_inference_names = BayesianModelSampling(self.bayesian_model_names)

        # Bayesian Model with state names and with latent variables
        self.bayesian_model_names_lat = BayesianNetwork(
            [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")],
            latents=["R", "Q"],
        )
        cpd_a_names = TabularCPD(
            "A", 2, [[0.2], [0.8]], state_names={"A": ["a0", "a1"]}
        )
        cpd_r_names = TabularCPD(
            "R", 2, [[0.4], [0.6]], state_names={"R": ["r0", "r1"]}
        )
        cpd_j_names = TabularCPD(
            "J",
            2,
            [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
            ["R", "A"],
            [2, 2],
            state_names={"J": ["j0", "j1"], "R": ["r0", "r1"], "A": ["a0", "a1"]},
        )
        cpd_q_names = TabularCPD(
            "Q",
            2,
            [[0.9, 0.2], [0.1, 0.8]],
            ["J"],
            [2],
            state_names={"Q": ["q0", "q1"], "J": ["j0", "j1"]},
        )
        cpd_l_names = TabularCPD(
            "L",
            2,
            [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
            ["G", "J"],
            [2, 2],
            state_names={"L": ["l0", "l1"], "G": ["g0", "g1"], "J": ["j0", "j1"]},
        )
        cpd_g_names = TabularCPD(
            "G", 2, [[0.6], [0.4]], state_names={"G": ["g0", "g1"]}
        )
        self.bayesian_model_names_lat.add_cpds(
            cpd_a_names, cpd_g_names, cpd_j_names, cpd_l_names, cpd_q_names, cpd_r_names
        )

        self.sampling_inference_names_lat = BayesianModelSampling(
            self.bayesian_model_names_lat
        )

        self.markov_model = MarkovNetwork()

    def test_init(self):
        with self.assertRaises(TypeError):
            BayesianModelSampling(self.markov_model)

    def test_pre_compute_reduce_maps(self):
        base_infer = BayesianModelInference(self.bayesian_model)
        state_to_index, index_to_weight = base_infer.pre_compute_reduce_maps(
            "J", ["A", "R"], [(1, 1), (1, 0)]
        )
        self.assertEqual(state_to_index[(1, 1)], 0)
        self.assertEqual(state_to_index[(1, 0)], 1)
        self.assertEqual(list(index_to_weight[0]), [0.1, 0.9])
        self.assertEqual(list(index_to_weight[1]), [0.6, 0.4])

        # Make sure the order of the evidence variables doen't matter
        state_to_index, index_to_weight = base_infer.pre_compute_reduce_maps(
            "J", ["R", "A"], [(1, 1), (1, 0)]
        )
        self.assertEqual(state_to_index[(1, 1)], 0)
        self.assertEqual(state_to_index[(1, 0)], 1)
        self.assertEqual(list(index_to_weight[0]), [0.1, 0.9])
        self.assertEqual(list(index_to_weight[1]), [0.7, 0.3])

    def test_pred_compute_reduce_maps_partial_evidence(self):
        base_infer = BayesianModelInference(self.bayesian_model)
        state_to_index, index_to_weight = base_infer.pre_compute_reduce_maps(
            "J", ["A"], [(1,), (0,)]
        )
        self.assertEqual(state_to_index[(1,)], 0)
        self.assertEqual(state_to_index[(0,)], 1)
        self.assertEqual(list(index_to_weight[0].round(2)), [0.35, 0.65])
        self.assertEqual(list(index_to_weight[1].round(2)), [0.8, 0.2])

        # Make sure the order of the evidence variables doen't matter
        state_to_index, index_to_weight = base_infer.pre_compute_reduce_maps(
            "J", ["R"], [(1,), (0,)]
        )
        self.assertEqual(state_to_index[(1,)], 0)
        self.assertEqual(state_to_index[(0,)], 1)
        self.assertEqual(list(index_to_weight[0].round(2)), [0.4, 0.6])
        self.assertEqual(list(index_to_weight[1].round(2)), [0.75, 0.25])

    def test_forward_sample(self):
        # Test without state names
        sample = self.sampling_inference.forward_sample(int(1e5))
        self.assertEqual(len(sample), int(1e5))
        self.assertEqual(len(sample.columns), 6)
        self.assertIn("A", sample.columns)
        self.assertIn("J", sample.columns)
        self.assertIn("R", sample.columns)
        self.assertIn("Q", sample.columns)
        self.assertIn("G", sample.columns)
        self.assertIn("L", sample.columns)
        self.assertTrue(set(sample.A).issubset({0, 1}))
        self.assertTrue(set(sample.J).issubset({0, 1}))
        self.assertTrue(set(sample.R).issubset({0, 1}))
        self.assertTrue(set(sample.Q).issubset({0, 1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

        # Test that the marginal distribution of samples is same as the model
        sample_marginals = {
            node: sample[node].value_counts() / sample.shape[0]
            for node in self.bayesian_model.nodes()
        }

        for node in self.bayesian_model.nodes():
            for state in [0, 1]:
                self.assertEqual(
                    round(self.forward_marginals[node].get_value(**{node: state}), 1),
                    round(sample_marginals[node].loc[state], 1),
                )

        # Test without state names and with latents
        sample = self.sampling_inference_lat.forward_sample(25, include_latents=True)
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 6)
        self.assertEqual(set(sample.columns), {"A", "J", "R", "Q", "G", "L"})
        self.assertTrue(set(sample.A).issubset({0, 1}))
        self.assertTrue(set(sample.J).issubset({0, 1}))
        self.assertTrue(set(sample.R).issubset({0, 1}))
        self.assertTrue(set(sample.Q).issubset({0, 1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

        sample = self.sampling_inference_lat.forward_sample(25, include_latents=False)
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 4)
        self.assertFalse("R" in sample.columns)
        self.assertFalse("Q" in sample.columns)

        # Test with state names
        sample = self.sampling_inference_names.forward_sample(25)
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 6)
        self.assertIn("A", sample.columns)
        self.assertIn("J", sample.columns)
        self.assertIn("R", sample.columns)
        self.assertIn("Q", sample.columns)
        self.assertIn("G", sample.columns)
        self.assertIn("L", sample.columns)
        self.assertTrue(set(sample.A).issubset({"a0", "a1"}))
        self.assertTrue(set(sample.J).issubset({"j0", "j1"}))
        self.assertTrue(set(sample.R).issubset({"r0", "r1"}))
        self.assertTrue(set(sample.Q).issubset({"q0", "q1"}))
        self.assertTrue(set(sample.G).issubset({"g0", "g1"}))
        self.assertTrue(set(sample.L).issubset({"l0", "l1"}))

        # Test with state names and with latents
        sample = self.sampling_inference_names_lat.forward_sample(
            25, include_latents=True
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 6)
        self.assertEqual(set(sample.columns), {"A", "J", "R", "Q", "G", "L"})
        self.assertTrue(set(sample.A).issubset({"a0", "a1"}))
        self.assertTrue(set(sample.J).issubset({"j0", "j1"}))
        self.assertTrue(set(sample.R).issubset({"r0", "r1"}))
        self.assertTrue(set(sample.Q).issubset({"q0", "q1"}))
        self.assertTrue(set(sample.G).issubset({"g0", "g1"}))
        self.assertTrue(set(sample.L).issubset({"l0", "l1"}))

        sample = self.sampling_inference_names_lat.forward_sample(
            25, include_latents=False
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 4)
        self.assertFalse("R" in sample.columns)
        self.assertFalse("Q" in sample.columns)

    def test_rejection_sample_basic(self):
        # Test without state names
        sample = self.sampling_inference.rejection_sample()
        sample = self.sampling_inference.rejection_sample(
            [State("A", 1), State("J", 1), State("R", 1)], int(1e5)
        )
        self.assertEqual(len(sample), int(1e5))
        self.assertEqual(len(sample.columns), 6)
        self.assertEqual(set(sample.columns), {"A", "J", "R", "Q", "G", "L"})
        self.assertTrue(set(sample.A).issubset({1}))
        self.assertTrue(set(sample.J).issubset({1}))
        self.assertTrue(set(sample.R).issubset({1}))
        self.assertTrue(set(sample.Q).issubset({0, 1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

        # Test that the marginal distributions is the same in model and samples
        self.rejection_marginals = VariableElimination(self.bayesian_model).query(
            ["Q", "G", "L"],
            evidence={"A": 1, "J": 1, "R": 1},
            joint=False,
            show_progress=False,
        )

        sample_marginals = {
            node: sample[node].value_counts() / sample.shape[0]
            for node in ["Q", "G", "L"]
        }

        for node in ["Q", "G", "L"]:
            for state in [0, 1]:
                self.assertEqual(
                    round(self.rejection_marginals[node].get_value(**{node: state}), 1),
                    round(sample_marginals[node].loc[state], 1),
                )

        # Test without state names with latent variables
        sample = self.sampling_inference_lat.rejection_sample(
            [State("A", 1), State("J", 1), State("R", 1)], 25, include_latents=True
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 6)
        self.assertTrue(set(sample.A).issubset({1}))
        self.assertTrue(set(sample.J).issubset({1}))
        self.assertTrue(set(sample.R).issubset({1}))
        self.assertTrue(set(sample.Q).issubset({0, 1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

        sample = self.sampling_inference_lat.rejection_sample(
            [State("A", 1), State("J", 1), State("R", 1)], 25, include_latents=False
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 4)
        self.assertTrue(set(sample.A).issubset({1}))
        self.assertTrue(set(sample.J).issubset({1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

        # Test with state names
        sample = self.sampling_inference_names.rejection_sample()
        sample = self.sampling_inference_names.rejection_sample(
            [State("A", "a1"), State("J", "j1"), State("R", "r1")], 25
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 6)
        self.assertEqual(set(sample.columns), {"A", "J", "R", "Q", "G", "L"})
        self.assertTrue(set(sample.A).issubset({"a1"}))
        self.assertTrue(set(sample.J).issubset({"j1"}))
        self.assertTrue(set(sample.R).issubset({"r1"}))
        self.assertTrue(set(sample.Q).issubset({"q0", "q1"}))
        self.assertTrue(set(sample.G).issubset({"g0", "g1"}))
        self.assertTrue(set(sample.L).issubset({"l0", "l1"}))

        # Test with state names and latent variables
        sample = self.sampling_inference_names_lat.rejection_sample(
            [State("A", "a1"), State("J", "j1"), State("R", "r1")],
            25,
            include_latents=True,
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 6)
        self.assertEqual(set(sample.columns), {"A", "J", "R", "Q", "G", "L"})
        self.assertTrue(set(sample.A).issubset({"a1"}))
        self.assertTrue(set(sample.J).issubset({"j1"}))
        self.assertTrue(set(sample.R).issubset({"r1"}))
        self.assertTrue(set(sample.Q).issubset({"q0", "q1"}))
        self.assertTrue(set(sample.G).issubset({"g0", "g1"}))
        self.assertTrue(set(sample.L).issubset({"l0", "l1"}))

        sample = self.sampling_inference_names_lat.rejection_sample(
            [State("A", "a1"), State("J", "j1"), State("R", "r1")],
            25,
            include_latents=False,
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 4)
        self.assertEqual(set(sample.columns), {"A", "J", "G", "L"})
        self.assertTrue(set(sample.A).issubset({"a1"}))
        self.assertTrue(set(sample.J).issubset({"j1"}))
        self.assertTrue(set(sample.G).issubset({"g0", "g1"}))
        self.assertTrue(set(sample.L).issubset({"l0", "l1"}))

    def test_likelihood_weighted_sample(self):
        # Test without state names
        sample = self.sampling_inference.likelihood_weighted_sample()
        sample = self.sampling_inference.likelihood_weighted_sample(
            [State("A", 0), State("J", 1), State("R", 0)], 25
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 7)
        self.assertEqual(set(sample.columns), {"A", "J", "R", "Q", "G", "L", "_weight"})
        self.assertTrue(set(sample.A).issubset({0}))
        self.assertTrue(set(sample.J).issubset({1}))
        self.assertTrue(set(sample.R).issubset({0}))
        self.assertTrue(set(sample.Q).issubset({0, 1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

        # Test without state names and with latent variables
        sample = self.sampling_inference_lat.likelihood_weighted_sample(
            [State("A", 0), State("J", 1), State("R", 0)], 25, include_latents=True
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 7)
        self.assertEqual(set(sample.columns), {"A", "J", "R", "Q", "G", "L", "_weight"})
        self.assertTrue(set(sample.A).issubset({0}))
        self.assertTrue(set(sample.J).issubset({1}))
        self.assertTrue(set(sample.R).issubset({0}))
        self.assertTrue(set(sample.Q).issubset({0, 1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

        sample = self.sampling_inference_lat.likelihood_weighted_sample(
            [State("A", 0), State("J", 1), State("R", 0)], 25, include_latents=False
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 5)
        self.assertEqual(set(sample.columns), {"A", "J", "G", "L", "_weight"})
        self.assertTrue(set(sample.A).issubset({0}))
        self.assertTrue(set(sample.J).issubset({1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

        # Test with state names
        sample = self.sampling_inference_names.likelihood_weighted_sample()
        sample = self.sampling_inference_names.likelihood_weighted_sample(
            [State("A", "a0"), State("J", "j1"), State("R", "r0")], 25
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 7)
        self.assertEqual(set(sample.columns), {"A", "J", "R", "Q", "G", "L", "_weight"})
        self.assertTrue(set(sample.A).issubset({"a0"}))
        self.assertTrue(set(sample.J).issubset({"j1"}))
        self.assertTrue(set(sample.R).issubset({"r0"}))
        self.assertTrue(set(sample.Q).issubset({"q0", "q1"}))
        self.assertTrue(set(sample.G).issubset({"g0", "g1"}))
        self.assertTrue(set(sample.L).issubset({"l0", "l1"}))

        # Test with state names and with latent variables
        sample = self.sampling_inference_names_lat.likelihood_weighted_sample(
            [State("A", "a0"), State("J", "j1"), State("R", "r0")],
            25,
            include_latents=True,
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 7)
        self.assertEqual(set(sample.columns), {"A", "J", "R", "Q", "G", "L", "_weight"})
        self.assertTrue(set(sample.A).issubset({"a0"}))
        self.assertTrue(set(sample.J).issubset({"j1"}))
        self.assertTrue(set(sample.R).issubset({"r0"}))
        self.assertTrue(set(sample.Q).issubset({"q0", "q1"}))
        self.assertTrue(set(sample.G).issubset({"g0", "g1"}))
        self.assertTrue(set(sample.L).issubset({"l0", "l1"}))

        sample = self.sampling_inference_names_lat.likelihood_weighted_sample(
            [State("A", "a0"), State("J", "j1"), State("R", "r0")],
            25,
            include_latents=False,
        )
        self.assertEqual(len(sample), 25)
        self.assertEqual(len(sample.columns), 5)
        self.assertEqual(set(sample.columns), {"A", "J", "G", "L", "_weight"})
        self.assertTrue(set(sample.A).issubset({"a0"}))
        self.assertTrue(set(sample.J).issubset({"j1"}))
        self.assertTrue(set(sample.G).issubset({"g0", "g1"}))
        self.assertTrue(set(sample.L).issubset({"l0", "l1"}))

    def tearDown(self):
        del self.sampling_inference
        del self.bayesian_model
        del self.markov_model


class TestGibbsSampling(unittest.TestCase):
    def setUp(self):
        # A test Bayesian model
        diff_cpd = TabularCPD("diff", 2, [[0.6], [0.4]])
        intel_cpd = TabularCPD("intel", 2, [[0.7], [0.3]])
        grade_cpd = TabularCPD(
            "grade",
            3,
            [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
            evidence=["diff", "intel"],
            evidence_card=[2, 2],
        )
        self.bayesian_model = BayesianNetwork()
        self.bayesian_model.add_nodes_from(["diff", "intel", "grade"])
        self.bayesian_model.add_edges_from([("diff", "grade"), ("intel", "grade")])
        self.bayesian_model.add_cpds(diff_cpd, intel_cpd, grade_cpd)

        # Another test Bayesian model
        cpt_cloudy = TabularCPD("Cloudy", 2, [[0.5], [0.5]])
        cpt_sprinkler = TabularCPD(
            "Sprinkler",
            2,
            [[0.5, 0.9], [0.5, 0.1]],
            evidence=["Cloudy"],
            evidence_card=[2],
        )
        cpt_rain = TabularCPD(
            "Rain", 2, [[0.8, 0.2], [0.2, 0.8]], evidence=["Cloudy"], evidence_card=[2]
        )
        cpt_wet_grass = TabularCPD(
            "Wet_Grass",
            2,
            [[1, 0.1, 0.1, 0.01], [0, 0.9, 0.9, 0.99]],
            evidence=["Sprinkler", "Rain"],
            evidence_card=[2, 2],
        )

        self.bayesian_model_sprinkler = BayesianNetwork()
        self.bayesian_model_sprinkler.add_edges_from(
            [
                ("Cloudy", "Sprinkler"),
                ("Cloudy", "Rain"),
                ("Sprinkler", "Wet_Grass"),
                ("Rain", "Wet_Grass"),
            ]
        )
        self.bayesian_model_sprinkler.add_cpds(
            cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass
        )

        # A test Markov model
        self.markov_model = MarkovNetwork([("A", "B"), ("C", "B"), ("B", "D")])
        factor_ab = DiscreteFactor(["A", "B"], [2, 3], [1, 2, 3, 4, 5, 6])
        factor_cb = DiscreteFactor(
            ["C", "B"], [4, 3], [3, 1, 4, 5, 7, 8, 1, 3, 10, 4, 5, 6]
        )
        factor_bd = DiscreteFactor(["B", "D"], [3, 2], [5, 7, 2, 1, 9, 3])
        self.markov_model.add_factors(factor_ab, factor_cb, factor_bd)

        self.gibbs = GibbsSampling(self.bayesian_model)

    def tearDown(self):
        del self.bayesian_model
        del self.bayesian_model_sprinkler
        del self.markov_model

    @patch("pgmpy.sampling.GibbsSampling._get_kernel_from_markov_model", autospec=True)
    def test_init_markov_model(self, get_kernel):
        model = MagicMock(spec_set=MarkovNetwork)
        gibbs = GibbsSampling(model)
        get_kernel.assert_called_once_with(gibbs, model)

    def test_get_kernel_from_bayesian_model(self):
        gibbs = GibbsSampling()
        gibbs._get_kernel_from_bayesian_model(self.bayesian_model)
        self.assertListEqual(list(gibbs.variables), list(self.bayesian_model.nodes()))
        self.assertDictEqual(gibbs.cardinalities, {"diff": 2, "intel": 2, "grade": 3})

    def test_get_kernel_from_bayesian_model_sprinkler(self):
        gibbs = GibbsSampling()
        gibbs._get_kernel_from_bayesian_model(self.bayesian_model_sprinkler)
        self.assertListEqual(
            list(gibbs.variables), list(self.bayesian_model_sprinkler.nodes())
        )
        self.assertDictEqual(
            gibbs.cardinalities,
            {"Cloudy": 2, "Rain": 2, "Sprinkler": 2, "Wet_Grass": 2},
        )

    def test_get_kernel_from_markov_model(self):
        gibbs = GibbsSampling()
        gibbs._get_kernel_from_markov_model(self.markov_model)
        self.assertListEqual(list(gibbs.variables), list(self.markov_model.nodes()))
        self.assertDictEqual(gibbs.cardinalities, {"A": 2, "B": 3, "C": 4, "D": 2})

    def test_sample(self):
        start_state = [State("diff", 0), State("intel", 0), State("grade", 0)]
        sample = self.gibbs.sample(start_state, 2)
        self.assertEqual(len(sample), 2)
        self.assertEqual(len(sample.columns), 3)
        self.assertIn("diff", sample.columns)
        self.assertIn("intel", sample.columns)
        self.assertIn("grade", sample.columns)
        self.assertTrue(set(sample["diff"]).issubset({0, 1}))
        self.assertTrue(set(sample["intel"]).issubset({0, 1}))
        self.assertTrue(set(sample["grade"]).issubset({0, 1, 2}))

    def test_sample_sprinkler(self):
        gibbs = GibbsSampling(self.bayesian_model_sprinkler)
        nodes = set(self.bayesian_model_sprinkler.nodes)
        sample = gibbs.sample(size=2)
        self.assertEqual(len(sample), 2)
        self.assertEqual(len(sample.columns), 4)
        self.assertEqual(nodes, set(sample.columns))
        for node in nodes:
            self.assertTrue(set(sample[node]).issubset({0, 1}))

    def test_sample_limit(self):
        samples = self.gibbs.sample(size=int(1e4))
        marginal_prob = VariableElimination(self.bayesian_model).query(
            list(self.bayesian_model.nodes()), joint=False, show_progress=False
        )
        sample_prob = {
            node: samples.loc[:, node].value_counts() / 1e4
            for node in self.bayesian_model.nodes()
        }
        for node in self.bayesian_model.nodes():
            self.assertTrue(
                np.allclose(
                    sorted(marginal_prob[node].values),
                    sorted(sample_prob[node].values),
                    atol=0.05,
                )
            )

    @patch("pgmpy.sampling.GibbsSampling.random_state", autospec=True)
    def test_sample_less_arg(self, random_state):
        self.gibbs.state = None
        random_state.return_value = [
            State("diff", 0),
            State("intel", 0),
            State("grade", 0),
        ]
        sample = self.gibbs.sample(size=2)
        random_state.assert_called_once_with(self.gibbs)
        self.assertEqual(len(sample), 2)

    def test_generate_sample(self):
        start_state = [State("diff", 0), State("intel", 0), State("grade", 0)]
        gen = self.gibbs.generate_sample(start_state, 2)
        samples = [sample for sample in gen]
        self.assertEqual(len(samples), 2)
        self.assertEqual(
            {samples[0][0].var, samples[0][1].var, samples[0][2].var},
            {"diff", "intel", "grade"},
        )
        self.assertEqual(
            {samples[1][0].var, samples[1][1].var, samples[1][2].var},
            {"diff", "intel", "grade"},
        )

    @patch("pgmpy.sampling.GibbsSampling.random_state", autospec=True)
    def test_generate_sample_less_arg(self, random_state):
        self.gibbs.state = None
        gen = self.gibbs.generate_sample(size=2)
        samples = [sample for sample in gen]
        random_state.assert_called_once_with(self.gibbs)
        self.assertEqual(len(samples), 2)
