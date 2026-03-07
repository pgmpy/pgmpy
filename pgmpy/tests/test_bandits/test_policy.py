import unittest

import numpy as np

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


def _build_simple_model():
    """Build a small BN: T -> Y with binary variables."""
    model = DiscreteBayesianNetwork([("T", "Y")])
    cpd_t = TabularCPD("T", 2, [[0.5], [0.5]], state_names={"T": ["0", "1"]})
    cpd_y = TabularCPD(
        "Y",
        2,
        [[0.8, 0.2], [0.2, 0.8]],
        evidence=["T"],
        evidence_card=[2],
        state_names={"Y": ["0", "1"], "T": ["0", "1"]},
    )
    model.add_cpds(cpd_t, cpd_y)
    model.check_model()
    return model


class TestCausalUCBInit(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalUCB

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalUCB(self.cbm)

    def test_init(self):
        self.assertIsNotNone(self.policy)
        self.assertEqual(self.policy.exploration_weight, 1.0)

    def test_custom_exploration_weight(self):
        from pgmpy.bandits import CausalUCB

        policy = CausalUCB(self.cbm, exploration_weight=2.0)
        self.assertEqual(policy.exploration_weight, 2.0)


class TestCausalUCBSelectAction(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalUCB

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalUCB(self.cbm)

    def test_select_action_returns_valid_action(self):
        action = self.policy.select_action()
        possible = self.cbm.get_possible_interventions()
        self.assertIn(action, possible)

    def test_select_action_explores_all_arms_first(self):
        """UCB should try each arm at least once before repeating."""
        selected = []
        n_arms = len(self.cbm.get_possible_interventions())
        for _ in range(n_arms):
            action = self.policy.select_action()
            selected.append(str(action))
            self.policy.update(action, 0.5)
        self.assertEqual(len(set(selected)), n_arms)


class TestCausalUCBUpdate(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalUCB

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalUCB(self.cbm)

    def test_update_increments_counts(self):
        action = {"T": "1"}
        self.policy.update(action, 1.0)
        key = self.policy._action_key(action)
        self.assertEqual(self.policy.counts[key], 1)

    def test_update_tracks_mean(self):
        action = {"T": "1"}
        self.policy.update(action, 1.0)
        self.policy.update(action, 0.0)
        key = self.policy._action_key(action)
        self.assertAlmostEqual(self.policy.q_values[key], 0.5)


class TestCausalUCBRecommend(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalUCB

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalUCB(self.cbm)

    def test_recommend_returns_best_arm(self):
        # Force arm {"T": "1"} to have higher mean
        self.policy.update({"T": "1"}, 1.0)
        self.policy.update({"T": "1"}, 1.0)
        self.policy.update({"T": "0"}, 0.0)
        self.policy.update({"T": "0"}, 0.0)
        self.policy.update({}, 0.0)
        recommended = self.policy.recommend()
        self.assertEqual(recommended, {"T": "1"})


class TestCausalThompsonSamplingInit(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalThompsonSampling

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalThompsonSampling(self.cbm)

    def test_init(self):
        self.assertIsNotNone(self.policy)

    def test_init_binary_uses_beta(self):
        # For binary rewards, should use Beta posterior
        n_arms = len(self.cbm.get_possible_interventions())
        self.assertEqual(len(self.policy.alpha), n_arms)
        self.assertEqual(len(self.policy.beta), n_arms)


class TestCausalThompsonSamplingSelectAction(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalThompsonSampling

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalThompsonSampling(self.cbm)

    def test_select_action_returns_valid_action(self):
        np.random.seed(42)
        action = self.policy.select_action()
        possible = self.cbm.get_possible_interventions()
        self.assertIn(action, possible)


class TestCausalThompsonSamplingUpdate(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalThompsonSampling

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalThompsonSampling(self.cbm)

    def test_update_binary_posterior(self):
        action = {"T": "1"}
        key = self.policy._action_key(action)
        alpha_before = self.policy.alpha[key]
        self.policy.update(action, 1.0)
        self.assertEqual(self.policy.alpha[key], alpha_before + 1)

    def test_update_binary_failure(self):
        action = {"T": "1"}
        key = self.policy._action_key(action)
        beta_before = self.policy.beta[key]
        self.policy.update(action, 0.0)
        self.assertEqual(self.policy.beta[key], beta_before + 1)


class TestCausalThompsonSamplingRecommend(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalThompsonSampling

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalThompsonSampling(self.cbm)

    def test_recommend_returns_best_arm(self):
        # Pump up alpha for {"T": "1"} heavily
        for _ in range(100):
            self.policy.update({"T": "1"}, 1.0)
            self.policy.update({"T": "0"}, 0.0)
            self.policy.update({}, 0.0)
        recommended = self.policy.recommend()
        self.assertEqual(recommended, {"T": "1"})


class TestCausalTSCategorical(unittest.TestCase):
    """Test Thompson Sampling with non-binary (categorical) reward variable."""

    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalThompsonSampling

        model = DiscreteBayesianNetwork([("T", "Y")])
        cpd_t = TabularCPD("T", 2, [[0.5], [0.5]], state_names={"T": ["0", "1"]})
        cpd_y = TabularCPD(
            "Y",
            3,
            [[0.5, 0.1], [0.3, 0.3], [0.2, 0.6]],
            evidence=["T"],
            evidence_card=[2],
            state_names={"Y": ["low", "med", "high"], "T": ["0", "1"]},
        )
        model.add_cpds(cpd_t, cpd_y)
        model.check_model()
        cbm = CausalBanditModel(
            model,
            reward_variable="Y",
            intervenable_variables=["T"],
            reward_mapping={"low": 0.0, "med": 0.5, "high": 1.0},
        )
        self.cbm = cbm
        self.policy = CausalThompsonSampling(cbm)

    def test_categorical_uses_gaussian(self):
        # For categorical/non-binary mapped rewards, should use Gaussian posterior
        n_arms = len(self.cbm.get_possible_interventions())
        self.assertEqual(len(self.policy.mu), n_arms)
        self.assertEqual(len(self.policy.tau), n_arms)

    def test_update_gaussian(self):
        action = {"T": "1"}
        key = self.policy._action_key(action)
        self.policy.update(action, 0.8)
        self.assertEqual(self.policy.counts[key], 1)


class TestPolicyFit(unittest.TestCase):
    """Test that fit() warm-starts from observational data."""

    def setUp(self):
        from pgmpy.bandits import CausalBanditModel, CausalUCB

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalUCB(self.cbm)

    def test_fit_with_observational_data(self):
        np.random.seed(42)
        data = self.model.simulate(n_samples=50, seed=42, show_progress=False)
        self.policy.fit(observational_data=data)
        # After fitting, the observational arm should have some count
        obs_key = self.policy._action_key({})
        self.assertGreater(self.policy.counts[obs_key], 0)


if __name__ == "__main__":
    unittest.main()
