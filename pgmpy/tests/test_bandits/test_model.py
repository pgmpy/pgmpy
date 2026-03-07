import unittest

import pandas as pd

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


def _build_simple_model():
    """Build a small BN: T -> Y with binary variables for testing."""
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


def _build_multi_var_model():
    """Build BN: T1, T2 -> Y with binary variables."""
    model = DiscreteBayesianNetwork([("T1", "Y"), ("T2", "Y")])
    cpd_t1 = TabularCPD("T1", 2, [[0.5], [0.5]], state_names={"T1": ["0", "1"]})
    cpd_t2 = TabularCPD("T2", 2, [[0.5], [0.5]], state_names={"T2": ["0", "1"]})
    cpd_y = TabularCPD(
        "Y",
        2,
        [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
        evidence=["T1", "T2"],
        evidence_card=[2, 2],
        state_names={"Y": ["0", "1"], "T1": ["0", "1"], "T2": ["0", "1"]},
    )
    model.add_cpds(cpd_t1, cpd_t2, cpd_y)
    model.check_model()
    return model


class TestCausalBanditModelInit(unittest.TestCase):
    def setUp(self):
        self.model = _build_simple_model()

    def test_valid_init(self):
        from pgmpy.bandits import CausalBanditModel

        cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.assertEqual(cbm.reward_variable, "Y")
        self.assertEqual(cbm.intervenable_variables, ["T"])
        self.assertEqual(cbm.intervention_type, "hard")

    def test_invalid_reward_variable(self):
        from pgmpy.bandits import CausalBanditModel

        self.assertRaises(
            ValueError,
            CausalBanditModel,
            self.model,
            reward_variable="Z",
            intervenable_variables=["T"],
        )

    def test_invalid_intervenable_variable(self):
        from pgmpy.bandits import CausalBanditModel

        self.assertRaises(
            ValueError,
            CausalBanditModel,
            self.model,
            reward_variable="Y",
            intervenable_variables=["Z"],
        )

    def test_reward_in_intervenable_raises(self):
        from pgmpy.bandits import CausalBanditModel

        self.assertRaises(
            ValueError,
            CausalBanditModel,
            self.model,
            reward_variable="Y",
            intervenable_variables=["Y"],
        )

    def test_soft_intervention_type(self):
        from pgmpy.bandits import CausalBanditModel

        cbm = CausalBanditModel(
            self.model,
            reward_variable="Y",
            intervenable_variables=["T"],
            intervention_type="soft",
        )
        self.assertEqual(cbm.intervention_type, "soft")

    def test_invalid_intervention_type(self):
        from pgmpy.bandits import CausalBanditModel

        self.assertRaises(
            ValueError,
            CausalBanditModel,
            self.model,
            reward_variable="Y",
            intervenable_variables=["T"],
            intervention_type="invalid",
        )


class TestInterventionSpace(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )

    def test_default_intervention_space(self):
        actions = self.cbm.get_possible_interventions()
        # Should include observational arm {} plus do(T=0), do(T=1)
        self.assertEqual(len(actions), 3)
        self.assertIn({}, actions)
        self.assertIn({"T": "0"}, actions)
        self.assertIn({"T": "1"}, actions)

    def test_set_intervention_space_restricts(self):
        self.cbm.set_intervention_space("T", ["1"])
        actions = self.cbm.get_possible_interventions()
        # Observational + do(T=1) only
        self.assertEqual(len(actions), 2)
        self.assertIn({}, actions)
        self.assertIn({"T": "1"}, actions)

    def test_set_intervention_space_invalid_variable(self):
        self.assertRaises(ValueError, self.cbm.set_intervention_space, "Z", ["0"])

    def test_set_intervention_space_invalid_state(self):
        self.assertRaises(ValueError, self.cbm.set_intervention_space, "T", ["bad"])

    def test_multi_variable_intervention_space(self):
        from pgmpy.bandits import CausalBanditModel

        model = _build_multi_var_model()
        cbm = CausalBanditModel(
            model, reward_variable="Y", intervenable_variables=["T1", "T2"]
        )
        actions = cbm.get_possible_interventions()
        # {} + 2x2 = 5 arms
        self.assertEqual(len(actions), 5)

    def test_include_observational_false(self):
        actions = self.cbm.get_possible_interventions(include_observational=False)
        self.assertNotIn({}, actions)
        self.assertEqual(len(actions), 2)


class TestExpectedReward(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )

    def test_expected_reward_returns_factor(self):
        from pgmpy.factors.discrete import DiscreteFactor

        factor = self.cbm.expected_reward({"T": "1"})
        self.assertIsInstance(factor, DiscreteFactor)
        self.assertIn("Y", factor.variables)

    def test_expected_reward_observational(self):
        factor = self.cbm.expected_reward({})
        self.assertAlmostEqual(sum(factor.values), 1.0, places=5)

    def test_expected_reward_values(self):
        # do(T=1) => P(Y=0)=0.2, P(Y=1)=0.8
        factor = self.cbm.expected_reward({"T": "1"})
        self.assertAlmostEqual(factor.get_value(Y="1"), 0.8, places=5)
        self.assertAlmostEqual(factor.get_value(Y="0"), 0.2, places=5)


class TestObserve(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )

    def test_observe_returns_dataframe(self):
        result = self.cbm.observe({"T": "1"}, n_samples=10, seed=42)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 10)
        self.assertIn("Y", result.columns)

    def test_observe_observational_arm(self):
        result = self.cbm.observe({}, n_samples=10, seed=42)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 10)

    def test_observe_seed_reproducibility(self):
        r1 = self.cbm.observe({"T": "1"}, n_samples=50, seed=42)
        r2 = self.cbm.observe({"T": "1"}, n_samples=50, seed=42)
        pd.testing.assert_frame_equal(r1, r2)


class TestSoftIntervention(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditModel

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model,
            reward_variable="Y",
            intervenable_variables=["T"],
            intervention_type="soft",
        )

    def test_set_soft_intervention_space(self):
        soft_cpd = TabularCPD("T", 2, [[0.3], [0.7]], state_names={"T": ["0", "1"]})
        self.cbm.set_intervention_space("T", [soft_cpd])
        actions = self.cbm.get_possible_interventions()
        # {} + 1 soft action
        self.assertEqual(len(actions), 2)

    def test_observe_soft_intervention(self):
        soft_cpd = TabularCPD("T", 2, [[0.3], [0.7]], state_names={"T": ["0", "1"]})
        self.cbm.set_intervention_space("T", [soft_cpd])
        actions = self.cbm.get_possible_interventions(include_observational=False)
        result = self.cbm.observe(actions[0], n_samples=50, seed=42)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 50)

    def test_invalid_soft_intervention_value(self):
        self.assertRaises(
            ValueError,
            self.cbm.set_intervention_space,
            "T",
            ["not_a_cpd"],
        )


class TestRewardType(unittest.TestCase):
    def test_auto_detect_binary(self):
        from pgmpy.bandits import CausalBanditModel

        model = _build_simple_model()
        cbm = CausalBanditModel(
            model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.assertEqual(cbm.reward_type, "binary")

    def test_auto_detect_categorical(self):
        from pgmpy.bandits import CausalBanditModel

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
            model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.assertEqual(cbm.reward_type, "categorical")

    def test_user_override(self):
        from pgmpy.bandits import CausalBanditModel

        model = _build_simple_model()
        cbm = CausalBanditModel(
            model,
            reward_variable="Y",
            intervenable_variables=["T"],
            reward_type="continuous",
        )
        self.assertEqual(cbm.reward_type, "continuous")


if __name__ == "__main__":
    unittest.main()
