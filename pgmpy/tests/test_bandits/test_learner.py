import unittest

import numpy as np
import pandas as pd

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


class TestCausalBanditLearnerInit(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditLearner, CausalBanditModel, CausalUCB

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalUCB(self.cbm)
        self.learner = CausalBanditLearner(self.cbm, self.policy)

    def test_init(self):
        self.assertIsNotNone(self.learner)
        self.assertEqual(len(self.learner.get_history()), 0)


class TestCausalBanditLearnerRun(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import CausalBanditLearner, CausalBanditModel, CausalUCB

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalUCB(self.cbm)
        self.learner = CausalBanditLearner(self.cbm, self.policy)

    def test_run_returns_history_length(self):
        self.learner.run(n_rounds=20, seed=42)
        history = self.learner.get_history()
        self.assertEqual(len(history), 20)

    def test_history_columns(self):
        self.learner.run(n_rounds=10, seed=42)
        history = self.learner.get_history()
        self.assertIsInstance(history, pd.DataFrame)
        self.assertIn("round", history.columns)
        self.assertIn("reward", history.columns)
        self.assertIn("T", history.columns)

    def test_history_round_numbers(self):
        self.learner.run(n_rounds=5, seed=42)
        history = self.learner.get_history()
        np.testing.assert_array_equal(history["round"].values, [0, 1, 2, 3, 4])

    def test_seed_reproducibility(self):
        from pgmpy.bandits import CausalBanditLearner, CausalBanditModel, CausalUCB

        cbm1 = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        policy1 = CausalUCB(cbm1)
        learner1 = CausalBanditLearner(cbm1, policy1)
        learner1.run(n_rounds=20, seed=42)

        cbm2 = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        policy2 = CausalUCB(cbm2)
        learner2 = CausalBanditLearner(cbm2, policy2)
        learner2.run(n_rounds=20, seed=42)

        pd.testing.assert_frame_equal(learner1.get_history(), learner2.get_history())


class TestCausalBanditLearnerWithTS(unittest.TestCase):
    def setUp(self):
        from pgmpy.bandits import (
            CausalBanditLearner,
            CausalBanditModel,
            CausalThompsonSampling,
        )

        self.model = _build_simple_model()
        self.cbm = CausalBanditModel(
            self.model, reward_variable="Y", intervenable_variables=["T"]
        )
        self.policy = CausalThompsonSampling(self.cbm)
        self.learner = CausalBanditLearner(self.cbm, self.policy)

    def test_run_with_ts(self):
        self.learner.run(n_rounds=20, seed=42)
        history = self.learner.get_history()
        self.assertEqual(len(history), 20)


class TestCausalBanditLearnerRewardMapping(unittest.TestCase):
    """Test learner with categorical rewards and a reward_mapping."""

    def setUp(self):
        from pgmpy.bandits import CausalBanditLearner, CausalBanditModel, CausalUCB

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
        self.cbm = CausalBanditModel(
            model,
            reward_variable="Y",
            intervenable_variables=["T"],
            reward_mapping={"low": 0.0, "med": 0.5, "high": 1.0},
        )
        self.policy = CausalUCB(self.cbm)
        self.learner = CausalBanditLearner(self.cbm, self.policy)

    def test_run_with_reward_mapping(self):
        self.learner.run(n_rounds=20, seed=42)
        history = self.learner.get_history()
        self.assertEqual(len(history), 20)
        # Rewards should be numeric values from the mapping
        self.assertTrue(history["reward"].dtype in [np.float64, np.int64, float])


if __name__ == "__main__":
    unittest.main()
