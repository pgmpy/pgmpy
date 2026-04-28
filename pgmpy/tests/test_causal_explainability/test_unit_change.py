import unittest

from pgmpy.causal_explainability.unit_change import UnitChangeAttribution
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class TestUnitChangeLinearGaussian(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("X1", "Y"), ("X2", "Y")])
        self.model.add_cpds(
            LinearGaussianCPD("X1", beta=[0.0], std=1.0),
            LinearGaussianCPD("X2", beta=[0.0], std=1.0),
            LinearGaussianCPD("Y", beta=[1.0, 2.0, 3.0], std=0.5, evidence=["X1", "X2"]),
        )

    def test_analytical_linear_gaussian(self):
        uca = UnitChangeAttribution()
        result = uca.attribute(
            self.model,
            data=None,
            target="Y",
            observation_old={"X1": 1.0, "X2": 2.0},
            observation_new={"X1": 3.0, "X2": 2.0},
        )
        self.assertAlmostEqual(result["X1"], 4.0, places=10)
        self.assertAlmostEqual(result["X2"], 0.0, places=10)

    def test_efficiency_linear_gaussian(self):
        uca = UnitChangeAttribution()
        obs_old = {"X1": 1.0, "X2": 2.0}
        obs_new = {"X1": 3.0, "X2": 5.0}
        result = uca.attribute(self.model, data=None, target="Y", observation_old=obs_old, observation_new=obs_new)
        cpd = self.model.get_cpds("Y")
        f_old = cpd.beta[0] + cpd.beta[1] * obs_old["X1"] + cpd.beta[2] * obs_old["X2"]
        f_new = cpd.beta[0] + cpd.beta[1] * obs_new["X1"] + cpd.beta[2] * obs_new["X2"]
        self.assertAlmostEqual(sum(result.values()), f_new - f_old, places=10)

    def test_no_change(self):
        uca = UnitChangeAttribution()
        obs = {"X1": 1.0, "X2": 2.0}
        result = uca.attribute(self.model, data=None, target="Y", observation_old=obs, observation_new=obs)
        for v in result.values():
            self.assertAlmostEqual(v, 0.0, places=10)


class TestUnitChangeDiscrete(unittest.TestCase):
    def setUp(self):
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.models import DiscreteBayesianNetwork

        self.model = DiscreteBayesianNetwork([("X", "Y")])
        cpd_x = TabularCPD("X", 2, [[0.4], [0.6]])
        cpd_y = TabularCPD("Y", 2, [[0.9, 0.3], [0.1, 0.7]], evidence=["X"], evidence_card=[2])
        self.model.add_cpds(cpd_x, cpd_y)

    def test_efficiency_discrete(self):
        uca = UnitChangeAttribution()
        result = uca.attribute(self.model, data=None, target="Y", observation_old={"X": 0}, observation_new={"X": 1})
        e_y_old = 0 * 0.9 + 1 * 0.1
        e_y_new = 0 * 0.3 + 1 * 0.7
        self.assertAlmostEqual(sum(result.values()), e_y_new - e_y_old, places=10)
