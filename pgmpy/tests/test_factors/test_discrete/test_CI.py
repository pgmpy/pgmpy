import pytest
import numpy as np
import numpy.testing as npt

# Assuming the classes are in pgmpy.factors.discrete.CI
# Adjust this import if your file structure is different.
from pgmpy.factors.discrete import BinaryInfluenceModel, MultilevelInfluenceModel
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class TestBinaryInfluenceModel:
    def test_init_noisy_or(self):
        cpd = BinaryInfluenceModel(
            variable="Y",
            evidence=["X1", "X2"],
            activation_magnitude=[0.6, 0.4],
            mode="OR",
        )
        assert cpd.variable == "Y"
        assert cpd.evidence == ["X1", "X2"]
        assert cpd.variable_card == 2
        npt.assert_array_equal(cpd.cardinality, [2, 2, 2])

        # P(Y=1 | X1=0, X2=0) = 1 - (1-0)*(1-0) = 0.0
        # P(Y=1 | X1=0, X2=1) = 1 - (1-0)*(1-0.4) = 0.4
        # P(Y=1 | X1=1, X2=0) = 1 - (1-0.6)*(1-0) = 0.6
        # P(Y=1 | X1=1, X2=1) = 1 - (1-0.6)*(1-0.4) = 1 - (0.4 * 0.6) = 0.76
        expected_values = np.array(
            [
                [1.0, 0.6, 0.4, 0.24],  # P(Y=0)
                [0.0, 0.4, 0.6, 0.76],  # P(Y=1)
            ]
        )
        npt.assert_allclose(cpd.get_values(), expected_values)

    def test_init_noisy_or_leak(self):
        cpd = BinaryInfluenceModel(
            variable="Y",
            evidence=["X1", "X2"],
            activation_magnitude=[0.6, 0.4],
            mode="OR",
            leak=0.1,
        )
        # P_no_leak(Y=1 | X1=0, X2=0) = 0.0 -> P(Y=1) = 1 - (1-0.0)*(1-0.1) = 0.1
        # P_no_leak(Y=1 | X1=0, X2=1) = 0.4 -> P(Y=1) = 1 - (1-0.4)*(1-0.1) = 1 - 0.6*0.9 = 0.46
        # P_no_leak(Y=1 | X1=1, X2=0) = 0.6 -> P(Y=1) = 1 - (1-0.6)*(1-0.1) = 1 - 0.4*0.9 = 0.64
        # P_no_leak(Y=1 | X1=1, X2=1) = 0.76 -> P(Y=1) = 1 - (1-0.76)*(1-0.1) = 1 - 0.24*0.9 = 0.784
        expected_values = np.array(
            [
                [0.9, 0.54, 0.36, 0.216],  # P(Y=0)
                [0.1, 0.46, 0.64, 0.784],  # P(Y=1)
            ]
        )
        npt.assert_allclose(cpd.get_values(), expected_values)

    def test_init_noisy_and(self):
        cpd = BinaryInfluenceModel(
            variable="Y",
            evidence=["X1", "X2"],
            activation_magnitude=[0.6, 0.4],
            mode="AND",
        )
        # P(Y=1 | X1=0, X2=0) = 0.0
        # P(Y=1 | X1=0, X2=1) = 0.0
        # P(Y=1 | X1=1, X2=0) = 0.0
        # P(Y=1 | X1=1, X2=1) = 0.6 * 0.4 = 0.24
        expected_values = np.array(
            [
                [1.0, 1.0, 1.0, 0.76],  # P(Y=0)
                [0.0, 0.0, 0.0, 0.24],  # P(Y=1)
            ]
        )
        npt.assert_allclose(cpd.get_values(), expected_values)

    def test_init_noisy_and_leak(self):
        cpd = BinaryInfluenceModel(
            variable="Y",
            evidence=["X1", "X2"],
            activation_magnitude=[0.6, 0.4],
            mode="AND",
            leak=0.1,
        )
        # P_no_leak(Y=1 | 0, 0) = 0.0 -> P(Y=1) = 1 - (1-0.0)*(1-0.1) = 0.1
        # P_no_leak(Y=1 | 0, 1) = 0.0 -> P(Y=1) = 1 - (1-0.0)*(1-0.1) = 0.1
        # P_no_leak(Y=1 | 1, 0) = 0.0 -> P(Y=1) = 1 - (1-0.0)*(1-0.1) = 0.1
        # P_no_leak(Y=1 | 1, 1) = 0.24 -> P(Y=1) = 1 - (1-0.24)*(1-0.1) = 1 - 0.76*0.9 = 0.316
        expected_values = np.array(
            [
                [0.9, 0.9, 0.9, 0.684],  # P(Y=0)
                [0.1, 0.1, 0.1, 0.316],  # P(Y=1)
            ]
        )
        npt.assert_allclose(cpd.get_values(), expected_values)

    def test_boolean_style(self):
        cpd = BinaryInfluenceModel(
            variable="Y",
            evidence=["X1", "X2"],
            activation_magnitude=[0.6, 0.4],
            mode="OR",
            isboolean_style=True,
        )
        assert cpd.state_names["Y"] == [False, True]
        assert cpd.state_names["X1"] == [False, True]
        assert cpd.state_names["X2"] == [False, True]

        # Values should be identical to the numeric test_init_noisy_or
        expected_values = np.array(
            [
                [1.0, 0.6, 0.4, 0.24],  # P(Y=False)
                [0.0, 0.4, 0.6, 0.76],  # P(Y=True)
            ]
        )
        npt.assert_allclose(cpd.get_values(), expected_values)

    def test_init_errors(self):
        with pytest.raises(ValueError, match="mode must be either 'OR' or 'AND'"):
            BinaryInfluenceModel("Y", ["X1"], [0.5], mode="XOR")

        with pytest.raises(
            ValueError, match="Number of activation magnitudes must match"
        ):
            BinaryInfluenceModel("Y", ["X1", "X2"], [0.5], mode="OR")

        with pytest.raises(ValueError, match="All activation probabilities"):
            BinaryInfluenceModel("Y", ["X1"], [1.1], mode="OR")
        with pytest.raises(ValueError, match="All activation probabilities"):
            BinaryInfluenceModel("Y", ["X1"], [-0.1], mode="OR")

        with pytest.raises(ValueError, match="Leak value must be in"):
            BinaryInfluenceModel("Y", ["X1"], [0.5], mode="OR", leak=1.5)
        with pytest.raises(ValueError, match="Leak value must be in"):
            BinaryInfluenceModel("Y", ["X1"], [0.5], mode="OR", leak=-0.5)

    def test_inference_integration(self):
        model = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
        cpd_b = TabularCPD("B", 2, [[0.7], [0.3]])
        
        # P(C=1 | A=0, B=0) = 0.1 (leak)
        # P(C=1 | A=0, B=1) = 1 - (1-0.4)*(1-0.1) = 0.46
        # P(C=1 | A=1, B=0) = 1 - (1-0.6)*(1-0.1) = 0.64
        # P(C=1 | A=1, B=1) = 1 - (1 - (1-0.6)*(1-0.4)) * (1-0.1) = 1 - (1-0.76)*0.9 = 0.784
        cpd_c = BinaryInfluenceModel(
            variable="C",
            evidence=["A", "B"],
            activation_magnitude=[0.6, 0.4],
            mode="OR",
            leak=0.1
        )
        model.add_cpds(cpd_a, cpd_b, cpd_c)

        infer = VariableElimination(model)
        
        # P(C=1) = P(C=1|0,0)P(A=0)P(B=0) + P(C=1|0,1)P(A=0)P(B=1) + ...
        # P(C=1) = (0.1 * 0.2 * 0.7) + (0.46 * 0.2 * 0.3) + (0.64 * 0.8 * 0.7) + (0.784 * 0.8 * 0.3)
        # P(C=1) = 0.014 + 0.0276 + 0.3584 + 0.18816 = 0.58816
        # P(C=0) = 1 - 0.58816 = 0.41184
        query = infer.query(["C"])
        npt.assert_allclose(query.values, [0.41184, 0.58816])


class TestMultilevelInfluenceModel:
    def setup_method(self):
        self.influence_tables = {
            "X1": {0: [1.0, 0.0, 0.0], 1: [0.2, 0.5, 0.3]},  # 0=inactive, 1=active
            "X2": {0: [1.0, 0.0, 0.0], 1: [0.4, 0.4, 0.2]},  # 0=inactive, 1=active
        }
        self.levels = 3
        self.evidence = ["X1", "X2"]
        self.variable = "Y"

    def test_init_noisy_max(self):
        cpd = MultilevelInfluenceModel(
            variable=self.variable,
            evidence=self.evidence,
            influence_tables=self.influence_tables,
            levels=self.levels,
            mode="MAX",
        )
        # Cumulative tables:
        # X1: { 0: [1.0, 1.0, 1.0], 1: [0.2, 0.7, 1.0] }
        # X2: { 0: [1.0, 1.0, 1.0], 1: [0.4, 0.8, 1.0] }

        # Case (X1=0, X2=0): cum_prob = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        # -> probs = [1.0, 0.0, 0.0]
        # Case (X1=0, X2=1): cum_prob = [1.0, 1.0, 1.0] * [0.4, 0.8, 1.0] = [0.4, 0.8, 1.0]
        # -> probs = [0.4, 0.4, 0.2]
        # Case (X1=1, X2=0): cum_prob = [0.2, 0.7, 1.0] * [1.0, 1.0, 1.0] = [0.2, 0.7, 1.0]
        # -> probs = [0.2, 0.5, 0.3]
        # Case (X1=1, X2=1): cum_prob = [0.2, 0.7, 1.0] * [0.4, 0.8, 1.0] = [0.08, 0.56, 1.0]
        # -> probs = [0.08, 0.48, 0.44]
        expected_values = np.array(
            [
                [1.0, 0.4, 0.2, 0.08],  # P(Y=0)
                [0.0, 0.4, 0.5, 0.48],  # P(Y=1)
                [0.0, 0.2, 0.3, 0.44],  # P(Y=2)
            ]
        )
        npt.assert_allclose(cpd.get_values(), expected_values)

    def test_init_noisy_max_leak(self):
        leak_dist = [0.8, 0.1, 0.1]
        leak_cum = np.array([0.8, 0.9, 1.0])
        cpd = MultilevelInfluenceModel(
            variable=self.variable,
            evidence=self.evidence,
            influence_tables=self.influence_tables,
            levels=self.levels,
            mode="MAX",
            leak=leak_dist,
        )

        # Case (X1=1, X2=1):
        # cum_prob_no_leak = [0.08, 0.56, 1.0]
        # cum_prob_with_leak = [0.08, 0.56, 1.0] * [0.8, 0.9, 1.0] = [0.064, 0.504, 1.0]
        # -> probs = [0.064, 0.44, 0.496]
        expected_probs_11 = [0.064, 0.44, 0.496]
        
        # Test one column
        npt.assert_allclose(cpd.get_values()[:, 3], expected_probs_11)
        
        # Test (X1=0, X2=0):
        # cum_prob_no_leak = [1.0, 1.0, 1.0]
        # cum_prob_with_leak = [1.0, 1.0, 1.0] * [0.8, 0.9, 1.0] = [0.8, 0.9, 1.0]
        # -> probs = [0.8, 0.1, 0.1] (just the leak)
        npt.assert_allclose(cpd.get_values()[:, 0], leak_dist)

    def test_init_noisy_min(self):
        cpd = MultilevelInfluenceModel(
            variable=self.variable,
            evidence=self.evidence,
            influence_tables=self.influence_tables,
            levels=self.levels,
            mode="MIN",
        )
        # Cumulative tables (1-F(x))
        # X1: { 0: [0.0, 0.0, 0.0], 1: [0.8, 0.3, 0.0] }
        # X2: { 0: [0.0, 0.0, 0.0], 1: [0.6, 0.2, 0.0] }
        
        # Note: The implementation calculates 1 - PROD(1 - F_i(x))
        # F_i(x) are the cumulative tables from setup_method
        # F_X1_1 = [0.2, 0.7, 1.0] -> (1 - F) = [0.8, 0.3, 0.0]
        # F_X2_1 = [0.4, 0.8, 1.0] -> (1 - F) = [0.6, 0.2, 0.0]
        
        # Case (X1=1, X2=1):
        # complement_prod = [0.8, 0.3, 0.0] * [0.6, 0.2, 0.0] = [0.48, 0.06, 0.0]
        # cum_prob = 1 - complement_prod = [0.52, 0.94, 1.0]
        # -> probs = [0.52, 0.42, 0.06]
        expected_probs_11 = [0.52, 0.42, 0.06]
        npt.assert_allclose(cpd.get_values()[:, 3], expected_probs_11)

        # Case (X1=0, X2=0):
        # F_X1_0 = [1.0, 1.0, 1.0] -> (1 - F) = [0.0, 0.0, 0.0]
        # F_X2_0 = [1.0, 1.0, 1.0] -> (1 - F) = [0.0, 0.0, 0.0]
        # complement_prod = [0.0, 0.0, 0.0] * [0.0, 0.0, 0.0] = [0.0, 0.0, 0.0]
        # cum_prob = 1 - complement_prod = [1.0, 1.0, 1.0]
        # -> probs = [1.0, 0.0, 0.0]
        npt.assert_allclose(cpd.get_values()[:, 0], [1.0, 0.0, 0.0])

    def test_init_errors(self):
        with pytest.raises(ValueError, match="mode must be 'MAX' or 'MIN'"):
            MultilevelInfluenceModel(
                "Y", ["X1"], {}, 3, mode="AVG"
            )

        with pytest.raises(ValueError, match="must sum to 1"):
            bad_table = {"X1": {0: [1.0, 0.0, 0.0], 1: [0.5, 0.5, 0.5]}}
            MultilevelInfluenceModel(
                "Y", ["X1"], bad_table, 3
            )
            
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            bad_table = {"X1": {0: [1.0, 0.0, 0.0], 1: [1.5, -0.5, 0.0]}}
            MultilevelInfluenceModel(
                "Y", ["X1"], bad_table, 3
            )

        with pytest.raises(ValueError, match="must sum to 1"):
            MultilevelInfluenceModel(
                "Y", ["X1"], self.influence_tables, 3, leak=[0.5, 0.6]
            )