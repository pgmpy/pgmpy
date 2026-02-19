import pytest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import Inference, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


@pytest.fixture
def setup_statename_init():
    """Fixture to set up data for TestStateNameInit tests."""
    sn2 = {
        "grade": ["A", "B", "F"],
        "diff": ["high", "low"],
        "intel": ["poor", "good", "very good"],
    }
    sn1 = {
        "speed": ["low", "medium", "high"],
        "switch": ["on", "off"],
        "time": ["day", "night"],
    }

    sn2_no_names = {"grade": [0, 1, 2], "diff": [0, 1], "intel": [0, 1, 2]}
    sn1_no_names = {"speed": [0, 1, 2], "switch": [0, 1], "time": [0, 1]}

    phi1 = DiscreteFactor(["speed", "switch", "time"], [3, 2, 2], np.ones(12))
    phi2 = DiscreteFactor(
        ["speed", "switch", "time"], [3, 2, 2], np.ones(12), state_names=sn1
    )

    cpd1 = TabularCPD(
        "grade",
        3,
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ],
        evidence=["diff", "intel"],
        evidence_card=[2, 3],
    )
    cpd2 = TabularCPD(
        "grade",
        3,
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ],
        evidence=["diff", "intel"],
        evidence_card=[2, 3],
        state_names=sn2,
    )

    student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
    diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
    intel_cpd = TabularCPD("intel", 2, [[0.3], [0.7]])
    grade_cpd = TabularCPD(
        "grade",
        3,
        [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8]],
        evidence=["diff", "intel"],
        evidence_card=[2, 2],
    )
    student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
    model1 = Inference(student)
    model2 = Inference(student)

    return {
        "sn2": sn2,
        "sn1": sn1,
        "sn2_no_names": sn2_no_names,
        "sn1_no_names": sn1_no_names,
        "phi1": phi1,
        "phi2": phi2,
        "cpd1": cpd1,
        "cpd2": cpd2,
        "model1": model1,
        "model2": model2,
    }


class TestStateNameInit:
    def test_factor_init_statename(self, setup_statename_init):
        data = setup_statename_init
        assert data["phi1"].state_names == data["sn1_no_names"]
        assert data["phi2"].state_names == data["sn1"]

    def test_cpd_init_statename(self, setup_statename_init):
        data = setup_statename_init
        assert data["cpd1"].state_names == data["sn2_no_names"]
        assert data["cpd2"].state_names == data["sn2"]


@pytest.fixture
def setup_statename_decorator():
    """Fixture to set up data for StateNameDecorator tests."""
    sn2 = {
        "grade": ["A", "B", "F"],
        "diff": ["high", "low"],
        "intel": ["poor", "good", "very good"],
    }
    sn1 = {
        "speed": ["low", "medium", "high"],
        "switch": ["on", "off"],
        "time": ["day", "night"],
    }

    phi1 = DiscreteFactor(["speed", "switch", "time"], [3, 2, 2], np.ones(12))
    phi2 = DiscreteFactor(
        ["speed", "switch", "time"], [3, 2, 2], np.ones(12), state_names=sn1
    )

    cpd1 = TabularCPD(
        "grade",
        3,
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ],
        evidence=["diff", "intel"],
        evidence_card=[2, 3],
    )
    cpd2 = TabularCPD(
        "grade",
        3,
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ],
        evidence=["diff", "intel"],
        evidence_card=[2, 3],
        state_names=sn2,
    )

    student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
    student_state_names = DiscreteBayesianNetwork(
        [("diff", "grade"), ("intel", "grade")]
    )

    diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
    intel_cpd = TabularCPD("intel", 2, [[0.3], [0.7]])
    grade_cpd = TabularCPD(
        "grade",
        3,
        [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8]],
        evidence=["diff", "intel"],
        evidence_card=[2, 2],
    )

    diff_cpd_state_names = TabularCPD(
        variable="diff",
        variable_card=2,
        values=[[0.2], [0.8]],
        state_names={"diff": ["high", "low"]},
    )
    intel_cpd_state_names = TabularCPD(
        variable="intel",
        variable_card=2,
        values=[[0.3], [0.7]],
        state_names={"intel": ["poor", "good", "very good"]},
    )
    grade_cpd_state_names = TabularCPD(
        "grade",
        3,
        [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8]],
        evidence=["diff", "intel"],
        evidence_card=[2, 2],
        state_names=sn2,
    )

    student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
    student_state_names.add_cpds(
        diff_cpd_state_names, intel_cpd_state_names, grade_cpd_state_names
    )

    model_no_state_names = VariableElimination(student)
    model_with_state_names = VariableElimination(student_state_names)

    return {
        "sn2": sn2,
        "sn1": sn1,
        "phi1": phi1,
        "phi2": phi2,
        "cpd1": cpd1,
        "cpd2": cpd2,
        "model_no_state_names": model_no_state_names,
        "model_with_state_names": model_with_state_names,
    }


class TestStateNameDecorator:
    def test_assignment_statename(self, setup_statename_decorator):
        data = setup_statename_decorator
        req_op1 = [
            [("speed", "low"), ("switch", "on"), ("time", "night")],
            [("speed", "low"), ("switch", "off"), ("time", "day")],
        ]
        req_op2 = [
            [("speed", 0), ("switch", 0), ("time", 1)],
            [("speed", 0), ("switch", 1), ("time", 0)],
        ]

        assert data["phi1"].assignment([1, 2]) == req_op2
        assert data["phi2"].assignment([1, 2]) == req_op1

    def test_factor_reduce_statename(self, setup_statename_decorator):
        data = setup_statename_decorator
        sn1 = data["sn1"]

        phi = DiscreteFactor(
            ["speed", "switch", "time"], [3, 2, 2], np.ones(12), state_names=sn1
        )
        phi.reduce([("speed", "medium"), ("time", "day")])
        assert phi.variables == ["switch"]
        assert phi.cardinality == [2]
        np_test.assert_array_equal(phi.values, np.array([1, 1]))

        phi = DiscreteFactor(
            ["speed", "switch", "time"], [3, 2, 2], np.ones(12), state_names=sn1
        )
        phi = phi.reduce([("speed", "medium"), ("time", "day")], inplace=False)
        assert phi.variables == ["switch"]
        assert phi.cardinality == [2]
        np_test.assert_array_equal(phi.values, np.array([1, 1]))

        phi = DiscreteFactor(["speed", "switch", "time"], [3, 2, 2], np.ones(12))
        phi.reduce([("speed", 1), ("time", 0)])
        assert phi.variables == ["switch"]
        assert phi.cardinality == [2]
        np_test.assert_array_equal(phi.values, np.array([1, 1]))

        phi = DiscreteFactor(["speed", "switch", "time"], [3, 2, 2], np.ones(12))
        phi = phi.reduce([("speed", 1), ("time", 0)], inplace=False)
        assert phi.variables == ["switch"]
        assert phi.cardinality == [2]
        np_test.assert_array_equal(phi.values, np.array([1, 1]))

    def test_reduce_cpd_statename(self, setup_statename_decorator):
        data = setup_statename_decorator
        sn2 = data["sn2"]

        cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
            state_names=sn2,
        )
        cpd.reduce([("diff", "high")])
        assert cpd.variable == "grade"
        assert cpd.variables == ["grade", "intel"]
        np_test.assert_array_equal(
            cpd.get_values(),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]),
        )

        cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        cpd.reduce([("diff", 0)])
        assert cpd.variable == "grade"
        assert cpd.variables == ["grade", "intel"]
        np_test.assert_array_equal(
            cpd.get_values(),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]),
        )

        cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
            state_names=sn2,
        )
        cpd = cpd.reduce([("diff", "high")], inplace=False)
        assert cpd.variable == "grade"
        assert cpd.variables == ["grade", "intel"]
        np_test.assert_array_equal(
            cpd.get_values(),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]),
        )

        cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        cpd = cpd.reduce([("diff", 0)], inplace=False)
        assert cpd.variable == "grade"
        assert cpd.variables == ["grade", "intel"]
        np_test.assert_array_equal(
            cpd.get_values(),
            np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]),
        )

    def test_inference_query_statename(self, setup_statename_decorator):
        data = setup_statename_decorator
        model_with_state_names = data["model_with_state_names"]
        model_no_state_names = data["model_no_state_names"]

        inf_op1 = model_with_state_names.query(
            ["grade"], evidence={"intel": "poor"}
        )
        inf_op2 = model_no_state_names.query(["grade"], evidence={"intel": 0})
        req_op = DiscreteFactor(
            ["grade"],
            [3],
            np.array([0.1, 0.1, 0.8]),
            state_names={"grade": ["A", "B", "F"]},
        )
        assert inf_op1 == req_op
        assert inf_op1 == req_op

        inf_op1 = model_with_state_names.map_query(
            ["grade"], evidence={"intel": "poor"}
        )
        inf_op2 = model_no_state_names.map_query(["grade"], evidence={"intel": 0})
        req_op1 = {"grade": "F"}
        req_op2 = {"grade": 2}

        assert inf_op1 == req_op1
        assert inf_op2 == req_op2

    def test_add_state_names(self):
        # Test string state names taking precedence over numeric ones
        numeric_states = DiscreteFactor(
            ["speed"], [3], np.ones(3), state_names={"speed": [0, 1, 2]}
        )
        string_states = DiscreteFactor(
            ["speed"], [3], np.ones(3), state_names={"speed": ["low", "medium", "high"]}
        )

        # Make a copy to test in both directions
        numeric_states_copy = DiscreteFactor(
            ["speed"], [3], np.ones(3), state_names={"speed": [0, 1, 2]}
        )

        # String states should take precedence
        numeric_states.add_state_names(string_states)
        assert numeric_states.state_names["speed"] == ["low", "medium", "high"]

        # Test the opposite direction - string states should still take precedence
        string_states.add_state_names(numeric_states_copy)
        assert string_states.state_names["speed"] == ["low", "medium", "high"]

        # Test conflicting string state names
        states1 = DiscreteFactor(
            ["switch"], [2], np.ones(2), state_names={"switch": ["on", "off"]}
        )
        states2 = DiscreteFactor(
            ["switch"], [2], np.ones(2), state_names={"switch": ["high", "low"]}
        )

        # Should raise a ValueError due to conflict
        with pytest.raises(ValueError):
            states1.add_state_names(states2)

        # Test merging non-conflicting state names for different variables
        factor1 = DiscreteFactor(
            ["speed"], [3], np.ones(3), state_names={"speed": ["low", "medium", "high"]}
        )
        factor2 = DiscreteFactor(
            ["switch"], [2], np.ones(2), state_names={"switch": ["on", "off"]}
        )

        # Should merge without conflict
        factor1.add_state_names(factor2)
        assert factor1.state_names["speed"] == ["low", "medium", "high"]
        assert factor1.state_names["switch"] == ["on", "off"]
