import os
import tempfile
import unittest
import warnings

import numpy as np
import numpy.testing as np_test
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import XDSLReader, XDSLWriter
from pgmpy.utils import get_example_model

TEST_FILE = """<?xml version="1.0" encoding="UTF-8"?>
<!-- This network was created in trial version of GeNIe, which can be used for evaluation purposes only -->
<smile version="1.0" id="Asia" numsamples="10000" discsamples="10000">
    <nodes>
        <cpt id="asia" diagtype="observation" ranked="true">
            <state id="no" />
            <state id="yes" />
            <probabilities>0.99 0.01</probabilities>
        </cpt>
        <cpt id="tub" diagtype="target">
            <state id="no" label="F5" />
            <state id="yes" label="F6" fault="true" />
            <parents>asia</parents>
            <probabilities>0.99 0.01 0.95 0.05</probabilities>
        </cpt>
        <cpt id="smoke" diagtype="observation" ranked="true">
            <state id="no" />
            <state id="yes" />
            <probabilities>0.5 0.5</probabilities>
        </cpt>
        <cpt id="lung" diagtype="target">
            <state id="no" label="F9" />
            <state id="yes" label="F10" fault="true" />
            <parents>smoke</parents>
            <probabilities>0.99 0.01 0.9 0.1</probabilities>
        </cpt>
        <cpt id="either">
            <state id="Nothing" />
            <state id="CancerORTuberculosis" />
            <parents>tub lung</parents>
            <probabilities>1.00 0.0 1.00 0.0 1.00 0.0 0.0 1.0</probabilities>
        </cpt>
        <cpt id="xray" diagtype="observation" ranked="true">
            <state id="Normal" />
            <state id="Abnormal" />
            <parents>either</parents>
            <probabilities>0.95 0.05 0.02 0.98</probabilities>
        </cpt>
        <cpt id="bronc" diagtype="target">
            <state id="Absent" label="F15" />
            <state id="Present" label="F16" fault="true" />
            <parents>smoke</parents>
            <probabilities>0.7 0.3 0.4 0.6</probabilities>
        </cpt>
        <cpt id="dysp" diagtype="observation" ranked="true">
            <state id="Absent" />
            <state id="Present" />
            <parents>either bronc</parents>
            <probabilities>0.9 0.1 0.2 0.8 0.3 0.7 0.1 0.9</probabilities>
        </cpt>
    </nodes>
</smile>"""

TEST_WHITESPACE_MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<!-- This network was created in trial version of GeNIe, which can be used for evaluation purposes only -->
<smile version="1.0" id="Asia" numsamples="10000" discsamples="10000">
    <nodes>
        <cpt id="node 1" diagtype="observation" ranked="true">
            <state id="no" />
            <state id="yes" />
            <probabilities>0.5 0.5</probabilities>
        </cpt>
        <cpt id="node 2" diagtype="target">
            <state id="no" label="F5" />
            <state id="yes" label="F6" fault="true" />
            <parents>node 1</parents>
            <probabilities>0.5 0.5 0.5 0.5</probabilities>
        </cpt>
    </nodes>
</smile>"""


class TestXDSLReaderMethodsString(unittest.TestCase):
    def setUp(self):
        self.reader = XDSLReader(string=TEST_FILE)

    def test_whitespace_error(self):
        with self.assertRaises(ValueError):
            self.model_with_whitespace = XDSLReader(string=TEST_WHITESPACE_MODEL)

    def test_get_variables(self):
        var_expected = [
            "asia",
            "tub",
            "smoke",
            "lung",
            "either",
            "xray",
            "bronc",
            "dysp",
        ]
        self.assertListEqual(self.reader.variables, var_expected)

    def test_get_parents(self):
        parents_expected = {
            "asia": [],
            "tub": ["asia"],
            "smoke": [],
            "lung": ["smoke"],
            "either": ["tub", "lung"],
            "xray": ["either"],
            "bronc": ["smoke"],
            "dysp": ["either", "bronc"],
        }
        parents = self.reader.variable_parents
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable], parents[variable])

    def test_get_states(self):
        states_expected = {
            "asia": ["no", "yes"],
            "tub": ["no", "yes"],
            "smoke": ["no", "yes"],
            "lung": ["no", "yes"],
            "either": ["Nothing", "CancerORTuberculosis"],
            "xray": ["Normal", "Abnormal"],
            "bronc": ["Absent", "Present"],
            "dysp": ["Absent", "Present"],
        }
        states = self.reader.variable_states
        for variable in states_expected:
            self.assertListEqual(states_expected[variable], states[variable])

    def test_get_edges(self):
        edges_expected = [
            ["asia", "tub"],
            ["smoke", "lung"],
            ["tub", "either"],
            ["lung", "either"],
            ["either", "xray"],
            ["smoke", "bronc"],
            ["either", "dysp"],
            ["bronc", "dysp"],
        ]
        self.assertListEqual(sorted(self.reader.edge_list), sorted(edges_expected))

    def test_get_values(self):
        cpd_expected = {
            "asia": np.array([[0.99], [0.01]]),
            "tub": np.array([[0.99, 0.95], [0.01, 0.05]]),
            "smoke": np.array([[0.5], [0.5]]),
            "lung": np.array([[0.99, 0.9], [0.01, 0.1]]),
            "either": np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            "xray": np.array([[0.95, 0.02], [0.05, 0.98]]),
            "bronc": np.array([[0.7, 0.4], [0.3, 0.6]]),
            "dysp": np.array([[0.9, 0.2, 0.3, 0.1], [0.1, 0.8, 0.7, 0.9]]),
        }
        cpd = self.reader.variable_CPD
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable], cpd[variable])

    def test_model(self):
        self.reader.get_model().check_model()

    def tearDown(self):
        del self.reader


DUMMY_FILE = """<?xml version="1.0" encoding="UTF-8"?>
<smile version="1.0" id="dummy" numsamples="10000" discsamples="10000">
    <nodes>
        <cpt id="A" >
            <state id="yes" />
            <state id="no" />
            <probabilities>0.92 0.08</probabilities>
        </cpt>
        <cpt id="B" >
            <state id="high" />
            <state id="low" />
            <probabilities>0.99 0.01</probabilities>
        </cpt>
        <cpt id="C" >
            <state id="true" />
            <state id="false" />
            <parents>A B</parents>
            <probabilities>0.8 0.2 0.75 0.25 0.33 0.67 0.99 0.01</probabilities>
        </cpt>
        <cpt id="D" >
            <state id="big" />
            <state id="medium" />
            <state id="small" />
            <parents>C</parents>
            <probabilities>0.6 0.3 0.1 0.4 0.4 0.2</probabilities>
        </cpt>
    </nodes>
</smile>"""


class TestXDSLWriterMethods(unittest.TestCase):
    def setUp(self):
        self.alarm_model_bn = get_example_model(model="alarm")

        self.dummy_model = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("C", "D")])
        self.cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.92], [0.08]])
        self.cpd_b = TabularCPD(variable="B", variable_card=2, values=[[0.99], [0.01]])

        self.cpd_c = TabularCPD(
            variable="C",
            variable_card=2,
            values=[
                [0.8, 0.75, 0.33, 0.99],
                [0.2, 0.25, 0.67, 0.01],
            ],
            evidence=["A", "B"],
            evidence_card=[2, 2],
        )

        self.cpd_d = TabularCPD(
            variable="D",
            variable_card=3,
            values=[[0.6, 0.4], [0.3, 0.4], [0.1, 0.2]],
            evidence=["C"],
            evidence_card=[2],
        )

        self.dummy_model.add_cpds(
            self.cpd_a, self.cpd_b, self.cpd_c, self.cpd_d
        )  # testing without state names
        self.writer_dummy = XDSLWriter(self.dummy_model)

        self.model_with_whitespaces = DiscreteBayesianNetwork()
        self.model_with_whitespaces.add_nodes_from(["first node", "second node"])
        self.model_with_whitespaces.add_edges_from([("first node", "second node")])
        cpd_a = TabularCPD("first node", 2, [[0.5], [0.5]])
        cpd_b = TabularCPD(
            "second node",
            2,
            [[0.5, 0.5], [0.5, 0.5]],
            evidence=["first node"],
            evidence_card=[2],
        )
        self.model_with_whitespaces.add_cpds(cpd_a, cpd_b)

    def test_whitespace_warning(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.model_with_whitespaces_xdsl = XDSLWriter(self.model_with_whitespaces)

    def assert_models_equivalent(self, expected, got):
        self.assertSetEqual(set(expected.nodes()), set(got.nodes()))
        for node in expected.nodes():
            self.assertListEqual(
                sorted(expected.get_parents(node)), sorted(got.get_parents(node))
            )
            cpds_expected = expected.get_cpds(node=node)
            cpds_got = got.get_cpds(node=node)
            self.assertEqual(cpds_expected, cpds_got)

    def test_writer_cpds(self):
        self.writer_dummy.write_xdsl(filename="dummy_model.xdsl")
        with open("dummy_model.xdsl", "r") as f:
            reader = XDSLReader(f)
        model = reader.get_model(state_name_type=int)
        self.assert_models_equivalent(self.dummy_model, model)
        os.remove("dummy_model.xdsl")

    def test_alarm_model(self):
        XDSLWriter(self.alarm_model_bn).write_xdsl("alarm_model.xdsl")

        with open("alarm_model.xdsl", "r") as f:
            file_text = f.read()
        alarm_model_bn_test = XDSLReader(string=file_text).get_model()
        self.assert_models_equivalent(self.alarm_model_bn, alarm_model_bn_test)

        os.remove("alarm_model.xdsl")

    def tearDown(self):
        del self.alarm_model_bn
        del self.dummy_model
        del self.writer_dummy


@unittest.skipUnless(
    _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestXDSLReaderMethodsStringTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")
        self.reader = XDSLReader(string=TEST_FILE)

    def test_whitespace_error(self):
        with self.assertRaises(ValueError):
            self.model_with_whitespace = XDSLReader(string=TEST_WHITESPACE_MODEL)

    def test_get_variables(self):
        var_expected = [
            "asia",
            "tub",
            "smoke",
            "lung",
            "either",
            "xray",
            "bronc",
            "dysp",
        ]
        self.assertListEqual(self.reader.variables, var_expected)

    def test_get_parents(self):
        parents_expected = {
            "asia": [],
            "tub": ["asia"],
            "smoke": [],
            "lung": ["smoke"],
            "either": ["tub", "lung"],
            "xray": ["either"],
            "bronc": ["smoke"],
            "dysp": ["either", "bronc"],
        }
        parents = self.reader.variable_parents
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable], parents[variable])

    def test_get_states(self):
        states_expected = {
            "asia": ["no", "yes"],
            "tub": ["no", "yes"],
            "smoke": ["no", "yes"],
            "lung": ["no", "yes"],
            "either": ["Nothing", "CancerORTuberculosis"],
            "xray": ["Normal", "Abnormal"],
            "bronc": ["Absent", "Present"],
            "dysp": ["Absent", "Present"],
        }
        states = self.reader.variable_states
        for variable in states_expected:
            self.assertListEqual(states_expected[variable], states[variable])

    def test_get_edges(self):
        edges_expected = [
            ["asia", "tub"],
            ["smoke", "lung"],
            ["tub", "either"],
            ["lung", "either"],
            ["either", "xray"],
            ["smoke", "bronc"],
            ["either", "dysp"],
            ["bronc", "dysp"],
        ]
        self.assertListEqual(sorted(self.reader.edge_list), sorted(edges_expected))

    def test_get_values(self):
        cpd_expected = {
            "asia": np.array([[0.99], [0.01]]),
            "tub": np.array([[0.99, 0.95], [0.01, 0.05]]),
            "smoke": np.array([[0.5], [0.5]]),
            "lung": np.array([[0.99, 0.9], [0.01, 0.1]]),
            "either": np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            "xray": np.array([[0.95, 0.02], [0.05, 0.98]]),
            "bronc": np.array([[0.7, 0.4], [0.3, 0.6]]),
            "dysp": np.array([[0.9, 0.2, 0.3, 0.1], [0.1, 0.8, 0.7, 0.9]]),
        }
        cpd = self.reader.variable_CPD
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable], cpd[variable])

    def test_model(self):
        self.reader.get_model().check_model()

    def tearDown(self):
        del self.reader
        config.set_backend("numpy")


@unittest.skipUnless(
    _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestXDSLWriterMethodsTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.alarm_model_bn = get_example_model(model="alarm")

        self.dummy_model = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("C", "D")])
        self.cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.92], [0.08]])
        self.cpd_b = TabularCPD(variable="B", variable_card=2, values=[[0.99], [0.01]])

        self.cpd_c = TabularCPD(
            variable="C",
            variable_card=2,
            values=[
                [0.8, 0.75, 0.33, 0.99],
                [0.2, 0.25, 0.67, 0.01],
            ],
            evidence=["A", "B"],
            evidence_card=[2, 2],
        )

        self.cpd_d = TabularCPD(
            variable="D",
            variable_card=3,
            values=[[0.6, 0.4], [0.3, 0.4], [0.1, 0.2]],
            evidence=["C"],
            evidence_card=[2],
        )

        self.dummy_model.add_cpds(
            self.cpd_a, self.cpd_b, self.cpd_c, self.cpd_d
        )  # testing without state names
        self.writer_dummy = XDSLWriter(self.dummy_model)

        self.model_with_whitespaces = DiscreteBayesianNetwork()
        self.model_with_whitespaces.add_nodes_from(["first node", "second node"])
        self.model_with_whitespaces.add_edges_from([("first node", "second node")])
        cpd_a = TabularCPD("first node", 2, [[0.5], [0.5]])
        cpd_b = TabularCPD(
            "second node",
            2,
            [[0.5, 0.5], [0.5, 0.5]],
            evidence=["first node"],
            evidence_card=[2],
        )
        self.model_with_whitespaces.add_cpds(cpd_a, cpd_b)

    def test_whitespace_warning(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.model_with_whitespaces_xdsl = XDSLWriter(self.model_with_whitespaces)

    def assert_models_equivalent(self, expected, got):
        self.assertSetEqual(set(expected.nodes()), set(got.nodes()))
        for node in expected.nodes():
            self.assertListEqual(
                sorted(expected.get_parents(node)), sorted(got.get_parents(node))
            )
            cpds_expected = expected.get_cpds(node=node)
            cpds_got = got.get_cpds(node=node)
            self.assertEqual(cpds_expected, cpds_got)

    def test_writer_cpds(self):
        self.writer_dummy.write_xdsl(filename="dummy_model.xdsl")
        with open("dummy_model.xdsl", "r") as f:
            reader = XDSLReader(f)
        model = reader.get_model(state_name_type=int)
        self.assert_models_equivalent(self.dummy_model, model)
        os.remove("dummy_model.xdsl")

    def test_alarm_model(self):
        XDSLWriter(self.alarm_model_bn).write_xdsl("alarm_model.xdsl")
        with open("alarm_model.xdsl", "r") as f:
            file_text = f.read()
        alarm_model_bn_test = XDSLReader(string=file_text).get_model()
        self.assert_models_equivalent(self.alarm_model_bn, alarm_model_bn_test)
        os.remove("alarm_model.xdsl")

    def tearDown(self):
        del self.alarm_model_bn
        del self.dummy_model
        del self.writer_dummy
        config.set_backend("numpy")


class TestXDSLCommaWarning(unittest.TestCase):
    def test_comma_state_name_warning(self):
        # Create a model with state names containing commas
        model = DiscreteBayesianNetwork([("A", "B")])
        cpd_a = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"A": ["state,1", "state,2"]},
        )
        cpd_b = TabularCPD(
            variable="B",
            variable_card=2,
            values=[[0.6, 0.4], [0.4, 0.6]],
            evidence=["A"],
            evidence_card=[2],
            state_names={"B": ["yes", "no"], "A": ["state,1", "state,2"]},
        )
        model.add_cpds(cpd_a, cpd_b)

        # Test that warning is raised when writing
        with tempfile.NamedTemporaryFile(suffix=".xdsl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with self.assertLogs("pgmpy", level="WARNING") as cm:
                writer = XDSLWriter(model)
                writer.write_xdsl(tmp_path)

                # Verify the warning was logged
                self.assertIn(
                    "State name 'state,1' for variable 'A' contains commas. "
                    "This may cause issues when loading the file. Consider removing any special characters.",
                    cm.output[0],
                )

            # Verify that the file can be loaded back with the same state names
            reader = XDSLReader(tmp_path)
            loaded_model = reader.get_model()

            # Check that the state names were preserved
            self.assertEqual(
                loaded_model.get_cpds("A").state_names["A"], ["state,1", "state,2"]
            )
            self.assertEqual(
                loaded_model.get_cpds("B").state_names["A"], ["state,1", "state,2"]
            )
            self.assertEqual(loaded_model.get_cpds("B").state_names["B"], ["yes", "no"])
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
