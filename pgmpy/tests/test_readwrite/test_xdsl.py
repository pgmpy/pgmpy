import os
import unittest
import xml.etree.ElementTree as etree

import numpy as np
import numpy.testing as np_test

from pgmpy import config
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite.XDSL import XDSLReader, XDSLWriter
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


class TestXDSLReaderMethodsString(unittest.TestCase):
    def setUp(self):
        self.reader = XDSLReader(string=TEST_FILE)

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


class TestXDSLWriterMethodsString(unittest.TestCase):
    def setUp(self):
        self.alarm_model_xdsl = r"pgmpy\tests\test_readwrite\testdata\Alarm.xdsl"
        self.alarm_model_bn = get_example_model(model="alarm")
        self.asia_model_xdsl = TEST_FILE

    def test_writer_cpds(self):
        # asia_model = XDSLWriter(self.asia_model_bn)
        # asia_cpd = TabularCPD()
        pass

    def assert_models_equivalent(self, expected, got):
        self.assertSetEqual(set(expected.nodes()), set(got.nodes()))
        for node in expected.nodes():
            self.assertListEqual(
                sorted(expected.get_parents(node)), sorted(got.get_parents(node))
            )
            cpds_expected = expected.get_cpds(node=node)
            cpds_got = got.get_cpds(node=node)
            self.assertEqual(cpds_expected, cpds_got)

    def test_write_xdsl(self):
        alarm_xdsl = XDSLWriter(self.alarm_model_bn).write_xdsl("alarm_model.xdsl")
        with open("alarm_model.xdsl", "r") as f:
            file_text = f.read()
        alarm_model_bn_test = XDSLReader(string=file_text).get_model()
        self.assert_models_equivalent(self.alarm_model_bn, alarm_model_bn_test)
        os.remove("alarm_model.xdsl")
