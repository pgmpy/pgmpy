import os
import unittest
import warnings
import numpy as np
import numpy.testing as np_test

from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD


try:
    from lxml import etree
except ImportError:
    try:
        import xml.etree.cElementTree as etree
    except ImportError:
        try:
            import xml.etree.ElementTree as etree
        except ImportError:
            warnings.warn("Failed to import ElementTree from any known place")

TEST_FILE = """<?xml version="1.0" encoding="US-ASCII"?>


<!--
    Bayesian network in XMLBIF v0.3 (BayesNet Interchange Format)
    Produced by JavaBayes (http://www.cs.cmu.edu/~javabayes/
    Output created Mon Aug 01 10:33:28 AEST 2016
-->



<!-- DTD for the XMLBIF 0.3 format -->
<!DOCTYPE BIF [
    <!ELEMENT BIF ( NETWORK )*>
          <!ATTLIST BIF VERSION CDATA #REQUIRED>
    <!ELEMENT NETWORK ( NAME, ( PROPERTY | VARIABLE | DEFINITION )* )>
    <!ELEMENT NAME (#PCDATA)>
    <!ELEMENT VARIABLE ( NAME, ( OUTCOME |  PROPERTY )* ) >
          <!ATTLIST VARIABLE TYPE (nature|decision|utility) "nature">
    <!ELEMENT OUTCOME (#PCDATA)>
    <!ELEMENT DEFINITION ( FOR | GIVEN | TABLE | PROPERTY )* >
    <!ELEMENT FOR (#PCDATA)>
    <!ELEMENT GIVEN (#PCDATA)>
    <!ELEMENT TABLE (#PCDATA)>
    <!ELEMENT PROPERTY (#PCDATA)>
]>


<BIF VERSION="0.3">
<NETWORK>
<NAME>Dog_Problem</NAME>

<!-- Variables -->
<VARIABLE TYPE="nature">
    <NAME>light_on</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (73, 165)</PROPERTY>
</VARIABLE>

<VARIABLE TYPE="nature">
    <NAME>bowel_problem</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (190, 69)</PROPERTY>
</VARIABLE>

<VARIABLE TYPE="nature">
    <NAME>dog_out</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (155, 165)</PROPERTY>
</VARIABLE>

<VARIABLE TYPE="nature">
    <NAME>hear_bark</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (154, 241)</PROPERTY>
</VARIABLE>

<VARIABLE TYPE="nature">
    <NAME>family_out</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (112, 69)</PROPERTY>
</VARIABLE>


<!-- Probability distributions -->
<DEFINITION>
    <FOR>light_on</FOR>
    <GIVEN>family_out</GIVEN>
    <TABLE>0.6 0.4 0.05 0.95 </TABLE>
</DEFINITION>

<DEFINITION>
    <FOR>bowel_problem</FOR>
    <TABLE>0.01 0.99 </TABLE>
</DEFINITION>

<DEFINITION>
    <FOR>dog_out</FOR>
    <GIVEN>bowel_problem</GIVEN>
    <GIVEN>family_out</GIVEN>
    <TABLE>0.99 0.01 0.97 0.03 0.9 0.1 0.3 0.7 </TABLE>
</DEFINITION>

<DEFINITION>
    <FOR>hear_bark</FOR>
    <GIVEN>dog_out</GIVEN>
    <TABLE>0.7 0.3 0.01 0.99 </TABLE>
</DEFINITION>

<DEFINITION>
    <FOR>family_out</FOR>
    <TABLE>0.15 0.85 </TABLE>
</DEFINITION>


</NETWORK>
</BIF>"""


class TestXMLBIFReaderMethods(unittest.TestCase):

    def setUp(self):
        self.reader = XMLBIFReader(string=TEST_FILE)

    def test_get_variables(self):
        var_expected = ['light_on', 'bowel_problem', 'dog_out',
                        'hear_bark', 'family_out']
        self.assertListEqual(self.reader.variables, var_expected)

    def test_get_states(self):
        states_expected = {'bowel_problem': ['true', 'false'],
                           'dog_out': ['true', 'false'],
                           'family_out': ['true', 'false'],
                           'hear_bark': ['true', 'false'],
                           'light_on': ['true', 'false']}
        states = self.reader.variable_states
        for variable in states_expected:
            self.assertListEqual(states_expected[variable],
                                 states[variable])

    def test_get_parents(self):
        parents_expected = {'bowel_problem': [],
                            'dog_out': ['bowel_problem', 'family_out'],
                            'family_out': [],
                            'hear_bark': ['dog_out'],
                            'light_on': ['family_out']}
        parents = self.reader.variable_parents
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable],
                                 parents[variable])

    def test_get_edges(self):
        edges_expected = [['family_out', 'dog_out'],
                          ['bowel_problem', 'dog_out'],
                          ['family_out', 'light_on'],
                          ['dog_out', 'hear_bark']]
        self.assertListEqual(sorted(self.reader.edge_list),
                             sorted(edges_expected))

    def test_get_values(self):
        cpd_expected = {'bowel_problem': np.array([[0.01],
                                                   [0.99]]),
                        'dog_out': np.array([[0.99, 0.97, 0.9, 0.3],
                                             [0.01, 0.03, 0.1, 0.7]]),
                        'family_out': np.array([[0.15],
                                                [0.85]]),
                        'hear_bark': np.array([[0.7, 0.01],
                                               [0.3, 0.99]]),
                        'light_on': np.array([[0.6, 0.05],
                                              [0.4, 0.95]])}
        cpd = self.reader.variable_CPD
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable],
                                       cpd[variable])

    def test_get_property(self):
        property_expected = {'bowel_problem': ['position = (190, 69)'],
                             'dog_out': ['position = (155, 165)'],
                             'family_out': ['position = (112, 69)'],
                             'hear_bark': ['position = (154, 241)'],
                             'light_on': ['position = (73, 165)']}
        prop = self.reader.variable_property
        for variable in property_expected:
            self.assertListEqual(property_expected[variable],
                                 prop[variable])

    def test_model(self):
        self.reader.get_model().check_model()

    def tearDown(self):
        del self.reader


class TestXMLBIFReaderMethodsFile(unittest.TestCase):

    def setUp(self):
        with open("dog_problem.xml", 'w') as fout:
            fout.write(TEST_FILE)
        self.reader = XMLBIFReader("dog_problem.xml")

    def test_get_variables(self):
        var_expected = ['light_on', 'bowel_problem', 'dog_out',
                        'hear_bark', 'family_out']
        self.assertListEqual(self.reader.variables, var_expected)

    def test_get_states(self):
        states_expected = {'bowel_problem': ['true', 'false'],
                           'dog_out': ['true', 'false'],
                           'family_out': ['true', 'false'],
                           'hear_bark': ['true', 'false'],
                           'light_on': ['true', 'false']}
        states = self.reader.variable_states
        for variable in states_expected:
            self.assertListEqual(states_expected[variable],
                                 states[variable])

    def test_get_parents(self):
        parents_expected = {'bowel_problem': [],
                            'dog_out': ['bowel_problem', 'family_out'],
                            'family_out': [],
                            'hear_bark': ['dog_out'],
                            'light_on': ['family_out']}
        parents = self.reader.variable_parents
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable],
                                 parents[variable])

    def test_get_edges(self):
        edges_expected = [['family_out', 'dog_out'],
                          ['bowel_problem', 'dog_out'],
                          ['family_out', 'light_on'],
                          ['dog_out', 'hear_bark']]
        self.assertListEqual(sorted(self.reader.edge_list),
                             sorted(edges_expected))

    def test_get_values(self):
        cpd_expected = {'bowel_problem': np.array([[0.01],
                                                   [0.99]]),
                        'dog_out': np.array([[0.99, 0.97, 0.9, 0.3],
                                             [0.01, 0.03, 0.1, 0.7]]),
                        'family_out': np.array([[0.15],
                                                [0.85]]),
                        'hear_bark': np.array([[0.7, 0.01],
                                               [0.3, 0.99]]),
                        'light_on': np.array([[0.6, 0.05],
                                              [0.4, 0.95]])}
        cpd = self.reader.variable_CPD
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable],
                                       cpd[variable])

    def test_get_property(self):
        property_expected = {'bowel_problem': ['position = (190, 69)'],
                             'dog_out': ['position = (155, 165)'],
                             'family_out': ['position = (112, 69)'],
                             'hear_bark': ['position = (154, 241)'],
                             'light_on': ['position = (73, 165)']}
        prop = self.reader.variable_property
        for variable in property_expected:
            self.assertListEqual(property_expected[variable],
                                 prop[variable])

    def test_model(self):
        self.reader.get_model().check_model()

    def tearDown(self):
        del self.reader
        os.remove("dog_problem.xml")


class TestXMLBIFWriterMethodsString(unittest.TestCase):

    def setUp(self):
        reader = XMLBIFReader(string=TEST_FILE)
        self.expected_model = reader.get_model()
        self.writer = XMLBIFWriter(self.expected_model)

        self.model_stateless = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])
        self.cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6, 0.4]])
        self.cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.3]])

        self.cpd_g = TabularCPD(variable='G', variable_card=3,
                                values=[[0.3, 0.05, 0.9,  0.5],
                                        [0.4, 0.25, 0.08, 0.3],
                                        [0.3, 0.7,  0.02, 0.2]],
                                evidence=['I', 'D'],
                                evidence_card=[2, 2])

        self.cpd_l = TabularCPD(variable='L', variable_card=2,
                                values=[[0.1, 0.4, 0.99],
                                        [0.9, 0.6, 0.01]],
                                evidence=['G'],
                                evidence_card=[3])

        self.cpd_s = TabularCPD(variable='S', variable_card=2,
                                values=[[0.95, 0.2],
                                        [0.05, 0.8]],
                                evidence=['I'],
                                evidence_card=[2])

        self.model_stateless.add_cpds(self.cpd_d, self.cpd_i, self.cpd_g, self.cpd_l, self.cpd_s)
        self.writer_stateless = XMLBIFWriter(self.model_stateless)

    def test_write_xmlbif_statefull(self):
        self.writer.write_xmlbif("dog_problem_output.xbif")
        with open("dog_problem_output.xbif", "r") as f:
            file_text = f.read()
        reader = XMLBIFReader(string=file_text)
        model = reader.get_model()
        self.assert_models_equivelent(self.expected_model, model)
        os.remove("dog_problem_output.xbif")

    def test_write_xmlbif_stateless(self):
        self.writer_stateless.write_xmlbif("grade_problem_output.xbif")
        with open("grade_problem_output.xbif", 'r') as f:
            reader = XMLBIFReader(f)
        model = reader.get_model()
        self.assert_models_equivelent(self.model_stateless, model)
        self.assertDictEqual({'G': ['state0', 'state1', 'state2'],
                              'I': ['state0', 'state1'],
                              'D': ['state0', 'state1'],
                              'S': ['state0', 'state1'],
                              'L': ['state0', 'state1']},
                             model.get_cpds('D').state_names)
        os.remove("grade_problem_output.xbif")

    def assert_models_equivelent(self, expected, got):
        self.assertSetEqual(set(expected.nodes()), set(got.nodes()))
        for node in expected.nodes():
            self.assertListEqual(expected.get_parents(node), got.get_parents(node))
            cpds_expected = expected.get_cpds(node=node)
            cpds_got = got.get_cpds(node=node)
            np_test.assert_array_equal(cpds_expected.values, cpds_got.values)
