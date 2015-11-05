import os
import unittest
import warnings

import numpy as np
import numpy.testing as np_test

from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
from pgmpy.extern.six.moves import map

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


class TestXMLBIFReaderMethodsString(unittest.TestCase):
    def setUp(self):
        self.reader = XMLBIFReader(string="""
        <BIF VERSION="0.3">
        <NETWORK>
        <NAME>Dog-Problem</NAME>

        <VARIABLE TYPE="nature">
            <NAME>light-on</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (73, 165)</PROPERTY>
        </VARIABLE>

        <VARIABLE TYPE="nature">
            <NAME>bowel-problem</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (190, 69)</PROPERTY>
        </VARIABLE>

        <VARIABLE TYPE="nature">
            <NAME>dog-out</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (155, 165)</PROPERTY>
        </VARIABLE>

        <VARIABLE TYPE="nature">
            <NAME>hear-bark</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (154, 241)</PROPERTY>
        </VARIABLE>

        <VARIABLE TYPE="nature">
            <NAME>family-out</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (112, 69)</PROPERTY>
        </VARIABLE>


        <DEFINITION>
            <FOR>light-on</FOR>
            <GIVEN>family-out</GIVEN>
            <TABLE>0.6 0.4 0.05 0.95 </TABLE>
        </DEFINITION>

        <DEFINITION>
            <FOR>bowel-problem</FOR>
            <TABLE>0.01 0.99 </TABLE>
        </DEFINITION>

        <DEFINITION>
            <FOR>dog-out</FOR>
            <GIVEN>bowel-problem</GIVEN>
            <GIVEN>family-out</GIVEN>
            <TABLE>0.99 0.01 0.97 0.03 0.9 0.1 0.3  0.7 </TABLE>
        </DEFINITION>

        <DEFINITION>
            <FOR>hear-bark</FOR>
            <GIVEN>dog-out</GIVEN>
            <TABLE>0.7 0.3 0.01 0.99 </TABLE>
        </DEFINITION>

        <DEFINITION>
            <FOR>family-out</FOR>
            <TABLE>0.15 0.85 </TABLE>
        </DEFINITION>


        </NETWORK>
        </BIF>
        """)

    def test_get_variables(self):
        var_expected = ['light-on', 'bowel-problem', 'dog-out',
                        'hear-bark', 'family-out']
        self.assertListEqual(self.reader.variables, var_expected)

    def test_get_states(self):
        states_expected = {'bowel-problem': ['true', 'false'],
                           'dog-out': ['true', 'false'],
                           'family-out': ['true', 'false'],
                           'hear-bark': ['true', 'false'],
                           'light-on': ['true', 'false']}
        states = self.reader.variable_states
        for variable in states_expected:
            self.assertListEqual(states_expected[variable],
                                 states[variable])

    def test_get_parents(self):
        parents_expected = {'bowel-problem': [],
                            'dog-out': ['family-out', 'bowel-problem'],
                            'family-out': [],
                            'hear-bark': ['dog-out'],
                            'light-on': ['family-out']}
        parents = self.reader.variable_parents
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable],
                                 parents[variable])

    def test_get_edges(self):
        edges_expected = [['family-out', 'dog-out'],
                          ['bowel-problem', 'dog-out'],
                          ['family-out', 'light-on'],
                          ['dog-out', 'hear-bark']]
        self.assertListEqual(sorted(self.reader.edge_list),
                             sorted(edges_expected))

    def test_get_cpd(self):
        cpd_expected = {'bowel-problem': np.array([[0.01],
                                                   [0.99]]),
                        'dog-out': np.array([[0.99, 0.01, 0.97, 0.03],
                                             [0.9, 0.1, 0.3, 0.7]]),
                        'family-out': np.array([[0.15],
                                                [0.85]]),
                        'hear-bark': np.array([[0.7, 0.3],
                                               [0.01, 0.99]]),
                        'light-on': np.array([[0.6, 0.4],
                                              [0.05, 0.95]])}
        cpd = self.reader.variable_CPD
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable],
                                       cpd[variable])

    def test_get_property(self):
        property_expected = {'bowel-problem': ['position = (190, 69)'],
                             'dog-out': ['position = (155, 165)'],
                             'family-out': ['position = (112, 69)'],
                             'hear-bark': ['position = (154, 241)'],
                             'light-on': ['position = (73, 165)']}
        prop = self.reader.variable_property
        for variable in property_expected:
            self.assertListEqual(property_expected[variable],
                                 prop[variable])

    def tearDown(self):
        del self.reader


class TestXMLBIFReaderMethodsFile(unittest.TestCase):
    def setUp(self):
        xml = """
        <BIF VERSION="0.3">
        <NETWORK>
        <NAME>Dog-Problem</NAME>

        <VARIABLE TYPE="nature">
            <NAME>light-on</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (73, 165)</PROPERTY>
        </VARIABLE>

        <VARIABLE TYPE="nature">
            <NAME>bowel-problem</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (190, 69)</PROPERTY>
        </VARIABLE>

        <VARIABLE TYPE="nature">
            <NAME>dog-out</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (155, 165)</PROPERTY>
        </VARIABLE>

        <VARIABLE TYPE="nature">
            <NAME>hear-bark</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (154, 241)</PROPERTY>
        </VARIABLE>

        <VARIABLE TYPE="nature">
            <NAME>family-out</NAME>
            <OUTCOME>true</OUTCOME>
            <OUTCOME>false</OUTCOME>
            <PROPERTY>position = (112, 69)</PROPERTY>
        </VARIABLE>


        <DEFINITION>
            <FOR>light-on</FOR>
            <GIVEN>family-out</GIVEN>
            <TABLE>0.6 0.4 0.05 0.95 </TABLE>
        </DEFINITION>

        <DEFINITION>
            <FOR>bowel-problem</FOR>
            <TABLE>0.01 0.99 </TABLE>
        </DEFINITION>

        <DEFINITION>
            <FOR>dog-out</FOR>
            <GIVEN>bowel-problem</GIVEN>
            <GIVEN>family-out</GIVEN>
            <TABLE>0.99 0.01 0.97 0.03 0.9 0.1 0.3  0.7 </TABLE>
        </DEFINITION>

        <DEFINITION>
            <FOR>hear-bark</FOR>
            <GIVEN>dog-out</GIVEN>
            <TABLE>0.7 0.3 0.01 0.99 </TABLE>
        </DEFINITION>

        <DEFINITION>
            <FOR>family-out</FOR>
            <TABLE>0.15 0.85 </TABLE>
        </DEFINITION>


        </NETWORK>
        </BIF>
        """
        with open("test_bif.xml", 'w') as fout:
            fout.write(xml)
        self.reader = XMLBIFReader("test_bif.xml")

    def test_get_variables(self):
        var_expected = ['light-on', 'bowel-problem', 'dog-out',
                        'hear-bark', 'family-out']
        self.assertListEqual(self.reader.variables, var_expected)

    def test_get_states(self):
        states_expected = {'bowel-problem': ['true', 'false'],
                           'dog-out': ['true', 'false'],
                           'family-out': ['true', 'false'],
                           'hear-bark': ['true', 'false'],
                           'light-on': ['true', 'false']}
        states = self.reader.variable_states
        for variable in states_expected:
            self.assertListEqual(states_expected[variable],
                                 states[variable])

    def test_get_parents(self):
        parents_expected = {'bowel-problem': [],
                            'dog-out': ['family-out', 'bowel-problem'],
                            'family-out': [],
                            'hear-bark': ['dog-out'],
                            'light-on': ['family-out']}
        parents = self.reader.variable_parents
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable],
                                 parents[variable])

    def test_get_edges(self):
        edges_expected = [['family-out', 'dog-out'],
                          ['bowel-problem', 'dog-out'],
                          ['family-out', 'light-on'],
                          ['dog-out', 'hear-bark']]
        self.assertListEqual(sorted(self.reader.edge_list),
                             sorted(edges_expected))

    def test_get_cpd(self):
        cpd_expected = {'bowel-problem': np.array([[0.01],
                                                   [0.99]]),
                        'dog-out': np.array([[0.99, 0.01, 0.97, 0.03],
                                             [0.9, 0.1, 0.3, 0.7]]),
                        'family-out': np.array([[0.15],
                                                [0.85]]),
                        'hear-bark': np.array([[0.7, 0.3],
                                               [0.01, 0.99]]),
                        'light-on': np.array([[0.6, 0.4],
                                              [0.05, 0.95]])}
        cpd = self.reader.variable_CPD
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable],
                                       cpd[variable])

    def test_get_property(self):
        property_expected = {'bowel-problem': ['position = (190, 69)'],
                             'dog-out': ['position = (155, 165)'],
                             'family-out': ['position = (112, 69)'],
                             'hear-bark': ['position = (154, 241)'],
                             'light-on': ['position = (73, 165)']}
        prop = self.reader.variable_property
        for variable in property_expected:
            self.assertListEqual(property_expected[variable],
                                 prop[variable])

    def tearDown(self):
        del self.reader
        os.remove("test_bif.xml")


class TestXMLBIFWriterMethodsString(unittest.TestCase):
    def setUp(self):
        edges = [['family-out', 'dog-out'],
                 ['bowel-problem', 'dog-out'],
                 ['family-out', 'light-on'],
                 ['dog-out', 'hear-bark']]
        cpds = {'bowel-problem': np.array([[0.01],
                                           [0.99]]),
                'dog-out': np.array([[0.99, 0.01, 0.97, 0.03],
                                     [0.9, 0.1, 0.3, 0.7]]),
                'family-out': np.array([[0.15],
                                        [0.85]]),
                'hear-bark': np.array([[0.7, 0.3],
                                       [0.01, 0.99]]),
                'light-on': np.array([[0.6, 0.4],
                                      [0.05, 0.95]])}
        states = {'bowel-problem': ['true', 'false'],
                  'dog-out': ['true', 'false'],
                  'family-out': ['true', 'false'],
                  'hear-bark': ['true', 'false'],
                  'light-on': ['true', 'false']}
        parents = {'bowel-problem': [],
                   'dog-out': ['family-out', 'bowel-problem'],
                   'family-out': [],
                   'hear-bark': ['dog-out'],
                   'light-on': ['family-out']}
        properties = {'bowel-problem': ['position = (190, 69)'],
                      'dog-out': ['position = (155, 165)'],
                      'family-out': ['position = (112, 69)'],
                      'hear-bark': ['position = (154, 241)'],
                      'light-on': ['position = (73, 165)']}

        self.model = BayesianModel(edges)

        tabular_cpds = []
        for var, values in cpds.items():
            cpd = TabularCPD(var, len(states[var]), values,
                             evidence=parents[var],
                             evidence_card=[len(states[evidence_var])
                                            for evidence_var in parents[var]])
            tabular_cpds.append(cpd)
        self.model.add_cpds(*tabular_cpds)

        for node, properties in properties.items():
            for prop in properties:
                prop_name, prop_value = map(lambda t: t.strip(), prop.split('='))
                self.model.node[node][prop_name] = prop_value

        self.writer = XMLBIFWriter(model=self.model)

    def test_file(self):
        self.expected_xml = etree.XML("""<BIF version="0.3">
  <NETWORK>
    <VARIABLE TYPE="nature">
      <NAME>bowel-problem</NAME>
      <OUTCOME>0</OUTCOME>
      <OUTCOME>1</OUTCOME>
      <PROPERTY>position = (190, 69)</PROPERTY>
    </VARIABLE>
    <VARIABLE TYPE="nature">
      <NAME>dog-out</NAME>
      <OUTCOME>0</OUTCOME>
      <OUTCOME>1</OUTCOME>
      <PROPERTY>position = (155, 165)</PROPERTY>
    </VARIABLE>
    <VARIABLE TYPE="nature">
      <NAME>family-out</NAME>
      <OUTCOME>0</OUTCOME>
      <OUTCOME>1</OUTCOME>
      <PROPERTY>position = (112, 69)</PROPERTY>
    </VARIABLE>
    <VARIABLE TYPE="nature">
      <NAME>hear-bark</NAME>
      <OUTCOME>0</OUTCOME>
      <OUTCOME>1</OUTCOME>
      <PROPERTY>position = (154, 241)</PROPERTY>
    </VARIABLE>
    <VARIABLE TYPE="nature">
      <NAME>light-on</NAME>
      <OUTCOME>0</OUTCOME>
      <OUTCOME>1</OUTCOME>
      <PROPERTY>position = (73, 165)</PROPERTY>
    </VARIABLE>
    <DEFINITION>
      <FOR>bowel-problem</FOR>
      <TABLE>0.01 0.99 </TABLE>
    </DEFINITION>
    <DEFINITION>
      <FOR>dog-out</FOR>
      <GIVEN>bowel-problem</GIVEN>
      <GIVEN>family-out</GIVEN>
      <TABLE>0.99 0.01 0.97 0.03 0.9 0.1 0.3 0.7 </TABLE>
    </DEFINITION>
    <DEFINITION>
      <FOR>family-out</FOR>
      <TABLE>0.15 0.85 </TABLE>
    </DEFINITION>
    <DEFINITION>
      <FOR>hear-bark</FOR>
      <GIVEN>dog-out</GIVEN>
      <TABLE>0.7 0.3 0.01 0.99 </TABLE>
    </DEFINITION>
    <DEFINITION>
      <FOR>light-on</FOR>
      <GIVEN>family-out</GIVEN>
      <TABLE>0.6 0.4 0.05 0.95 </TABLE>
    </DEFINITION>
  </NETWORK>
</BIF>""")
        self.maxDiff = None
        self.writer.write_xmlbif("test_bif.xml")
        with open("test_bif.xml", "r") as myfile:
            data = myfile.read()
        self.assertEqual(str(self.writer.__str__()[:-1]), str(etree.tostring(self.expected_xml)))
        self.assertEqual(str(data), str(etree.tostring(self.expected_xml).decode('utf-8')))
        
