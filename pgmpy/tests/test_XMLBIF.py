import unittest
from pgmpy.readwrite import XMLBIFReader
import numpy as np
import numpy.testing as np_test
import os


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
        self.assertListEqual(self.reader.get_variables(), var_expected)

    def test_get_states(self):
        states_expected = {'bowel-problem': ['true', 'false'],
                           'dog-out': ['true', 'false'],
                           'family-out': ['true', 'false'],
                           'hear-bark': ['true', 'false'],
                           'light-on': ['true', 'false']}
        states = self.reader.get_states()
        for variable in states_expected:
            self.assertListEqual(states_expected[variable],
                                 states[variable])

    def test_get_parents(self):
        parents_expected = {'bowel-problem': [],
                            'dog-out': ['family-out', 'bowel-problem'],
                            'family-out': [],
                            'hear-bark': ['dog-out'],
                            'light-on': ['family-out']}
        parents = self.reader.get_parents()
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable],
                                 parents[variable])

    def test_get_edges(self):
        edges_expected = [['family-out', 'dog-out'],
                          ['bowel-problem', 'dog-out'],
                          ['family-out', 'light-on'],
                          ['dog-out', 'hear-bark']]
        self.assertListEqual(sorted(self.reader.get_edges()),
                             sorted(edges_expected))

    def test_get_cpd(self):
        cpd_expected = {'bowel-problem': np.array([[0.01],
                                                   [0.99]]),
                        'dog-out': np.array([[0.99, 0.01, 0.97, 0.03],
                                             [0.9, 0.1, 0.3, 0.7]]),
                        'family-out': np.array([[0.15],
                                                [0.85]]),
                        'hear-bark':  np.array([[0.7,  0.3],
                                                [0.01,  0.99]]),
                        'light-on': np.array([[0.6,  0.4],
                                              [0.05,  0.95]])}
        cpd = self.reader.get_cpd()
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable],
                                       cpd[variable])

    def test_get_property(self):
        property_expected = {'bowel-problem': ['position = (190, 69)'],
                             'dog-out': ['position = (155, 165)'],
                             'family-out': ['position = (112, 69)'],
                             'hear-bark': ['position = (154, 241)'],
                             'light-on': ['position = (73, 165)']}
        property = self.reader.get_property()
        for variable in property_expected:
            self.assertListEqual(property_expected[variable],
                                 property[variable])

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
        self.assertListEqual(self.reader.get_variables(), var_expected)

    def test_get_states(self):
        states_expected = {'bowel-problem': ['true', 'false'],
                           'dog-out': ['true', 'false'],
                           'family-out': ['true', 'false'],
                           'hear-bark': ['true', 'false'],
                           'light-on': ['true', 'false']}
        states = self.reader.get_states()
        for variable in states_expected:
            self.assertListEqual(states_expected[variable],
                                 states[variable])

    def test_get_parents(self):
        parents_expected = {'bowel-problem': [],
                            'dog-out': ['family-out', 'bowel-problem'],
                            'family-out': [],
                            'hear-bark': ['dog-out'],
                            'light-on': ['family-out']}
        parents = self.reader.get_parents()
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable],
                                 parents[variable])

    def test_get_edges(self):
        edges_expected = [['family-out', 'dog-out'],
                          ['bowel-problem', 'dog-out'],
                          ['family-out', 'light-on'],
                          ['dog-out', 'hear-bark']]
        self.assertListEqual(sorted(self.reader.get_edges()),
                             sorted(edges_expected))

    def test_get_cpd(self):
        cpd_expected = {'bowel-problem': np.array([[0.01],
                                                   [0.99]]),
                        'dog-out': np.array([[0.99, 0.01, 0.97, 0.03],
                                             [0.9, 0.1, 0.3, 0.7]]),
                        'family-out': np.array([[0.15],
                                                [0.85]]),
                        'hear-bark':  np.array([[0.7,  0.3],
                                                [0.01,  0.99]]),
                        'light-on': np.array([[0.6,  0.4],
                                              [0.05,  0.95]])}
        cpd = self.reader.get_cpd()
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable],
                                       cpd[variable])

    def test_get_property(self):
        property_expected = {'bowel-problem': ['position = (190, 69)'],
                             'dog-out': ['position = (155, 165)'],
                             'family-out': ['position = (112, 69)'],
                             'hear-bark': ['position = (154, 241)'],
                             'light-on': ['position = (73, 165)']}
        property = self.reader.get_property()
        for variable in property_expected:
            self.assertListEqual(property_expected[variable],
                                 property[variable])

    def tearDown(self):
        del self.reader
        os.remove("test_bif.xml")

