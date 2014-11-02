import unittest
from io import StringIO
import numpy as np
import numpy.testing as np_test
from pgmpy.readwrite import XMLBeliefNetwork


class TestXBNReader(unittest.TestCase):
    def setUp(self):
        string = """<ANALYSISNOTEBOOK NAME="Notebook.Cancer Example From Neapolitan" ROOT="Cancer">
                       <BNMODEL NAME="Cancer">
                          <STATICPROPERTIES>
                             <FORMAT VALUE="MSR DTAS XML"/>
                             <VERSION VALUE="0.2"/>
                             <CREATOR VALUE="Microsoft Research DTAS"/>
                          </STATICPROPERTIES>
                          <VARIABLES>
                             <VAR NAME="a" TYPE="discrete" XPOS="13495" YPOS="10465">
                                <DESCRIPTION>(a) Metastatic Cancer</DESCRIPTION>
                                <STATENAME>Present</STATENAME>
                                <STATENAME>Absent</STATENAME>
                             </VAR>
                             <VAR NAME="b" TYPE="discrete" XPOS="11290" YPOS="11965">
                                <DESCRIPTION>(b) Serum Calcium Increase</DESCRIPTION>
                                <STATENAME>Present</STATENAME>
                                <STATENAME>Absent</STATENAME>
                             </VAR>
                             <VAR NAME="c" TYPE="discrete" XPOS="15250" YPOS="11935">
                                <DESCRIPTION>(c) Brain Tumor</DESCRIPTION>
                                <STATENAME>Present</STATENAME>
                                <STATENAME>Absent</STATENAME>
                             </VAR>
                             <VAR NAME="d" TYPE="discrete" XPOS="13960" YPOS="12985">
                                <DESCRIPTION>(d) Coma</DESCRIPTION>
                                <STATENAME>Present</STATENAME>
                                <STATENAME>Absent</STATENAME>
                             </VAR>
                             <VAR NAME="e" TYPE="discrete" XPOS="17305" YPOS="13240">
                                <DESCRIPTION>(e) Papilledema</DESCRIPTION>
                                <STATENAME>Present</STATENAME>
                                <STATENAME>Absent</STATENAME>
                             </VAR>
                          </VARIABLES>
                          <STRUCTURE>
                             <ARC PARENT="a" CHILD="b"/>
                             <ARC PARENT="a" CHILD="c"/>
                             <ARC PARENT="b" CHILD="d"/>
                             <ARC PARENT="c" CHILD="d"/>
                             <ARC PARENT="c" CHILD="e"/>
                          </STRUCTURE>
                          <DISTRIBUTIONS>
                             <DIST TYPE="discrete">
                                <PRIVATE NAME="a"/>
                                <DPIS>
                                   <DPI> 0.2 0.8</DPI>
                                </DPIS>
                             </DIST>
                             <DIST TYPE="discrete">
                                <CONDSET>
                                   <CONDELEM NAME="a"/>
                                </CONDSET>
                                <PRIVATE NAME="b"/>
                                <DPIS>
                                   <DPI INDEXES=" 0 "> 0.8 0.2</DPI>
                                   <DPI INDEXES=" 1 "> 0.2 0.8</DPI>
                                </DPIS>
                             </DIST>
                             <DIST TYPE="discrete">
                                <CONDSET>
                                   <CONDELEM NAME="a"/>
                                </CONDSET>
                                <PRIVATE NAME="c"/>
                                <DPIS>
                                   <DPI INDEXES=" 0 "> 0.2 0.8</DPI>
                                   <DPI INDEXES=" 1 "> 0.05 0.95</DPI>
                                </DPIS>
                             </DIST>
                             <DIST TYPE="discrete">
                                <CONDSET>
                                   <CONDELEM NAME="b"/>
                                   <CONDELEM NAME="c"/>
                                </CONDSET>
                                <PRIVATE NAME="d"/>
                                <DPIS>
                                   <DPI INDEXES=" 0 0 "> 0.8 0.2</DPI>
                                   <DPI INDEXES=" 0 1 "> 0.9 0.1</DPI>
                                   <DPI INDEXES=" 1 0 "> 0.7 0.3</DPI>
                                   <DPI INDEXES=" 1 1 "> 0.05 0.95</DPI>
                                </DPIS>
                             </DIST>
                             <DIST TYPE="discrete">
                                <CONDSET>
                                   <CONDELEM NAME="c"/>
                                </CONDSET>
                                <PRIVATE NAME="e"/>
                                <DPIS>
                                   <DPI INDEXES=" 0 "> 0.8 0.2</DPI>
                                   <DPI INDEXES=" 1 "> 0.6 0.4</DPI>
                                </DPIS>
                             </DIST>
                          </DISTRIBUTIONS>
                       </BNMODEL>
                    </ANALYSISNOTEBOOK>"""

        self.reader_string = XMLBeliefNetwork.XBNReader(string=string)
        self.reader_file = XMLBeliefNetwork.XBNReader(path=StringIO(string))

    def test_init_exception(self):
        self.assertRaises(ValueError, XMLBeliefNetwork.XBNReader)

    def test_get_analysis_notebook(self):
        self.assertEqual(self.reader_string.get_analysisnotebook_values()['NAME'],
                         "Notebook.Cancer Example From Neapolitan")
        self.assertEqual(self.reader_string.get_analysisnotebook_values()['ROOT'], "Cancer")
        self.assertEqual(self.reader_file.get_analysisnotebook_values()['NAME'],
                         "Notebook.Cancer Example From Neapolitan")
        self.assertEqual(self.reader_file.get_analysisnotebook_values()['ROOT'], "Cancer")

    def test_get_bnmodel_name(self):
        self.assertEqual(self.reader_string.get_bnmodel_name(), "Cancer")
        self.assertEqual(self.reader_file.get_bnmodel_name(), "Cancer")

    def test_get_static_properties(self):
        properties = self.reader_string.get_static_properties()
        self.assertEqual(properties['FORMAT'], "MSR DTAS XML")
        self.assertEqual(properties['VERSION'], "0.2")
        self.assertEqual(properties['CREATOR'], "Microsoft Research DTAS")
        properties = self.reader_file.get_static_properties()
        self.assertEqual(properties['FORMAT'], "MSR DTAS XML")
        self.assertEqual(properties['VERSION'], "0.2")
        self.assertEqual(properties['CREATOR'], "Microsoft Research DTAS")

    def test_get_variables(self):
        self.assertListEqual(sorted(list(self.reader_string.get_variables())), ['a', 'b', 'c', 'd', 'e'])
        self.assertListEqual(sorted(list(self.reader_file.get_variables())), ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(self.reader_string.get_variables()['a']['TYPE'], 'discrete')
        self.assertEqual(self.reader_string.get_variables()['a']['XPOS'], '13495')
        self.assertEqual(self.reader_string.get_variables()['a']['YPOS'], '10465')
        self.assertEqual(self.reader_string.get_variables()['a']['DESCRIPTION'], '(a) Metastatic Cancer')
        self.assertListEqual(self.reader_string.get_variables()['a']['STATES'], ['Present', 'Absent'])
        self.assertEqual(self.reader_file.get_variables()['a']['TYPE'], 'discrete')
        self.assertEqual(self.reader_file.get_variables()['a']['XPOS'], '13495')
        self.assertEqual(self.reader_file.get_variables()['a']['YPOS'], '10465')
        self.assertEqual(self.reader_file.get_variables()['a']['DESCRIPTION'], '(a) Metastatic Cancer')
        self.assertListEqual(self.reader_file.get_variables()['a']['STATES'], ['Present', 'Absent'])

    def test_get_edges(self):
        self.assertListEqual(self.reader_string.get_edges(),
                             [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'), ('c', 'e')])
        self.assertListEqual(self.reader_file.get_edges(), [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'), ('c', 'e')])

    def test_get_distribution(self):
        distribution = self.reader_string.get_distributions()
        self.assertEqual(distribution['a']['TYPE'], 'discrete')
        self.assertListEqual(distribution['b']['CONDSET'], ['a'])
        np_test.assert_array_equal(distribution['a']['DPIS'], np.array([[0.2, 0.8]]))
        np_test.assert_array_equal(distribution['e']['DPIS'], np.array([[0.8, 0.2], [0.6, 0.4]]))
        np_test.assert_array_equal(distribution['e']['CARDINALITY'], np.array([2]))
        np_test.assert_array_equal(distribution['d']['DPIS'],
                                   np.array([[0.8, 0.2], [0.9, 0.1], [0.7, 0.3], [0.05, 0.95]]))
        np_test.assert_array_equal(distribution['b']['DPIS'], np.array([[0.8, 0.2], [0.2, 0.8]]))
        np_test.assert_array_equal(distribution['d']['CARDINALITY'], np.array([2, 2]))
        np_test.assert_array_equal(distribution['c']['DPIS'], np.array([[0.2, 0.8], [0.05, 0.95]]))
        np_test.assert_array_equal(distribution['c']['CARDINALITY'], np.array([2]))
        distribution = self.reader_file.get_distributions()
        self.assertEqual(distribution['a']['TYPE'], 'discrete')
        self.assertListEqual(distribution['b']['CONDSET'], ['a'])
        np_test.assert_array_equal(distribution['a']['DPIS'], np.array([[0.2, 0.8]]))
        np_test.assert_array_equal(distribution['e']['DPIS'], np.array([[0.8, 0.2], [0.6, 0.4]]))
        np_test.assert_array_equal(distribution['e']['CARDINALITY'], np.array([2]))
        np_test.assert_array_equal(distribution['d']['DPIS'],
                                   np.array([[0.8, 0.2], [0.9, 0.1], [0.7, 0.3], [0.05, 0.95]]))
        np_test.assert_array_equal(distribution['d']['CARDINALITY'], np.array([2, 2]))
        np_test.assert_array_equal(distribution['b']['DPIS'], np.array([[0.8, 0.2], [0.2, 0.8]]))
        np_test.assert_array_equal(distribution['c']['DPIS'], np.array([[0.2, 0.8], [0.05, 0.95]]))
        np_test.assert_array_equal(distribution['c']['CARDINALITY'], np.array([2]))
