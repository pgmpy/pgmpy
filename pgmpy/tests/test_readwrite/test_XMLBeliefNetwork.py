import io
import sys
import unittest
import xml.etree.ElementTree as etree

import networkx as nx
import numpy as np
import numpy.testing as np_test

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
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
                             <VAR NAME="f" TYPE="discrete" XPOS="13440" YPOS="10489">
                                <DESCRIPTION>(f) Asthma</DESCRIPTION>
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
                             <DIST TYPE="discrete">
                                <PRIVATE NAME="f"/>
                                <DPIS>
                                   <DPI> 0.3 0.7</DPI>
                                </DPIS>
                             </DIST>
                          </DISTRIBUTIONS>
                       </BNMODEL>
                    </ANALYSISNOTEBOOK>"""

        self.reader_string = XMLBeliefNetwork.XBNReader(string=string)
        self.reader_file = XMLBeliefNetwork.XBNReader(path=io.StringIO(string))

    def test_init_exception(self):
        self.assertRaises(ValueError, XMLBeliefNetwork.XBNReader)

    def test_get_analysis_notebook(self):
        self.assertEqual(
            self.reader_string.get_analysisnotebook_values()["NAME"],
            "Notebook.Cancer Example From Neapolitan",
        )
        self.assertEqual(
            self.reader_string.get_analysisnotebook_values()["ROOT"], "Cancer"
        )
        self.assertEqual(
            self.reader_file.get_analysisnotebook_values()["NAME"],
            "Notebook.Cancer Example From Neapolitan",
        )
        self.assertEqual(
            self.reader_file.get_analysisnotebook_values()["ROOT"], "Cancer"
        )

    def test_get_bnmodel_name(self):
        self.assertEqual(self.reader_string.get_bnmodel_name(), "Cancer")
        self.assertEqual(self.reader_file.get_bnmodel_name(), "Cancer")

    def test_get_static_properties(self):
        properties = self.reader_string.get_static_properties()
        self.assertEqual(properties["FORMAT"], "MSR DTAS XML")
        self.assertEqual(properties["VERSION"], "0.2")
        self.assertEqual(properties["CREATOR"], "Microsoft Research DTAS")
        properties = self.reader_file.get_static_properties()
        self.assertEqual(properties["FORMAT"], "MSR DTAS XML")
        self.assertEqual(properties["VERSION"], "0.2")
        self.assertEqual(properties["CREATOR"], "Microsoft Research DTAS")

    def test_get_variables(self):
        self.assertListEqual(
            sorted(list(self.reader_string.get_variables())),
            ["a", "b", "c", "d", "e", "f"],
        )
        self.assertListEqual(
            sorted(list(self.reader_file.get_variables())),
            ["a", "b", "c", "d", "e", "f"],
        )
        self.assertEqual(self.reader_string.get_variables()["a"]["TYPE"], "discrete")
        self.assertEqual(self.reader_string.get_variables()["a"]["XPOS"], "13495")
        self.assertEqual(self.reader_string.get_variables()["a"]["YPOS"], "10465")
        self.assertEqual(
            self.reader_string.get_variables()["a"]["DESCRIPTION"],
            "(a) Metastatic Cancer",
        )
        self.assertListEqual(
            self.reader_string.get_variables()["a"]["STATES"], ["Present", "Absent"]
        )
        self.assertEqual(self.reader_file.get_variables()["a"]["TYPE"], "discrete")
        self.assertEqual(self.reader_file.get_variables()["a"]["XPOS"], "13495")
        self.assertEqual(self.reader_file.get_variables()["a"]["YPOS"], "10465")
        self.assertEqual(
            self.reader_file.get_variables()["a"]["DESCRIPTION"],
            "(a) Metastatic Cancer",
        )
        self.assertListEqual(
            self.reader_file.get_variables()["a"]["STATES"], ["Present", "Absent"]
        )

    def test_get_edges(self):
        self.assertListEqual(
            self.reader_string.get_edges(),
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("c", "e")],
        )
        self.assertListEqual(
            self.reader_file.get_edges(),
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("c", "e")],
        )

    def test_get_distribution(self):
        distribution = self.reader_string.get_distributions()
        self.assertEqual(distribution["a"]["TYPE"], "discrete")
        self.assertListEqual(distribution["b"]["CONDSET"], ["a"])
        np_test.assert_array_equal(distribution["a"]["DPIS"], np.array([[0.2], [0.8]]))
        np_test.assert_array_equal(distribution["f"]["DPIS"], np.array([[0.3], [0.7]]))
        np_test.assert_array_equal(
            distribution["e"]["DPIS"], np.array([[0.8, 0.6], [0.2, 0.4]])
        )
        np_test.assert_array_equal(distribution["e"]["CARDINALITY"], np.array([2]))
        np_test.assert_array_equal(
            distribution["d"]["DPIS"],
            np.array([[0.8, 0.9, 0.7, 0.05], [0.2, 0.1, 0.3, 0.95]]),
        )
        np_test.assert_array_equal(
            distribution["b"]["DPIS"], np.array([[0.8, 0.2], [0.2, 0.8]])
        )
        np_test.assert_array_equal(distribution["d"]["CARDINALITY"], np.array([2, 2]))
        np_test.assert_array_equal(
            distribution["c"]["DPIS"], np.array([[0.2, 0.05], [0.8, 0.95]])
        )
        np_test.assert_array_equal(distribution["c"]["CARDINALITY"], np.array([2]))
        distribution = self.reader_file.get_distributions()
        self.assertEqual(distribution["a"]["TYPE"], "discrete")
        self.assertListEqual(distribution["b"]["CONDSET"], ["a"])
        np_test.assert_array_equal(distribution["a"]["DPIS"], np.array([[0.2], [0.8]]))
        np_test.assert_array_equal(distribution["f"]["DPIS"], np.array([[0.3], [0.7]]))
        np_test.assert_array_equal(
            distribution["e"]["DPIS"], np.array([[0.8, 0.6], [0.2, 0.4]])
        )
        np_test.assert_array_equal(distribution["e"]["CARDINALITY"], np.array([2]))
        np_test.assert_array_equal(
            distribution["d"]["DPIS"],
            np.array([[0.8, 0.9, 0.7, 0.05], [0.2, 0.1, 0.3, 0.95]]),
        )
        np_test.assert_array_equal(distribution["d"]["CARDINALITY"], np.array([2, 2]))
        np_test.assert_array_equal(
            distribution["b"]["DPIS"], np.array([[0.8, 0.2], [0.2, 0.8]])
        )
        np_test.assert_array_equal(
            distribution["c"]["DPIS"], np.array([[0.2, 0.05], [0.8, 0.95]])
        )
        np_test.assert_array_equal(distribution["c"]["CARDINALITY"], np.array([2]))

    def test_get_model(self):
        model = self.reader_string.get_model()
        node_expected = {
            "c": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(c) Brain Tumor",
                "YPOS": "11935",
                "XPOS": "15250",
                "TYPE": "discrete",
            },
            "a": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(a) Metastatic Cancer",
                "YPOS": "10465",
                "XPOS": "13495",
                "TYPE": "discrete",
            },
            "b": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(b) Serum Calcium Increase",
                "YPOS": "11965",
                "XPOS": "11290",
                "TYPE": "discrete",
            },
            "e": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(e) Papilledema",
                "YPOS": "13240",
                "XPOS": "17305",
                "TYPE": "discrete",
            },
            "f": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(f) Asthma",
                "YPOS": "10489",
                "XPOS": "13440",
                "TYPE": "discrete",
            },
            "d": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(d) Coma",
                "YPOS": "12985",
                "XPOS": "13960",
                "TYPE": "discrete",
            },
        }
        cpds_expected = {
            "b": np.array([[0.8, 0.2], [0.2, 0.8]]),
            "e": np.array([[0.8, 0.6], [0.2, 0.4]]),
            "f": np.array([[0.3], [0.7]]),
            "c": np.array([[0.2, 0.05], [0.8, 0.95]]),
            "a": np.array([[0.2], [0.8]]),
            "d": np.array([[0.8, 0.9, 0.7, 0.05], [0.2, 0.1, 0.3, 0.95]]),
        }
        for cpd in model.get_cpds():
            np_test.assert_array_equal(cpd.get_values(), cpds_expected[cpd.variable])
        self.assertListEqual(
            sorted(model.edges()),
            sorted([("b", "d"), ("a", "b"), ("a", "c"), ("c", "d"), ("c", "e")]),
        )
        self.assertDictEqual(dict(model.nodes), node_expected)


class TestXBNWriter(unittest.TestCase):
    def setUp(self):
        nodes = {
            "c": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(c) Brain Tumor",
                "YPOS": "11935",
                "XPOS": "15250",
                "TYPE": "discrete",
            },
            "a": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(a) Metastatic Cancer",
                "YPOS": "10465",
                "XPOS": "13495",
                "TYPE": "discrete",
            },
            "b": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(b) Serum Calcium Increase",
                "YPOS": "11965",
                "XPOS": "11290",
                "TYPE": "discrete",
            },
            "e": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(e) Papilledema",
                "YPOS": "13240",
                "XPOS": "17305",
                "TYPE": "discrete",
            },
            "f": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(f) Asthma",
                "YPOS": "10489",
                "XPOS": "13440",
                "TYPE": "discrete",
            },
            "d": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(d) Coma",
                "YPOS": "12985",
                "XPOS": "13960",
                "TYPE": "discrete",
            },
        }
        model = BayesianNetwork()
        model.add_nodes_from(["a", "b", "c", "d", "e", "f"])
        model.add_edges_from(
            [("b", "d"), ("a", "b"), ("a", "c"), ("c", "d"), ("c", "e")]
        )
        cpd_distribution = {
            "a": {"TYPE": "discrete", "DPIS": np.array([[0.2], [0.8]])},
            "e": {
                "TYPE": "discrete",
                "DPIS": np.array([[0.8, 0.6], [0.2, 0.4]]),
                "CONDSET": ["c"],
                "CARDINALITY": [2],
            },
            "f": {"TYPE": "discrete", "DPIS": np.array([[0.3], [0.7]])},
            "b": {
                "TYPE": "discrete",
                "DPIS": np.array([[0.8, 0.2], [0.2, 0.8]]),
                "CONDSET": ["a"],
                "CARDINALITY": [2],
            },
            "c": {
                "TYPE": "discrete",
                "DPIS": np.array([[0.2, 0.05], [0.8, 0.95]]),
                "CONDSET": ["a"],
                "CARDINALITY": [2],
            },
            "d": {
                "TYPE": "discrete",
                "DPIS": np.array([[0.8, 0.9, 0.7, 0.05], [0.2, 0.1, 0.3, 0.95]]),
                "CONDSET": ["b", "c"],
                "CARDINALITY": [2, 2],
            },
        }

        tabular_cpds = []
        for var, values in cpd_distribution.items():
            evidence = values["CONDSET"] if "CONDSET" in values else []
            cpd = values["DPIS"]
            evidence_card = values["CARDINALITY"] if "CARDINALITY" in values else []
            states = nodes[var]["STATES"]
            cpd = TabularCPD(
                var, len(states), cpd, evidence=evidence, evidence_card=evidence_card
            )
            tabular_cpds.append(cpd)
        model.add_cpds(*tabular_cpds)

        if nx.__version__.startswith("1"):
            for var, properties in nodes.items():
                model.nodes[var] = properties
        else:
            for var, properties in nodes.items():
                model._node[var] = properties

        self.maxDiff = None
        self.writer = XMLBeliefNetwork.XBNWriter(model=model)

    @unittest.skipIf(sys.version_info[1] >= 8, "xml ordering different in python 3.8")
    def test_file(self):
        self.expected_xml = etree.XML(
            """<ANALYSISNOTEBOOK>
  <BNMODEL>
    <VARIABLES>
      <VAR NAME="a" TYPE="discrete" XPOS="13495" YPOS="10465">
        <DESCRIPTION DESCRIPTION="(a) Metastatic Cancer"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="b" TYPE="discrete" XPOS="11290" YPOS="11965">
        <DESCRIPTION DESCRIPTION="(b) Serum Calcium Increase"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="c" TYPE="discrete" XPOS="15250" YPOS="11935">
        <DESCRIPTION DESCRIPTION="(c) Brain Tumor"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="d" TYPE="discrete" XPOS="13960" YPOS="12985">
        <DESCRIPTION DESCRIPTION="(d) Coma"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="e" TYPE="discrete" XPOS="17305" YPOS="13240">
        <DESCRIPTION DESCRIPTION="(e) Papilledema"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="f" TYPE="discrete" XPOS="13440" YPOS="10489">
        <DESCRIPTION DESCRIPTION="(f) Asthma"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
    </VARIABLES>
    <STRUCTURE>
      <ARC CHILD="b" PARENT="a"/>
      <ARC CHILD="c" PARENT="a"/>
      <ARC CHILD="d" PARENT="b"/>
      <ARC CHILD="d" PARENT="c"/>
      <ARC CHILD="e" PARENT="c"/>
    </STRUCTURE>
    <DISTRIBUTIONS>
      <DIST TYPE="discrete">
        <PRIVATE NAME="a"/>
        <DPIS>
          <DPI> 0.2 0.8 </DPI>
        </DPIS>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="b"/>
        <DPIS>
          <DPI INDEXES=" 0 "> 0.8 0.2 </DPI>
          <DPI INDEXES=" 1 "> 0.2 0.8 </DPI>
        </DPIS>
        <CONDSET>
          <CONDELEM NAME="a"/>
        </CONDSET>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="c"/>
        <DPIS>
          <DPI INDEXES=" 0 "> 0.2 0.8 </DPI>
          <DPI INDEXES=" 1 "> 0.05 0.95 </DPI>
        </DPIS>
        <CONDSET>
          <CONDELEM NAME="a"/>
        </CONDSET>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="d"/>
        <DPIS>
          <DPI INDEXES=" 0 0 "> 0.8 0.2 </DPI>
          <DPI INDEXES=" 0 1 "> 0.9 0.1 </DPI>
          <DPI INDEXES=" 1 0 "> 0.7 0.3 </DPI>
          <DPI INDEXES=" 1 1 "> 0.05 0.95 </DPI>
        </DPIS>
        <CONDSET>
          <CONDELEM NAME="b"/>
          <CONDELEM NAME="c"/>
        </CONDSET>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="e"/>
        <DPIS>
          <DPI INDEXES=" 0 "> 0.8 0.2 </DPI>
          <DPI INDEXES=" 1 "> 0.6 0.4 </DPI>
        </DPIS>
        <CONDSET>
          <CONDELEM NAME="c"/>
        </CONDSET>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="f"/>
        <DPIS>
          <DPI> 0.3 0.7 </DPI>
        </DPIS>
      </DIST>
    </DISTRIBUTIONS>
  </BNMODEL>
</ANALYSISNOTEBOOK>"""
        )
        output = str(self.writer.__str__())
        expected = str(etree.tostring(self.expected_xml))
        self.assertEqual(
            str(self.writer.__str__()[:-1]), str(etree.tostring(self.expected_xml))
        )


class TestXBNReaderTorch(unittest.TestCase):
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
                             <VAR NAME="f" TYPE="discrete" XPOS="13440" YPOS="10489">
                                <DESCRIPTION>(f) Asthma</DESCRIPTION>
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
                             <DIST TYPE="discrete">
                                <PRIVATE NAME="f"/>
                                <DPIS>
                                   <DPI> 0.3 0.7</DPI>
                                </DPIS>
                             </DIST>
                          </DISTRIBUTIONS>
                       </BNMODEL>
                    </ANALYSISNOTEBOOK>"""

        self.reader_string = XMLBeliefNetwork.XBNReader(string=string)
        self.reader_file = XMLBeliefNetwork.XBNReader(path=io.StringIO(string))

    def test_init_exception(self):
        self.assertRaises(ValueError, XMLBeliefNetwork.XBNReader)

    def test_get_analysis_notebook(self):
        self.assertEqual(
            self.reader_string.get_analysisnotebook_values()["NAME"],
            "Notebook.Cancer Example From Neapolitan",
        )
        self.assertEqual(
            self.reader_string.get_analysisnotebook_values()["ROOT"], "Cancer"
        )
        self.assertEqual(
            self.reader_file.get_analysisnotebook_values()["NAME"],
            "Notebook.Cancer Example From Neapolitan",
        )
        self.assertEqual(
            self.reader_file.get_analysisnotebook_values()["ROOT"], "Cancer"
        )

    def test_get_bnmodel_name(self):
        self.assertEqual(self.reader_string.get_bnmodel_name(), "Cancer")
        self.assertEqual(self.reader_file.get_bnmodel_name(), "Cancer")

    def test_get_static_properties(self):
        properties = self.reader_string.get_static_properties()
        self.assertEqual(properties["FORMAT"], "MSR DTAS XML")
        self.assertEqual(properties["VERSION"], "0.2")
        self.assertEqual(properties["CREATOR"], "Microsoft Research DTAS")
        properties = self.reader_file.get_static_properties()
        self.assertEqual(properties["FORMAT"], "MSR DTAS XML")
        self.assertEqual(properties["VERSION"], "0.2")
        self.assertEqual(properties["CREATOR"], "Microsoft Research DTAS")

    def test_get_variables(self):
        self.assertListEqual(
            sorted(list(self.reader_string.get_variables())),
            ["a", "b", "c", "d", "e", "f"],
        )
        self.assertListEqual(
            sorted(list(self.reader_file.get_variables())),
            ["a", "b", "c", "d", "e", "f"],
        )
        self.assertEqual(self.reader_string.get_variables()["a"]["TYPE"], "discrete")
        self.assertEqual(self.reader_string.get_variables()["a"]["XPOS"], "13495")
        self.assertEqual(self.reader_string.get_variables()["a"]["YPOS"], "10465")
        self.assertEqual(
            self.reader_string.get_variables()["a"]["DESCRIPTION"],
            "(a) Metastatic Cancer",
        )
        self.assertListEqual(
            self.reader_string.get_variables()["a"]["STATES"], ["Present", "Absent"]
        )
        self.assertEqual(self.reader_file.get_variables()["a"]["TYPE"], "discrete")
        self.assertEqual(self.reader_file.get_variables()["a"]["XPOS"], "13495")
        self.assertEqual(self.reader_file.get_variables()["a"]["YPOS"], "10465")
        self.assertEqual(
            self.reader_file.get_variables()["a"]["DESCRIPTION"],
            "(a) Metastatic Cancer",
        )
        self.assertListEqual(
            self.reader_file.get_variables()["a"]["STATES"], ["Present", "Absent"]
        )

    def test_get_edges(self):
        self.assertListEqual(
            self.reader_string.get_edges(),
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("c", "e")],
        )
        self.assertListEqual(
            self.reader_file.get_edges(),
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("c", "e")],
        )

    def test_get_distribution(self):
        distribution = self.reader_string.get_distributions()
        self.assertEqual(distribution["a"]["TYPE"], "discrete")
        self.assertListEqual(distribution["b"]["CONDSET"], ["a"])
        np_test.assert_array_equal(distribution["a"]["DPIS"], np.array([[0.2], [0.8]]))
        np_test.assert_array_equal(distribution["f"]["DPIS"], np.array([[0.3], [0.7]]))
        np_test.assert_array_equal(
            distribution["e"]["DPIS"], np.array([[0.8, 0.6], [0.2, 0.4]])
        )
        np_test.assert_array_equal(distribution["e"]["CARDINALITY"], np.array([2]))
        np_test.assert_array_equal(
            distribution["d"]["DPIS"],
            np.array([[0.8, 0.9, 0.7, 0.05], [0.2, 0.1, 0.3, 0.95]]),
        )
        np_test.assert_array_equal(
            distribution["b"]["DPIS"], np.array([[0.8, 0.2], [0.2, 0.8]])
        )
        np_test.assert_array_equal(distribution["d"]["CARDINALITY"], np.array([2, 2]))
        np_test.assert_array_equal(
            distribution["c"]["DPIS"], np.array([[0.2, 0.05], [0.8, 0.95]])
        )
        np_test.assert_array_equal(distribution["c"]["CARDINALITY"], np.array([2]))
        distribution = self.reader_file.get_distributions()
        self.assertEqual(distribution["a"]["TYPE"], "discrete")
        self.assertListEqual(distribution["b"]["CONDSET"], ["a"])
        np_test.assert_array_equal(distribution["a"]["DPIS"], np.array([[0.2], [0.8]]))
        np_test.assert_array_equal(distribution["f"]["DPIS"], np.array([[0.3], [0.7]]))
        np_test.assert_array_equal(
            distribution["e"]["DPIS"], np.array([[0.8, 0.6], [0.2, 0.4]])
        )
        np_test.assert_array_equal(distribution["e"]["CARDINALITY"], np.array([2]))
        np_test.assert_array_equal(
            distribution["d"]["DPIS"],
            np.array([[0.8, 0.9, 0.7, 0.05], [0.2, 0.1, 0.3, 0.95]]),
        )
        np_test.assert_array_equal(distribution["d"]["CARDINALITY"], np.array([2, 2]))
        np_test.assert_array_equal(
            distribution["b"]["DPIS"], np.array([[0.8, 0.2], [0.2, 0.8]])
        )
        np_test.assert_array_equal(
            distribution["c"]["DPIS"], np.array([[0.2, 0.05], [0.8, 0.95]])
        )
        np_test.assert_array_equal(distribution["c"]["CARDINALITY"], np.array([2]))

    def test_get_model(self):
        model = self.reader_string.get_model()
        node_expected = {
            "c": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(c) Brain Tumor",
                "YPOS": "11935",
                "XPOS": "15250",
                "TYPE": "discrete",
            },
            "a": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(a) Metastatic Cancer",
                "YPOS": "10465",
                "XPOS": "13495",
                "TYPE": "discrete",
            },
            "b": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(b) Serum Calcium Increase",
                "YPOS": "11965",
                "XPOS": "11290",
                "TYPE": "discrete",
            },
            "e": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(e) Papilledema",
                "YPOS": "13240",
                "XPOS": "17305",
                "TYPE": "discrete",
            },
            "f": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(f) Asthma",
                "YPOS": "10489",
                "XPOS": "13440",
                "TYPE": "discrete",
            },
            "d": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(d) Coma",
                "YPOS": "12985",
                "XPOS": "13960",
                "TYPE": "discrete",
            },
        }
        cpds_expected = {
            "b": np.array([[0.8, 0.2], [0.2, 0.8]]),
            "e": np.array([[0.8, 0.6], [0.2, 0.4]]),
            "f": np.array([[0.3], [0.7]]),
            "c": np.array([[0.2, 0.05], [0.8, 0.95]]),
            "a": np.array([[0.2], [0.8]]),
            "d": np.array([[0.8, 0.9, 0.7, 0.05], [0.2, 0.1, 0.3, 0.95]]),
        }
        for cpd in model.get_cpds():
            np_test.assert_array_equal(cpd.get_values(), cpds_expected[cpd.variable])
        self.assertListEqual(
            sorted(model.edges()),
            sorted([("b", "d"), ("a", "b"), ("a", "c"), ("c", "d"), ("c", "e")]),
        )
        self.assertDictEqual(dict(model.nodes), node_expected)


class TestXBNWriterTorch(unittest.TestCase):
    def setUp(self):
        nodes = {
            "c": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(c) Brain Tumor",
                "YPOS": "11935",
                "XPOS": "15250",
                "TYPE": "discrete",
            },
            "a": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(a) Metastatic Cancer",
                "YPOS": "10465",
                "XPOS": "13495",
                "TYPE": "discrete",
            },
            "b": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(b) Serum Calcium Increase",
                "YPOS": "11965",
                "XPOS": "11290",
                "TYPE": "discrete",
            },
            "e": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(e) Papilledema",
                "YPOS": "13240",
                "XPOS": "17305",
                "TYPE": "discrete",
            },
            "f": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(f) Asthma",
                "YPOS": "10489",
                "XPOS": "13440",
                "TYPE": "discrete",
            },
            "d": {
                "STATES": ["Present", "Absent"],
                "DESCRIPTION": "(d) Coma",
                "YPOS": "12985",
                "XPOS": "13960",
                "TYPE": "discrete",
            },
        }
        model = BayesianNetwork()
        model.add_nodes_from(["a", "b", "c", "d", "e", "f"])
        model.add_edges_from(
            [("b", "d"), ("a", "b"), ("a", "c"), ("c", "d"), ("c", "e")]
        )
        cpd_distribution = {
            "a": {"TYPE": "discrete", "DPIS": np.array([[0.2], [0.8]])},
            "e": {
                "TYPE": "discrete",
                "DPIS": np.array([[0.8, 0.6], [0.2, 0.4]]),
                "CONDSET": ["c"],
                "CARDINALITY": [2],
            },
            "f": {"TYPE": "discrete", "DPIS": np.array([[0.3], [0.7]])},
            "b": {
                "TYPE": "discrete",
                "DPIS": np.array([[0.8, 0.2], [0.2, 0.8]]),
                "CONDSET": ["a"],
                "CARDINALITY": [2],
            },
            "c": {
                "TYPE": "discrete",
                "DPIS": np.array([[0.2, 0.05], [0.8, 0.95]]),
                "CONDSET": ["a"],
                "CARDINALITY": [2],
            },
            "d": {
                "TYPE": "discrete",
                "DPIS": np.array([[0.8, 0.9, 0.7, 0.05], [0.2, 0.1, 0.3, 0.95]]),
                "CONDSET": ["b", "c"],
                "CARDINALITY": [2, 2],
            },
        }

        tabular_cpds = []
        for var, values in cpd_distribution.items():
            evidence = values["CONDSET"] if "CONDSET" in values else []
            cpd = values["DPIS"]
            evidence_card = values["CARDINALITY"] if "CARDINALITY" in values else []
            states = nodes[var]["STATES"]
            cpd = TabularCPD(
                var, len(states), cpd, evidence=evidence, evidence_card=evidence_card
            )
            tabular_cpds.append(cpd)
        model.add_cpds(*tabular_cpds)

        if nx.__version__.startswith("1"):
            for var, properties in nodes.items():
                model.nodes[var] = properties
        else:
            for var, properties in nodes.items():
                model._node[var] = properties

        self.maxDiff = None
        self.writer = XMLBeliefNetwork.XBNWriter(model=model)

    @unittest.skipIf(sys.version_info[1] >= 8, "xml ordering different in python 3.8")
    def test_file(self):
        self.expected_xml = etree.XML(
            """<ANALYSISNOTEBOOK>
  <BNMODEL>
    <VARIABLES>
      <VAR NAME="a" TYPE="discrete" XPOS="13495" YPOS="10465">
        <DESCRIPTION DESCRIPTION="(a) Metastatic Cancer"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="b" TYPE="discrete" XPOS="11290" YPOS="11965">
        <DESCRIPTION DESCRIPTION="(b) Serum Calcium Increase"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="c" TYPE="discrete" XPOS="15250" YPOS="11935">
        <DESCRIPTION DESCRIPTION="(c) Brain Tumor"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="d" TYPE="discrete" XPOS="13960" YPOS="12985">
        <DESCRIPTION DESCRIPTION="(d) Coma"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="e" TYPE="discrete" XPOS="17305" YPOS="13240">
        <DESCRIPTION DESCRIPTION="(e) Papilledema"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
      <VAR NAME="f" TYPE="discrete" XPOS="13440" YPOS="10489">
        <DESCRIPTION DESCRIPTION="(f) Asthma"/>
        <STATENAME>Present</STATENAME>
        <STATENAME>Absent</STATENAME>
      </VAR>
    </VARIABLES>
    <STRUCTURE>
      <ARC CHILD="b" PARENT="a"/>
      <ARC CHILD="c" PARENT="a"/>
      <ARC CHILD="d" PARENT="b"/>
      <ARC CHILD="d" PARENT="c"/>
      <ARC CHILD="e" PARENT="c"/>
    </STRUCTURE>
    <DISTRIBUTIONS>
      <DIST TYPE="discrete">
        <PRIVATE NAME="a"/>
        <DPIS>
          <DPI> 0.2 0.8 </DPI>
        </DPIS>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="b"/>
        <DPIS>
          <DPI INDEXES=" 0 "> 0.8 0.2 </DPI>
          <DPI INDEXES=" 1 "> 0.2 0.8 </DPI>
        </DPIS>
        <CONDSET>
          <CONDELEM NAME="a"/>
        </CONDSET>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="c"/>
        <DPIS>
          <DPI INDEXES=" 0 "> 0.2 0.8 </DPI>
          <DPI INDEXES=" 1 "> 0.05 0.95 </DPI>
        </DPIS>
        <CONDSET>
          <CONDELEM NAME="a"/>
        </CONDSET>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="d"/>
        <DPIS>
          <DPI INDEXES=" 0 0 "> 0.8 0.2 </DPI>
          <DPI INDEXES=" 0 1 "> 0.9 0.1 </DPI>
          <DPI INDEXES=" 1 0 "> 0.7 0.3 </DPI>
          <DPI INDEXES=" 1 1 "> 0.05 0.95 </DPI>
        </DPIS>
        <CONDSET>
          <CONDELEM NAME="b"/>
          <CONDELEM NAME="c"/>
        </CONDSET>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="e"/>
        <DPIS>
          <DPI INDEXES=" 0 "> 0.8 0.2 </DPI>
          <DPI INDEXES=" 1 "> 0.6 0.4 </DPI>
        </DPIS>
        <CONDSET>
          <CONDELEM NAME="c"/>
        </CONDSET>
      </DIST>
      <DIST TYPE="discrete">
        <PRIVATE NAME="f"/>
        <DPIS>
          <DPI> 0.3 0.7 </DPI>
        </DPIS>
      </DIST>
    </DISTRIBUTIONS>
  </BNMODEL>
</ANALYSISNOTEBOOK>"""
        )
        output = str(self.writer.__str__())
        expected = str(etree.tostring(self.expected_xml))
        self.assertEqual(
            str(self.writer.__str__()[:-1]), str(etree.tostring(self.expected_xml))
        )
