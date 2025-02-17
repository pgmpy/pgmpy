import os
import unittest

import networkx as nx
import numpy as np
import numpy.testing as np_test

from pgmpy import config
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFReader, BIFWriter


class TestBIFReader(unittest.TestCase):
    def setUp(self):
        self.reader = BIFReader(
            string="""
                // Bayesian Network in the Interchange Format
                // Produced by BayesianNetworks package in JavaBayes
                // Output created Sun Nov 02 17:49:49 GMT+00:00 1997
                // Bayesian network
                network "Dog-Problem" { //5 variables and 5 probability distributions
                        property "credal-set constant-density-bounded 1.1" ;
                }
                variable  "light-on" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (218, 195)" ;
                }
                variable  "bowel-problem" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (335, 99)" ;
                }
                variable  "dog-out" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (300, 195)" ;
                }
                variable  "hear-bark" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (296, 268)" ;
                }
                variable  "family-out" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (257, 99)" ;
                }
                probability (  "light-on"  "family-out" ) { //2 variable(s) and 4 values
                        (true) 0.6 0.4 ;
                        (false) 0.05 0.95 ;
                }
                probability (  "bowel-problem" ) { //1 variable(s) and 2 values
                        table 0.01 0.99 ;
                }
                probability (  "dog-out"  "bowel-problem"  "family-out" ) { //3 variable(s) and 8 values
                        table 0.99 0.97 0.9 0.3 0.01 0.03 0.1 0.7 ;
                }
                probability (  "hear-bark"  "dog-out" ) { //2 variable(s) and 4 values
                        table 0.7 0.01 0.3 0.99 ;
                }
                probability (  "family-out" ) { //1 variable(s) and 2 values
                        table 0.15 0.85 ;
                }
                """,
            include_properties=True,
        )

        self.water_model = BIFReader(
            "pgmpy/tests/test_readwrite/testdata/water.bif", include_properties=True
        )

    def test_network_name(self):
        name_expected = "Dog-Problem"
        self.assertEqual(self.reader.network_name, name_expected)

    def test_get_variables(self):
        var_expected = [
            "light-on",
            "bowel-problem",
            "dog-out",
            "hear-bark",
            "family-out",
        ]
        self.assertListEqual(self.reader.get_variables(), var_expected)

    def test_states(self):
        states_expected = {
            "bowel-problem": ["true", "false"],
            "dog-out": ["true", "false"],
            "family-out": ["true", "false"],
            "hear-bark": ["true", "false"],
            "light-on": ["true", "false"],
        }
        states = self.reader.get_states()
        for variable in states_expected:
            self.assertListEqual(states_expected[variable], states[variable])

    def test_get_property(self):
        property_expected = {
            "bowel-problem": ["position = (335, 99)"],
            "dog-out": ["position = (300, 195)"],
            "family-out": ["position = (257, 99)"],
            "hear-bark": ["position = (296, 268)"],
            "light-on": ["position = (218, 195)"],
        }
        prop = self.reader.get_property()
        for variable in property_expected:
            self.assertListEqual(property_expected[variable], prop[variable])

    def test_get_values(self):
        cpd_expected = {
            "bowel-problem": np.array([[0.01], [0.99]]),
            "dog-out": np.array([[0.99, 0.97, 0.9, 0.3], [0.01, 0.03, 0.1, 0.7]]),
            "family-out": np.array([[0.15], [0.85]]),
            "hear-bark": np.array([[0.7, 0.01], [0.3, 0.99]]),
            "light-on": np.array([[0.6, 0.05], [0.4, 0.95]]),
        }
        cpd = self.reader.variable_cpds
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable], cpd[variable])

    def test_get_values_reordered(self):
        cancer_values1 = BIFReader(
            string="""
                network unknown {
                }
                variable Pollution {
                  type discrete [ 2 ] { low, high };
                }
                variable Smoker {
                  type discrete [ 2 ] { True, False };
                }
                variable Cancer {
                  type discrete [ 2 ] { True, False };
                }
                probability ( Cancer | Pollution, Smoker ) {
                  (low, True) 0.03, 0.97;
                  (low, False) 0.001, 0.999;
                  (high, True) 0.05, 0.95;
                  (high, False) 0.02, 0.98;
                }"""
        ).get_values()

        cancer_values2 = BIFReader(
            string="""
                network unknown {
                }
                variable Pollution {
                  type discrete [ 2 ] { low, high };
                }
                variable Smoker {
                  type discrete [ 2 ] { True, False };
                }
                variable Cancer {
                  type discrete [ 2 ] { True, False };
                }
                probability ( Cancer | Pollution, Smoker ) {
                  (low, True) 0.03, 0.97;
                  (high, True) 0.05, 0.95;
                  (low, False) 0.001, 0.999;
                  (high, False) 0.02, 0.98;
                }"""
        ).get_values()

        for var in cancer_values1:
            np_test.assert_array_equal(cancer_values1[var], cancer_values2[var])

    def test_get_parents(self):
        parents_expected = {
            "bowel-problem": [],
            "dog-out": ["bowel-problem", "family-out"],
            "family-out": [],
            "hear-bark": ["dog-out"],
            "light-on": ["family-out"],
        }
        parents = self.reader.get_parents()
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable], parents[variable])

    def test_get_edges(self):
        edges_expected = [
            ["family-out", "dog-out"],
            ["bowel-problem", "dog-out"],
            ["family-out", "light-on"],
            ["dog-out", "hear-bark"],
        ]
        self.assertListEqual(sorted(self.reader.variable_edges), sorted(edges_expected))

    def test_get_model(self):
        edges_expected = [
            ("family-out", "dog-out"),
            ("bowel-problem", "dog-out"),
            ("family-out", "light-on"),
            ("dog-out", "hear-bark"),
        ]
        nodes_expected = [
            "bowel-problem",
            "hear-bark",
            "light-on",
            "dog-out",
            "family-out",
        ]
        edge_expected = {
            "bowel-problem": {"dog-out": {"weight": None}},
            "dog-out": {"hear-bark": {"weight": None}},
            "family-out": {"dog-out": {"weight": None}, "light-on": {"weight": None}},
            "hear-bark": {},
            "light-on": {},
        }
        node_expected = {
            "bowel-problem": {"weight": None, "position": "(335, 99)"},
            "dog-out": {"weight": None, "position": "(300, 195)"},
            "family-out": {"weight": None, "position": "(257, 99)"},
            "hear-bark": {"weight": None, "position": "(296, 268)"},
            "light-on": {"weight": None, "position": "(218, 195)"},
        }
        cpds_expected = [
            TabularCPD(
                variable="bowel-problem",
                variable_card=2,
                values=np.array([[0.01], [0.99]]),
                state_names={"bowel-problem": ["true", "false"]},
            ),
            TabularCPD(
                variable="dog-out",
                variable_card=2,
                values=np.array([[0.99, 0.97, 0.9, 0.3], [0.01, 0.03, 0.1, 0.7]]),
                evidence=["bowel-problem", "family-out"],
                evidence_card=[2, 2],
                state_names={
                    "dog-out": ["true", "false"],
                    "bowel-problem": ["true", "false"],
                    "family-out": ["true", "false"],
                },
            ),
            TabularCPD(
                variable="family-out",
                variable_card=2,
                values=np.array([[0.15], [0.85]]),
                state_names={"family-out": ["true", "false"]},
            ),
            TabularCPD(
                variable="hear-bark",
                variable_card=2,
                values=np.array([[0.7, 0.01], [0.3, 0.99]]),
                evidence=["dog-out"],
                evidence_card=[2],
                state_names={
                    "hear-bark": ["true", "false"],
                    "dog-out": ["true", "false"],
                },
            ),
            TabularCPD(
                variable="light-on",
                variable_card=2,
                values=np.array([[0.6, 0.05], [0.4, 0.95]]),
                evidence=["family-out"],
                evidence_card=[2],
                state_names={
                    "light-on": ["true", "false"],
                    "family-out": ["true", "false"],
                },
            ),
        ]
        model = self.reader.get_model()
        model_cpds = model.get_cpds()
        for cpd_index in range(5):
            self.assertEqual(model_cpds[cpd_index], cpds_expected[cpd_index])

        self.assertDictEqual(dict(model.nodes), node_expected)
        self.assertDictEqual(dict(model.adj), edge_expected)

        self.assertListEqual(sorted(model.nodes()), sorted(nodes_expected))
        self.assertListEqual(sorted(model.edges()), sorted(edges_expected))

    def test_water_model(self):
        model = self.water_model.get_model()
        self.assertEqual(len(model.nodes()), 32)
        self.assertEqual(len(model.edges()), 66)
        self.assertEqual(len(model.get_cpds()), 32)

    def tearDown(self):
        del self.reader

    def test_default_attribut_equal_table(self):
        default_reader = BIFReader(
            string="""
                network "Dog-Problem" {
                }
                variable  "light-on" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (218, 195)" ;
                }
                variable  "bowel-problem" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (335, 99)" ;
                }
                variable  "dog-out" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (300, 195)" ;
                }
                variable  "hear-bark" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (296, 268)" ;
                }
                variable  "family-out" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (257, 99)" ;
                }
                probability (  "light-on"  "family-out" ) {
                        (true) 0.6 0.4 ;
                        (false) 0.05 0.95 ;
                }
                probability (  "bowel-problem" ) {
                        default 0.01 0.99 ;
                }
                probability (  "dog-out"  "bowel-problem"  "family-out" ) {
                        default 0.99 0.97 0.9 0.3 0.01 0.03 0.1 0.7 ;
                }
                probability (  "hear-bark"  "dog-out" ) {
                        default 0.7 0.01 0.3 0.99 ;
                }
                probability (  "family-out" ) {
                        default 0.15 0.85 ;
                }
                """,
            include_properties=True,
        )
        table_model = self.reader.get_model()
        default_model = default_reader.get_model()
        self.assertEqual(sorted(table_model.nodes()), sorted(default_model.nodes()))
        self.assertEqual(sorted(table_model.edges()), sorted(default_model.edges()))
        for var in table_model.nodes():
            self.assertEqual(table_model.get_cpds(var), default_model.get_cpds(var))


class TestBIFWriter(unittest.TestCase):
    def setUp(self):
        variables = [
            "kid",
            "bowel-problem",
            "dog-out",
            "family-out",
            "hear-bark",
            "light-on",
        ]

        edges = [
            ["family-out", "dog-out"],
            ["bowel-problem", "dog-out"],
            ["family-out", "light-on"],
            ["dog-out", "hear-bark"],
        ]

        cpds = {
            "kid": np.array([[0.3], [0.7]]),
            "bowel-problem": np.array([[0.01], [0.99]]),
            "dog-out": np.array([[0.99, 0.9, 0.97, 0.3], [0.01, 0.1, 0.03, 0.7]]),
            "family-out": np.array([[0.15], [0.85]]),
            "hear-bark": np.array([[0.7, 0.01], [0.3, 0.99]]),
            "light-on": np.array([[0.6, 0.95], [0.4, 0.05]]),
        }

        states = {
            "kid": ["true", "false"],
            "bowel-problem": ["true", "false"],
            "dog-out": ["true", "false"],
            "family-out": ["true", "false"],
            "hear-bark": ["true", "false"],
            "light-on": ["true", "false"],
        }

        parents = {
            "kid": [],
            "bowel-problem": [],
            "dog-out": ["bowel-problem", "family-out"],
            "family-out": [],
            "hear-bark": ["dog-out"],
            "light-on": ["family-out"],
        }

        properties = {
            "kid": ["position = (100, 165)"],
            "bowel-problem": ["position = (335, 99)"],
            "dog-out": ["position = (300, 195)"],
            "family-out": ["position = (257, 99)"],
            "hear-bark": ["position = (296, 268)"],
            "light-on": ["position = (218, 195)"],
        }

        self.model = BayesianNetwork()
        self.model.add_nodes_from(variables)
        self.model.add_edges_from(edges)

        tabular_cpds = []
        for var in sorted(cpds.keys()):
            values = cpds[var]
            cpd = TabularCPD(
                var,
                len(states[var]),
                values,
                evidence=parents[var],
                evidence_card=[
                    len(states[evidence_var]) for evidence_var in parents[var]
                ],
            )
            tabular_cpds.append(cpd)
        self.model.add_cpds(*tabular_cpds)

        for node, properties in properties.items():
            for prop in properties:
                prop_name, prop_value = map(lambda t: t.strip(), prop.split("="))
                self.model.nodes[node][prop_name] = prop_value

        self.writer = BIFWriter(model=self.model)

    def test_str(self):
        self.expected_string = """network unknown {
}
variable bowel-problem {
    type discrete [ 2 ] { 0, 1 };
    property position = (335, 99) ;
    property weight = None ;
}
variable dog-out {
    type discrete [ 2 ] { 0, 1 };
    property position = (300, 195) ;
    property weight = None ;
}
variable family-out {
    type discrete [ 2 ] { 0, 1 };
    property position = (257, 99) ;
    property weight = None ;
}
variable hear-bark {
    type discrete [ 2 ] { 0, 1 };
    property position = (296, 268) ;
    property weight = None ;
}
variable kid {
    type discrete [ 2 ] { 0, 1 };
    property position = (100, 165) ;
    property weight = None ;
}
variable light-on {
    type discrete [ 2 ] { 0, 1 };
    property position = (218, 195) ;
    property weight = None ;
}
probability ( bowel-problem ) {
    table 0.01, 0.99 ;
}
probability ( dog-out | bowel-problem, family-out ) {
    ( 0, 0 ) 0.99, 0.01;
    ( 0, 1 ) 0.9, 0.1;
    ( 1, 0 ) 0.97, 0.03;
    ( 1, 1 ) 0.3, 0.7;

}
probability ( family-out ) {
    table 0.15, 0.85 ;
}
probability ( hear-bark | dog-out ) {
    ( 0 ) 0.7, 0.3;
    ( 1 ) 0.01, 0.99;

}
probability ( kid ) {
    table 0.3, 0.7 ;
}
probability ( light-on | family-out ) {
    ( 0 ) 0.6, 0.4;
    ( 1 ) 0.95, 0.05;

}
"""
        self.maxDiff = None
        self.assertEqual(self.writer.__str__(), self.expected_string)

    def test_write_read_equal(self):
        self.writer.write_bif("test_bif.bif")
        reader = BIFReader("test_bif.bif")
        read_model = reader.get_model(state_name_type=int)
        self.assertEqual(sorted(self.model.nodes()), sorted(read_model.nodes()))
        self.assertEqual(sorted(self.model.edges()), sorted(read_model.edges()))
        for var in self.model.nodes():
            self.assertEqual(self.model.get_cpds(var), read_model.get_cpds(var))
        os.remove("test_bif.bif")


class TestBIFReaderTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.reader = BIFReader(
            string="""
                // Bayesian Network in the Interchange Format
                // Produced by BayesianNetworks package in JavaBayes
                // Output created Sun Nov 02 17:49:49 GMT+00:00 1997
                // Bayesian network
                network "Dog-Problem" { //5 variables and 5 probability distributions
                        property "credal-set constant-density-bounded 1.1" ;
                }
                variable  "light-on" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (218, 195)" ;
                }
                variable  "bowel-problem" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (335, 99)" ;
                }
                variable  "dog-out" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (300, 195)" ;
                }
                variable  "hear-bark" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (296, 268)" ;
                }
                variable  "family-out" { //2 values
                        type discrete[2] {  "true"  "false" };
                        property "position = (257, 99)" ;
                }
                probability (  "light-on"  "family-out" ) { //2 variable(s) and 4 values
                        (true) 0.6 0.4 ;
                        (false) 0.05 0.95 ;
                }
                probability (  "bowel-problem" ) { //1 variable(s) and 2 values
                        table 0.01 0.99 ;
                }
                probability (  "dog-out"  "bowel-problem"  "family-out" ) { //3 variable(s) and 8 values
                        table 0.99 0.97 0.9 0.3 0.01 0.03 0.1 0.7 ;
                }
                probability (  "hear-bark"  "dog-out" ) { //2 variable(s) and 4 values
                        table 0.7 0.01 0.3 0.99 ;
                }
                probability (  "family-out" ) { //1 variable(s) and 2 values
                        table 0.15 0.85 ;
                }
                """,
            include_properties=True,
        )

        self.water_model = BIFReader(
            "pgmpy/tests/test_readwrite/testdata/water.bif", include_properties=True
        )

    def test_network_name(self):
        name_expected = "Dog-Problem"
        self.assertEqual(self.reader.network_name, name_expected)

    def test_get_variables(self):
        var_expected = [
            "light-on",
            "bowel-problem",
            "dog-out",
            "hear-bark",
            "family-out",
        ]
        self.assertListEqual(self.reader.get_variables(), var_expected)

    def test_states(self):
        states_expected = {
            "bowel-problem": ["true", "false"],
            "dog-out": ["true", "false"],
            "family-out": ["true", "false"],
            "hear-bark": ["true", "false"],
            "light-on": ["true", "false"],
        }
        states = self.reader.get_states()
        for variable in states_expected:
            self.assertListEqual(states_expected[variable], states[variable])

    def test_get_property(self):
        property_expected = {
            "bowel-problem": ["position = (335, 99)"],
            "dog-out": ["position = (300, 195)"],
            "family-out": ["position = (257, 99)"],
            "hear-bark": ["position = (296, 268)"],
            "light-on": ["position = (218, 195)"],
        }
        prop = self.reader.get_property()
        for variable in property_expected:
            self.assertListEqual(property_expected[variable], prop[variable])

    def test_get_values(self):
        cpd_expected = {
            "bowel-problem": np.array([[0.01], [0.99]]),
            "dog-out": np.array([[0.99, 0.97, 0.9, 0.3], [0.01, 0.03, 0.1, 0.7]]),
            "family-out": np.array([[0.15], [0.85]]),
            "hear-bark": np.array([[0.7, 0.01], [0.3, 0.99]]),
            "light-on": np.array([[0.6, 0.05], [0.4, 0.95]]),
        }
        cpd = self.reader.variable_cpds
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable], cpd[variable])

    def test_get_values_reordered(self):
        cancer_values1 = BIFReader(
            string="""
                network unknown {
                }
                variable Pollution {
                  type discrete [ 2 ] { low, high };
                }
                variable Smoker {
                  type discrete [ 2 ] { True, False };
                }
                variable Cancer {
                  type discrete [ 2 ] { True, False };
                }
                probability ( Cancer | Pollution, Smoker ) {
                  (low, True) 0.03, 0.97;
                  (low, False) 0.001, 0.999;
                  (high, True) 0.05, 0.95;
                  (high, False) 0.02, 0.98;
                }"""
        ).get_values()

        cancer_values2 = BIFReader(
            string="""
                network unknown {
                }
                variable Pollution {
                  type discrete [ 2 ] { low, high };
                }
                variable Smoker {
                  type discrete [ 2 ] { True, False };
                }
                variable Cancer {
                  type discrete [ 2 ] { True, False };
                }
                probability ( Cancer | Pollution, Smoker ) {
                  (low, True) 0.03, 0.97;
                  (high, True) 0.05, 0.95;
                  (low, False) 0.001, 0.999;
                  (high, False) 0.02, 0.98;
                }"""
        ).get_values()

        for var in cancer_values1:
            np_test.assert_array_equal(cancer_values1[var], cancer_values2[var])

    def test_get_parents(self):
        parents_expected = {
            "bowel-problem": [],
            "dog-out": ["bowel-problem", "family-out"],
            "family-out": [],
            "hear-bark": ["dog-out"],
            "light-on": ["family-out"],
        }
        parents = self.reader.get_parents()
        for variable in parents_expected:
            self.assertListEqual(parents_expected[variable], parents[variable])

    def test_get_edges(self):
        edges_expected = [
            ["family-out", "dog-out"],
            ["bowel-problem", "dog-out"],
            ["family-out", "light-on"],
            ["dog-out", "hear-bark"],
        ]
        self.assertListEqual(sorted(self.reader.variable_edges), sorted(edges_expected))

    def test_get_model(self):
        edges_expected = [
            ("family-out", "dog-out"),
            ("bowel-problem", "dog-out"),
            ("family-out", "light-on"),
            ("dog-out", "hear-bark"),
        ]
        nodes_expected = [
            "bowel-problem",
            "hear-bark",
            "light-on",
            "dog-out",
            "family-out",
        ]
        edge_expected = {
            "bowel-problem": {"dog-out": {"weight": None}},
            "dog-out": {"hear-bark": {"weight": None}},
            "family-out": {"dog-out": {"weight": None}, "light-on": {"weight": None}},
            "hear-bark": {},
            "light-on": {},
        }
        node_expected = {
            "bowel-problem": {"weight": None, "position": "(335, 99)"},
            "dog-out": {"weight": None, "position": "(300, 195)"},
            "family-out": {"weight": None, "position": "(257, 99)"},
            "hear-bark": {"weight": None, "position": "(296, 268)"},
            "light-on": {"weight": None, "position": "(218, 195)"},
        }
        cpds_expected = [
            TabularCPD(
                variable="bowel-problem",
                variable_card=2,
                values=np.array([[0.01], [0.99]]),
                state_names={"bowel-problem": ["true", "false"]},
            ),
            TabularCPD(
                variable="dog-out",
                variable_card=2,
                values=np.array([[0.99, 0.97, 0.9, 0.3], [0.01, 0.03, 0.1, 0.7]]),
                evidence=["bowel-problem", "family-out"],
                evidence_card=[2, 2],
                state_names={
                    "dog-out": ["true", "false"],
                    "bowel-problem": ["true", "false"],
                    "family-out": ["true", "false"],
                },
            ),
            TabularCPD(
                variable="family-out",
                variable_card=2,
                values=np.array([[0.15], [0.85]]),
                state_names={"family-out": ["true", "false"]},
            ),
            TabularCPD(
                variable="hear-bark",
                variable_card=2,
                values=np.array([[0.7, 0.01], [0.3, 0.99]]),
                evidence=["dog-out"],
                evidence_card=[2],
                state_names={
                    "hear-bark": ["true", "false"],
                    "dog-out": ["true", "false"],
                },
            ),
            TabularCPD(
                variable="light-on",
                variable_card=2,
                values=np.array([[0.6, 0.05], [0.4, 0.95]]),
                evidence=["family-out"],
                evidence_card=[2],
                state_names={
                    "light-on": ["true", "false"],
                    "family-out": ["true", "false"],
                },
            ),
        ]
        model = self.reader.get_model()
        model_cpds = model.get_cpds()
        for cpd_index in range(5):
            self.assertEqual(model_cpds[cpd_index], cpds_expected[cpd_index])

        self.assertDictEqual(dict(model.nodes), node_expected)
        self.assertDictEqual(dict(model.adj), edge_expected)

        self.assertListEqual(sorted(model.nodes()), sorted(nodes_expected))
        self.assertListEqual(sorted(model.edges()), sorted(edges_expected))

    def test_water_model(self):
        model = self.water_model.get_model()
        self.assertEqual(len(model.nodes()), 32)
        self.assertEqual(len(model.edges()), 66)
        self.assertEqual(len(model.get_cpds()), 32)

    def test_default_attribut_equal_table(self):
        default_reader = BIFReader(
            string="""
                network "Dog-Problem" {
                }
                variable  "light-on" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (218, 195)" ;
                }
                variable  "bowel-problem" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (335, 99)" ;
                }
                variable  "dog-out" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (300, 195)" ;
                }
                variable  "hear-bark" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (296, 268)" ;
                }
                variable  "family-out" {
                        type discrete[2] {  "true"  "false" };
                        property "position = (257, 99)" ;
                }
                probability (  "light-on"  "family-out" ) {
                        (true) 0.6 0.4 ;
                        (false) 0.05 0.95 ;
                }
                probability (  "bowel-problem" ) {
                        default 0.01 0.99 ;
                }
                probability (  "dog-out"  "bowel-problem"  "family-out" ) {
                        default 0.99 0.97 0.9 0.3 0.01 0.03 0.1 0.7 ;
                }
                probability (  "hear-bark"  "dog-out" ) {
                        default 0.7 0.01 0.3 0.99 ;
                }
                probability (  "family-out" ) {
                        default 0.15 0.85 ;
                }
                """,
            include_properties=True,
        )
        table_model = self.reader.get_model()
        default_model = default_reader.get_model()
        self.assertEqual(sorted(table_model.nodes()), sorted(default_model.nodes()))
        self.assertEqual(sorted(table_model.edges()), sorted(default_model.edges()))
        for var in table_model.nodes():
            self.assertEqual(table_model.get_cpds(var), default_model.get_cpds(var))

    def tearDown(self):
        del self.reader
        config.set_backend("numpy")


class TestBIFWriterTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        variables = [
            "kid",
            "bowel-problem",
            "dog-out",
            "family-out",
            "hear-bark",
            "light-on",
        ]

        edges = [
            ["family-out", "dog-out"],
            ["bowel-problem", "dog-out"],
            ["family-out", "light-on"],
            ["dog-out", "hear-bark"],
        ]

        cpds = {
            "kid": np.array([[0.3], [0.7]]),
            "bowel-problem": np.array([[0.01], [0.99]]),
            "dog-out": np.array([[0.99, 0.9, 0.97, 0.3], [0.01, 0.1, 0.03, 0.7]]),
            "family-out": np.array([[0.15], [0.85]]),
            "hear-bark": np.array([[0.7, 0.01], [0.3, 0.99]]),
            "light-on": np.array([[0.6, 0.95], [0.4, 0.05]]),
        }

        states = {
            "kid": ["true", "false"],
            "bowel-problem": ["true", "false"],
            "dog-out": ["true", "false"],
            "family-out": ["true", "false"],
            "hear-bark": ["true", "false"],
            "light-on": ["true", "false"],
        }

        parents = {
            "kid": [],
            "bowel-problem": [],
            "dog-out": ["bowel-problem", "family-out"],
            "family-out": [],
            "hear-bark": ["dog-out"],
            "light-on": ["family-out"],
        }

        properties = {
            "kid": ["position = (100, 165)"],
            "bowel-problem": ["position = (335, 99)"],
            "dog-out": ["position = (300, 195)"],
            "family-out": ["position = (257, 99)"],
            "hear-bark": ["position = (296, 268)"],
            "light-on": ["position = (218, 195)"],
        }

        self.model = BayesianNetwork()
        self.model.add_nodes_from(variables)
        self.model.add_edges_from(edges)

        tabular_cpds = []
        for var in sorted(cpds.keys()):
            values = cpds[var]
            cpd = TabularCPD(
                var,
                len(states[var]),
                values,
                evidence=parents[var],
                evidence_card=[
                    len(states[evidence_var]) for evidence_var in parents[var]
                ],
            )
            tabular_cpds.append(cpd)
        self.model.add_cpds(*tabular_cpds)

        for node, properties in properties.items():
            for prop in properties:
                prop_name, prop_value = map(lambda t: t.strip(), prop.split("="))
                self.model.nodes[node][prop_name] = prop_value

        self.writer = BIFWriter(model=self.model, round_values=4)

    def test_str(self):
        self.expected_string = """network unknown {
}
variable bowel-problem {
    type discrete [ 2 ] { 0, 1 };
    property position = (335, 99) ;
    property weight = None ;
}
variable dog-out {
    type discrete [ 2 ] { 0, 1 };
    property position = (300, 195) ;
    property weight = None ;
}
variable family-out {
    type discrete [ 2 ] { 0, 1 };
    property position = (257, 99) ;
    property weight = None ;
}
variable hear-bark {
    type discrete [ 2 ] { 0, 1 };
    property position = (296, 268) ;
    property weight = None ;
}
variable kid {
    type discrete [ 2 ] { 0, 1 };
    property position = (100, 165) ;
    property weight = None ;
}
variable light-on {
    type discrete [ 2 ] { 0, 1 };
    property position = (218, 195) ;
    property weight = None ;
}
probability ( bowel-problem ) {
    table 0.01, 0.99 ;
}
probability ( dog-out | bowel-problem, family-out ) {
    ( 0, 0 ) 0.99, 0.01;
    ( 0, 1 ) 0.9, 0.1;
    ( 1, 0 ) 0.97, 0.03;
    ( 1, 1 ) 0.3, 0.7;

}
probability ( family-out ) {
    table 0.15, 0.85 ;
}
probability ( hear-bark | dog-out ) {
    ( 0 ) 0.7, 0.3;
    ( 1 ) 0.01, 0.99;

}
probability ( kid ) {
    table 0.3, 0.7 ;
}
probability ( light-on | family-out ) {
    ( 0 ) 0.6, 0.4;
    ( 1 ) 0.95, 0.05;

}
"""
        self.maxDiff = None
        self.assertEqual(self.writer.__str__(), self.expected_string)

    def test_write_read_equal(self):
        self.writer.write_bif("test_bif.bif")
        reader = BIFReader("test_bif.bif")
        read_model = reader.get_model(state_name_type=int)
        self.assertEqual(sorted(self.model.nodes()), sorted(read_model.nodes()))
        self.assertEqual(sorted(self.model.edges()), sorted(read_model.edges()))
        for var in self.model.nodes():
            self.assertEqual(self.model.get_cpds(var), read_model.get_cpds(var))
        os.remove("test_bif.bif")

    def tearDown(self):
        config.set_backend("numpy")
