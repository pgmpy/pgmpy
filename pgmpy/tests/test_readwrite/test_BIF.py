import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.readwrite import BIFReader, BIFWriter
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.extern.six.moves import map, range


class TestBIFReader(unittest.TestCase):

    def setUp(self):

        self.reader = BIFReader(string="""
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
""")

    def test_network_name(self):

        name_expected = 'Dog-Problem'
        self.assertEqual(self.reader.network_name, name_expected)

    def test_get_variables(self):

        var_expected = ['light-on', 'bowel-problem', 'dog-out',
                        'hear-bark', 'family-out']
        self.assertListEqual(self.reader.get_variables(), var_expected)

    def test_states(self):

        states_expected = {'bowel-problem': ['true', 'false'],
                           'dog-out': ['true', 'false'],
                           'family-out': ['true', 'false'],
                           'hear-bark': ['true', 'false'],
                           'light-on': ['true', 'false']}
        states = self.reader.get_states()
        for variable in states_expected:
            self.assertListEqual(states_expected[variable], states[variable])

    def test_get_property(self):

        property_expected = {'bowel-problem': ['position = (335, 99)'],
                             'dog-out': ['position = (300, 195)'],
                             'family-out': ['position = (257, 99)'],
                             'hear-bark': ['position = (296, 268)'],
                             'light-on': ['position = (218, 195)']}
        prop = self.reader.get_property()
        for variable in property_expected:
            self.assertListEqual(property_expected[variable],
                                 prop[variable])

    def test_get_values(self):

        cpd_expected = {'bowel-problem': np.array([[0.01],
                                                   [0.99]]),
                        'dog-out': np.array([[0.99, 0.97, 0.9, 0.3],
                                             [0.01, 0.03, 0.1, 0.7]]),
                        'family-out': np.array([[0.15],
                                                [0.85]]),
                        'hear-bark': np.array([[0.7, 0.01],
                                               [0.3, 0.99]]),
                        'light-on': np.array([[0.6, 0.05],
                                              [0.4, 0.95]])}
        cpd = self.reader.variable_cpds
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable],
                                       cpd[variable])

    def test_get_parents(self):

        parents_expected = {'bowel-problem': [],
                            'dog-out': ['bowel-problem', 'family-out'],
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
        self.assertListEqual(sorted(self.reader.variable_edges),
                             sorted(edges_expected))

    def test_get_model(self):
        edges_expected = [('family-out', 'dog-out'),
                          ('bowel-problem', 'dog-out'),
                          ('family-out', 'light-on'),
                          ('dog-out', 'hear-bark')]
        nodes_expected = ['bowel-problem', 'hear-bark', 'light-on',
                          'dog-out', 'family-out']
        edge_expected = {'bowel-problem': {'dog-out': {'weight': None}},
                         'dog-out': {'hear-bark': {'weight': None}},
                         'family-out': {'dog-out': {'weight': None},
                                        'light-on': {'weight': None}},
                         'hear-bark': {},
                         'light-on': {}}
        node_expected = {'bowel-problem': {'weight': None,
                                           'position': '(335, 99)'},
                         'dog-out': {'weight': None,
                                     'position': '(300, 195)'},
                         'family-out': {'weight': None,
                                        'position': '(257, 99)'},
                         'hear-bark': {'weight': None,
                                       'position': '(296, 268)'},
                         'light-on': {'weight': None,
                                      'position': '(218, 195)'}}
        cpds_expected = [np.array([[0.01],
                                   [0.99]]),
                         np.array([[0.99, 0.97, 0.9, 0.3],
                                   [0.01, 0.03, 0.1, 0.7]]),
                         np.array([[0.15],
                                   [0.85]]),
                         np.array([[0.7, 0.01],
                                   [0.3, 0.99]]),
                         np.array([[0.6, 0.05],
                                   [0.4, 0.95]])]
        model = self.reader.get_model()
        for cpd_index in range(0, len(cpds_expected)):
            np_test.assert_array_equal(model.get_cpds()[cpd_index].get_values(),
                                       cpds_expected[cpd_index])
        self.assertDictEqual(model.node, node_expected)
        self.assertDictEqual(model.edge, edge_expected)
        self.assertListEqual(sorted(model.nodes()), sorted(nodes_expected))
        self.assertListEqual(sorted(model.edges()), sorted(edges_expected))

    def tearDown(self):
        del self.reader


class TestBIFWriter(unittest.TestCase):

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

        properties = {'bowel-problem': ['position = (335, 99)'],
                      'dog-out': ['position = (300, 195)'],
                      'family-out': ['position = (257, 99)'],
                      'hear-bark': ['position = (296, 268)'],
                      'light-on': ['position = (218, 195)']}

        self.model = BayesianModel(edges)

        tabular_cpds = []
        for var in sorted(cpds.keys()):
            values = cpds[var]
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

        self.writer = BIFWriter(model=self.model)

    def test_str(self):
        self.expected_string = """network unknown {
}
variable bowel-problem {
    type discrete [ 2 ] { bowel-problem_0, bowel-problem_1 };
    property position = (335, 99) ;
}
variable dog-out {
    type discrete [ 2 ] { dog-out_0, dog-out_1 };
    property position = (300, 195) ;
}
variable family-out {
    type discrete [ 2 ] { family-out_0, family-out_1 };
    property position = (257, 99) ;
}
variable hear-bark {
    type discrete [ 2 ] { hear-bark_0, hear-bark_1 };
    property position = (296, 268) ;
}
variable light-on {
    type discrete [ 2 ] { light-on_0, light-on_1 };
    property position = (218, 195) ;
}
probability ( bowel-problem ) {
    table 0.01, 0.99 ;
}
probability ( dog-out | bowel-problem, family-out ) {
    table 0.99, 0.01, 0.97, 0.03, 0.9, 0.1, 0.3, 0.7 ;
}
probability ( family-out ) {
    table 0.15, 0.85 ;
}
probability ( hear-bark | dog-out ) {
    table 0.7, 0.3, 0.01, 0.99 ;
}
probability ( light-on | family-out ) {
    table 0.6, 0.4, 0.05, 0.95 ;
}
"""
        self.maxDiff = None
        self.assertEqual(self.writer.__str__(), self.expected_string)
