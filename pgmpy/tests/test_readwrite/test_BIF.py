import unittest
from pgmpy.readwrite import BIFReader
import numpy as np
import numpy.testing as np_test


class TestBifReader(unittest.TestCase):

    def setUp(self):

        self.reader = BIFReader(string = """
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
        table 0.6 0.05 0.4 0.95 ;
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
        self.assertEqual(self.reader.network_name,name_expected)

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
            self.assertListEqual(states_expected[variable],
                                states[variable])

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

    def test_get_cpd(self):

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

    def tearDown(self):
        del self.reader
