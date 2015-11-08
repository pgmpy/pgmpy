import unittest
from pgmpy.readwrite import BifReader

class TestBifReader(unittest.TestCase):

    def setUp(self):

        self.reader = BifReader(string = """
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
    def test_get_variables(self):
        var_expected = ['light-on', 'bowel-problem', 'dog-out',
                        'hear-bark', 'family-out']
        self.assertListEqual(self.reader.variable_names, var_expected)
    
    def test_states(self):
        states_expected = {'bowel-problem': ['true', 'false'],
                           'dog-out': ['true', 'false'],
                           'family-out': ['true', 'false'],
                           'hear-bark': ['true', 'false'],
                           'light-on': ['true', 'false']}
        states = self.reader.variable_states
        for variable in states_expected:
            self.assertListEqual(states_expected[variable],
                                states[variable])
    
    def test_get_property(self):
        property_expected = {'bowel-problem': ['position = (335, 99)'],
                             'dog-out': ['position = (300, 195)'],
                             'family-out': ['position = (257, 99)'],
                             'hear-bark': ['position = (296, 268)'],
                             'light-on': ['position = (218, 195)']}
        prop = self.reader.variable_properties
        for variable in property_expected:
            self.assertListEqual(property_expected[variable],
                                 prop[variable])

