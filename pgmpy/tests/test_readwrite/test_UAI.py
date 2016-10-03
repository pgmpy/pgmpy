import numpy as np
import unittest

from pgmpy.readwrite import UAIReader, UAIWriter
from pgmpy.models import BayesianModel, MarkovModel
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.extern.six.moves import map


class TestUAIReader(unittest.TestCase):
    def setUp(self):
        string = """MARKOV
3
2 2 3
2
2 0 1
3 0 1 2

4
 4.000 2.400
 1.000 0.000

12
 2.2500 3.2500 3.7500
 0.0000 0.0000 10.0000
 1.8750 4.0000 3.3330
 2.0000 2.0000 3.4000"""
        self.maxDiff = None
        self.reader_string = UAIReader(string=string)
        self.reader_file = UAIReader('pgmpy/tests/test_readwrite/testdata/grid4x4.uai')

    def test_get_network_type(self):
        network_type_expected = "MARKOV"
        self.assertEqual(self.reader_string.network_type, network_type_expected)

    def test_get_variables(self):
        variables_expected = ['var_0', 'var_1', 'var_2']
        self.assertListEqual(self.reader_string.variables, variables_expected)

    def test_get_domain(self):
        domain_expected = {'var_1': '2', 'var_2': '3', 'var_0': '2'}
        self.assertDictEqual(self.reader_string.domain, domain_expected)

    def test_get_edges(self):
        edges_expected = {('var_0', 'var_1'), ('var_0', 'var_2'), ('var_1', 'var_2')}
        self.assertSetEqual(self.reader_string.edges, edges_expected)

    def test_get_tables(self):
        tables_expected = [(['var_0', 'var_1'],
                            ['4.000', '2.400', '1.000', '0.000']),
                           (['var_0', 'var_1', 'var_2'],
                            ['2.2500', '3.2500', '3.7500', '0.0000', '0.0000', '10.0000',
                             '1.8750', '4.0000', '3.3330', '2.0000', '2.0000', '3.4000'])]
        self.assertListEqual(self.reader_string.tables, tables_expected)

    def test_get_model(self):
        model = self.reader_string.get_model()
        edge_expected = {
            'var_2': {'var_0': {'weight': None},
                      'var_1': {'weight': None}},
            'var_0': {'var_2': {'weight': None},
                      'var_1': {'weight': None}},
            'var_1': {'var_2': {'weight': None},
                      'var_0': {'weight': None}}}
        self.assertListEqual(sorted(model.nodes()), sorted(['var_0', 'var_2', 'var_1']))
        self.assertDictEqual(model.edge, edge_expected)

    def test_read_file(self):
        model = self.reader_file.get_model()
        node_expected = {'var_3': {}, 'var_8': {}, 'var_5': {}, 'var_14': {},
                         'var_15': {}, 'var_0': {}, 'var_9': {}, 'var_7': {},
                         'var_6': {}, 'var_13': {}, 'var_10': {}, 'var_12': {},
                         'var_1': {}, 'var_11': {}, 'var_2': {}, 'var_4': {}}
        self.assertDictEqual(model.node, node_expected)


class TestUAIWriter(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
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
                   'dog-out': ['bowel-problem', 'family-out'],
                   'family-out': [],
                   'hear-bark': ['dog-out'],
                   'light-on': ['family-out']}

        self.bayesmodel = BayesianModel(edges)

        tabular_cpds = []
        for var, values in cpds.items():
            cpd = TabularCPD(var, len(states[var]), values,
                             evidence=parents[var],
                             evidence_card=[len(states[evidence_var])
                                            for evidence_var in parents[var]])
            tabular_cpds.append(cpd)
        self.bayesmodel.add_cpds(*tabular_cpds)
        self.bayeswriter = UAIWriter(self.bayesmodel)

        edges = {('var_0', 'var_1'), ('var_0', 'var_2'), ('var_1', 'var_2')}
        self.markovmodel = MarkovModel(edges)
        tables = [(['var_0', 'var_1'],
                   ['4.000', '2.400', '1.000', '0.000']),
                  (['var_0', 'var_1', 'var_2'],
                   ['2.2500', '3.2500', '3.7500', '0.0000', '0.0000', '10.0000',
                    '1.8750', '4.0000', '3.3330', '2.0000', '2.0000', '3.4000'])]
        domain = {'var_1': '2', 'var_2': '3', 'var_0': '2'}
        factors = []
        for table in tables:
            variables = table[0]
            cardinality = [int(domain[var]) for var in variables]
            values = list(map(float, table[1]))
            factor = DiscreteFactor(variables, cardinality, values)
            factors.append(factor)
        self.markovmodel.add_factors(*factors)
        self.markovwriter = UAIWriter(self.markovmodel)

    def test_bayes_model(self):
        self.expected_bayes_file = """BAYES
5
2 2 2 2 2
5
1 0
3 2 0 1
1 2
2 1 3
2 2 4

2
0.01 0.99
8
0.99 0.01 0.97 0.03 0.9 0.1 0.3 0.7
2
0.15 0.85
4
0.7 0.3 0.01 0.99
4
0.6 0.4 0.05 0.95"""
        self.assertEqual(str(self.bayeswriter.__str__()), str(self.expected_bayes_file))

    def test_markov_model(self):
        self.expected_markov_file = """MARKOV
3
2 2 3
2
2 0 1
3 0 1 2

4
4.0 2.4 1.0 0.0
12
2.25 3.25 3.75 0.0 0.0 10.0 1.875 4.0 3.333 2.0 2.0 3.4"""
        self.assertEqual(str(self.markovwriter.__str__()), str(self.expected_markov_file))
