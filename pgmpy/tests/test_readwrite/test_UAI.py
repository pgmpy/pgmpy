import unittest
from pgmpy.readwrite import UAIReader


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
                            ['2.2500', '3.2500', '3.7500', '0.0000', '0.0000', '10.0000', '1.8750', '4.0000', '3.3330', '2.0000', '2.0000', '3.4000'])]
        self.assertListEqual(self.reader_string.tables, tables_expected)

    def test_get_model(self):
        model = self.reader_string.get_model()
        edge_expected = {'var_2': {'var_0': {},
                                   'var_1': {}},
                         'var_0': {'var_2': {},
                                   'var_1': {}},
                         'var_1': {'var_2': {},
                                   'var_0': {}}}
        self.assertListEqual(sorted(model.nodes()), sorted(['var_0', 'var_2', 'var_1']))
        self.assertDictEqual(model.edge, edge_expected)
