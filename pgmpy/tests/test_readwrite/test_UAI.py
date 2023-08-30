import unittest

import networkx as nx
import numpy as np

from pgmpy import config
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork, MarkovNetwork
from pgmpy.readwrite import UAIReader, UAIWriter


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

        string_with_comment = """MARKOV
3
2 2 3
2 # comment
2 0 1
3 0 1 2
# comment
4
 4.000 2.400
 1.000 0.000

12 #another comment
 2.2500 3.2500 3.7500
 0.0000 0.0000 10.0000
 1.8750 4.0000 3.3330
 2.0000 2.0000 3.4000"""
        self.maxDiff = None
        self.reader_string = UAIReader(string=string)
        self.reader_string_with_comment = UAIReader(string=string_with_comment)
        self.reader_file = UAIReader("pgmpy/tests/test_readwrite/testdata/grid4x4.uai")

    def test_get_network_type(self):
        network_type_expected = "MARKOV"
        self.assertEqual(self.reader_string.network_type, network_type_expected)
        self.assertEqual(
            self.reader_string_with_comment.network_type, network_type_expected
        )

    def test_get_variables(self):
        variables_expected = ["var_0", "var_1", "var_2"]
        self.assertListEqual(self.reader_string.variables, variables_expected)
        self.assertListEqual(
            self.reader_string_with_comment.variables, variables_expected
        )

    def test_get_domain(self):
        domain_expected = {"var_1": "2", "var_2": "3", "var_0": "2"}
        self.assertDictEqual(self.reader_string.domain, domain_expected)
        self.assertDictEqual(self.reader_string_with_comment.domain, domain_expected)

    def test_get_edges(self):
        edges_expected = {("var_0", "var_1"), ("var_0", "var_2"), ("var_1", "var_2")}
        self.assertSetEqual(self.reader_string.edges, edges_expected)
        self.assertSetEqual(self.reader_string_with_comment.edges, edges_expected)

    def test_get_tables(self):
        tables_expected = [
            (["var_0", "var_1"], ["4.000", "2.400", "1.000", "0.000"]),
            (
                ["var_0", "var_1", "var_2"],
                [
                    "2.2500",
                    "3.2500",
                    "3.7500",
                    "0.0000",
                    "0.0000",
                    "10.0000",
                    "1.8750",
                    "4.0000",
                    "3.3330",
                    "2.0000",
                    "2.0000",
                    "3.4000",
                ],
            ),
        ]
        self.assertListEqual(self.reader_string.tables, tables_expected)
        self.assertListEqual(self.reader_string_with_comment.tables, tables_expected)

    def test_get_model(self):
        model = self.reader_string.get_model()
        edge_expected = {
            "var_2": {"var_0": {"weight": None}, "var_1": {"weight": None}},
            "var_0": {"var_2": {"weight": None}, "var_1": {"weight": None}},
            "var_1": {"var_2": {"weight": None}, "var_0": {"weight": None}},
        }

        self.assertListEqual(sorted(model.nodes()), sorted(["var_0", "var_2", "var_1"]))
        if nx.__version__.startswith("1"):
            self.assertDictEqual(dict(model.edge), edge_expected)
        else:
            self.assertDictEqual(dict(model.adj), edge_expected)

    def test_read_file(self):
        model = self.reader_file.get_model()
        node_expected = {
            "var_3": {},
            "var_8": {},
            "var_5": {},
            "var_14": {},
            "var_15": {},
            "var_0": {},
            "var_9": {},
            "var_7": {},
            "var_6": {},
            "var_13": {},
            "var_10": {},
            "var_12": {},
            "var_1": {},
            "var_11": {},
            "var_2": {},
            "var_4": {},
        }
        self.assertDictEqual(dict(model.nodes), node_expected)


class TestUAIWriter(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
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
            "dog-out": np.array([[0.99, 0.01, 0.97, 0.03], [0.9, 0.1, 0.3, 0.7]]),
            "family-out": np.array([[0.15], [0.85]]),
            "hear-bark": np.array([[0.7, 0.3], [0.01, 0.99]]),
            "light-on": np.array([[0.6, 0.4], [0.05, 0.95]]),
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

        self.bayesmodel = BayesianNetwork()
        self.bayesmodel.add_nodes_from(variables)
        self.bayesmodel.add_edges_from(edges)

        tabular_cpds = []
        for var, values in cpds.items():
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
        self.bayesmodel.add_cpds(*tabular_cpds)
        self.bayeswriter = UAIWriter(self.bayesmodel)

        edges = {("var_0", "var_1"), ("var_0", "var_2"), ("var_1", "var_2")}
        self.markovmodel = MarkovNetwork(edges)
        tables = [
            (["var_0", "var_1"], ["4.000", "2.400", "1.000", "0.000"]),
            (
                ["var_0", "var_1", "var_2"],
                [
                    "2.2500",
                    "3.2500",
                    "3.7500",
                    "0.0000",
                    "0.0000",
                    "10.0000",
                    "1.8750",
                    "4.0000",
                    "3.3330",
                    "2.0000",
                    "2.0000",
                    "3.4000",
                ],
            ),
        ]
        domain = {"var_1": "2", "var_2": "3", "var_0": "2"}
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
6
2 2 2 2 2 2
6
1 0
3 2 0 1
1 2
2 1 3
1 4
2 2 5

2
0.01 0.99
8
0.99 0.01 0.97 0.03 0.9 0.1 0.3 0.7
2
0.15 0.85
4
0.7 0.3 0.01 0.99
2
0.3 0.7
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
        self.assertEqual(
            str(self.markovwriter.__str__()), str(self.expected_markov_file)
        )


class TestUAIReaderTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

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

        string_with_comment = """MARKOV
3
2 2 3
2 # comment
2 0 1
3 0 1 2
# comment
4
 4.000 2.400
 1.000 0.000

12 #another comment
 2.2500 3.2500 3.7500
 0.0000 0.0000 10.0000
 1.8750 4.0000 3.3330
 2.0000 2.0000 3.4000"""
        self.maxDiff = None
        self.reader_string = UAIReader(string=string)
        self.reader_string_with_comment = UAIReader(string=string_with_comment)
        self.reader_file = UAIReader("pgmpy/tests/test_readwrite/testdata/grid4x4.uai")

    def test_get_network_type(self):
        network_type_expected = "MARKOV"
        self.assertEqual(self.reader_string.network_type, network_type_expected)
        self.assertEqual(
            self.reader_string_with_comment.network_type, network_type_expected
        )

    def test_get_variables(self):
        variables_expected = ["var_0", "var_1", "var_2"]
        self.assertListEqual(self.reader_string.variables, variables_expected)
        self.assertListEqual(
            self.reader_string_with_comment.variables, variables_expected
        )

    def test_get_domain(self):
        domain_expected = {"var_1": "2", "var_2": "3", "var_0": "2"}
        self.assertDictEqual(self.reader_string.domain, domain_expected)
        self.assertDictEqual(self.reader_string_with_comment.domain, domain_expected)

    def test_get_edges(self):
        edges_expected = {("var_0", "var_1"), ("var_0", "var_2"), ("var_1", "var_2")}
        self.assertSetEqual(self.reader_string.edges, edges_expected)
        self.assertSetEqual(self.reader_string_with_comment.edges, edges_expected)

    def test_get_tables(self):
        tables_expected = [
            (["var_0", "var_1"], ["4.000", "2.400", "1.000", "0.000"]),
            (
                ["var_0", "var_1", "var_2"],
                [
                    "2.2500",
                    "3.2500",
                    "3.7500",
                    "0.0000",
                    "0.0000",
                    "10.0000",
                    "1.8750",
                    "4.0000",
                    "3.3330",
                    "2.0000",
                    "2.0000",
                    "3.4000",
                ],
            ),
        ]
        self.assertListEqual(self.reader_string.tables, tables_expected)
        self.assertListEqual(self.reader_string_with_comment.tables, tables_expected)

    def test_get_model(self):
        model = self.reader_string.get_model()
        edge_expected = {
            "var_2": {"var_0": {"weight": None}, "var_1": {"weight": None}},
            "var_0": {"var_2": {"weight": None}, "var_1": {"weight": None}},
            "var_1": {"var_2": {"weight": None}, "var_0": {"weight": None}},
        }

        self.assertListEqual(sorted(model.nodes()), sorted(["var_0", "var_2", "var_1"]))
        if nx.__version__.startswith("1"):
            self.assertDictEqual(dict(model.edge), edge_expected)
        else:
            self.assertDictEqual(dict(model.adj), edge_expected)

    def test_read_file(self):
        model = self.reader_file.get_model()
        node_expected = {
            "var_3": {},
            "var_8": {},
            "var_5": {},
            "var_14": {},
            "var_15": {},
            "var_0": {},
            "var_9": {},
            "var_7": {},
            "var_6": {},
            "var_13": {},
            "var_10": {},
            "var_12": {},
            "var_1": {},
            "var_11": {},
            "var_2": {},
            "var_4": {},
        }
        self.assertDictEqual(dict(model.nodes), node_expected)

    def tearDown(self):
        config.set_backend("numpy")


class TestUAIWriterTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.maxDiff = None
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
            "dog-out": np.array([[0.99, 0.01, 0.97, 0.03], [0.9, 0.1, 0.3, 0.7]]),
            "family-out": np.array([[0.15], [0.85]]),
            "hear-bark": np.array([[0.7, 0.3], [0.01, 0.99]]),
            "light-on": np.array([[0.6, 0.4], [0.05, 0.95]]),
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

        self.bayesmodel = BayesianNetwork()
        self.bayesmodel.add_nodes_from(variables)
        self.bayesmodel.add_edges_from(edges)

        tabular_cpds = []
        for var, values in cpds.items():
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
        self.bayesmodel.add_cpds(*tabular_cpds)
        self.bayeswriter = UAIWriter(self.bayesmodel, round_values=4)

        edges = {("var_0", "var_1"), ("var_0", "var_2"), ("var_1", "var_2")}
        self.markovmodel = MarkovNetwork(edges)
        tables = [
            (["var_0", "var_1"], ["4.000", "2.400", "1.000", "0.000"]),
            (
                ["var_0", "var_1", "var_2"],
                [
                    "2.2500",
                    "3.2500",
                    "3.7500",
                    "0.0000",
                    "0.0000",
                    "10.0000",
                    "1.8750",
                    "4.0000",
                    "3.3330",
                    "2.0000",
                    "2.0000",
                    "3.4000",
                ],
            ),
        ]
        domain = {"var_1": "2", "var_2": "3", "var_0": "2"}
        factors = []
        for table in tables:
            variables = table[0]
            cardinality = [int(domain[var]) for var in variables]
            values = list(map(float, table[1]))
            factor = DiscreteFactor(variables, cardinality, values)
            factors.append(factor)
        self.markovmodel.add_factors(*factors)
        self.markovwriter = UAIWriter(self.markovmodel, round_values=4)

    def test_bayes_model(self):
        self.expected_bayes_file = """BAYES
6
2 2 2 2 2 2
6
1 0
3 2 0 1
1 2
2 1 3
1 4
2 2 5

2
0.01 0.99
8
0.99 0.01 0.97 0.03 0.9 0.1 0.3 0.7
2
0.15 0.85
4
0.7 0.3 0.01 0.99
2
0.3 0.7
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
        self.assertEqual(
            str(self.markovwriter.__str__()), str(self.expected_markov_file)
        )

    def tearDown(self):
        config.set_backend("numpy")
