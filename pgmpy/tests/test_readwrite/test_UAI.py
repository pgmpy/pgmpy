import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork
from pgmpy.readwrite import UAIReader, UAIWriter


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("torch not installed")
        config.set_backend("torch")
    yield request.param
    config.set_backend("numpy")


@pytest.fixture
def uai_string():
    return """MARKOV
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


@pytest.fixture
def uai_string_with_comment():
    return """MARKOV
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


@pytest.fixture
def reader_string(uai_string, backend):
    return UAIReader(string=uai_string)


@pytest.fixture
def reader_string_with_comment(uai_string_with_comment, backend):
    return UAIReader(string=uai_string_with_comment)


@pytest.fixture
def reader_file(backend):
    return UAIReader("pgmpy/tests/test_readwrite/testdata/grid4x4.uai")


class TestUAIReader:
    def test_get_network_type(self, reader_string, reader_string_with_comment):
        network_type_expected = "MARKOV"
        assert reader_string.network_type == network_type_expected
        assert reader_string_with_comment.network_type == network_type_expected

    def test_get_variables(self, reader_string, reader_string_with_comment):
        variables_expected = ["var_0", "var_1", "var_2"]
        assert reader_string.variables == variables_expected
        assert reader_string_with_comment.variables == variables_expected

    def test_get_domain(self, reader_string, reader_string_with_comment):
        domain_expected = {"var_1": "2", "var_2": "3", "var_0": "2"}
        assert reader_string.domain == domain_expected
        assert reader_string_with_comment.domain == domain_expected

    def test_get_edges(self, reader_string, reader_string_with_comment):
        edges_expected = {("var_0", "var_1"), ("var_0", "var_2"), ("var_1", "var_2")}
        assert reader_string.edges == edges_expected
        assert reader_string_with_comment.edges == edges_expected

    def test_get_tables(self, reader_string, reader_string_with_comment):
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
        assert reader_string.tables == tables_expected
        assert reader_string_with_comment.tables == tables_expected

    def test_get_model(self, reader_string):
        model = reader_string.read()
        edge_expected = {
            "var_2": {"var_0": {"weight": None}, "var_1": {"weight": None}},
            "var_0": {"var_2": {"weight": None}, "var_1": {"weight": None}},
            "var_1": {"var_2": {"weight": None}, "var_0": {"weight": None}},
        }

        assert sorted(model.nodes()) == sorted(["var_0", "var_2", "var_1"])
        assert dict(model.adj) == edge_expected

    def test_read_file(self, reader_file):
        model = reader_file.read()
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
        assert dict(model.nodes) == node_expected


@pytest.fixture
def bayes_model(backend):
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

    bayesmodel = DiscreteBayesianNetwork()
    bayesmodel.add_nodes_from(variables)
    bayesmodel.add_edges_from(edges)

    tabular_cpds = []
    for var, values in cpds.items():
        cpd = TabularCPD(
            var,
            len(states[var]),
            values,
            evidence=parents[var],
            evidence_card=[len(states[evidence_var]) for evidence_var in parents[var]],
        )
        tabular_cpds.append(cpd)
    bayesmodel.add_cpds(*tabular_cpds)
    return bayesmodel


@pytest.fixture
def markov_model(backend):
    edges = {("var_0", "var_1"), ("var_0", "var_2"), ("var_1", "var_2")}
    markovmodel = DiscreteMarkovNetwork(edges)
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
    markovmodel.add_factors(*factors)
    return markovmodel


class TestUAIWriter:
    @pytest.fixture
    def bayes_writer(self, bayes_model, backend):
        return UAIWriter(bayes_model, round_values=4 if backend == "torch" else None)

    @pytest.fixture
    def markov_writer(self, markov_model, backend):
        return UAIWriter(markov_model, round_values=4 if backend == "torch" else None)

    def test_bayes_model(self, bayes_writer):
        expected_bayes_file = """BAYES
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
        assert str(bayes_writer) == str(expected_bayes_file)

    def test_markov_model(self, markov_writer):
        expected_markov_file = """MARKOV
3
2 2 3
2
2 0 1
3 0 1 2

4
4.0 2.4 1.0 0.0
12
2.25 3.25 3.75 0.0 0.0 10.0 1.875 4.0 3.333 2.0 2.0 3.4"""
        assert str(markov_writer) == str(expected_markov_file)
