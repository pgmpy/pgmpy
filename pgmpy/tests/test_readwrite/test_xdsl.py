import warnings

import numpy as np
import numpy.testing as np_test
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import XDSLReader, XDSLWriter

TEST_FILE = """<?xml version="1.0" encoding="UTF-8"?>
<!-- This network was created in trial version of GeNIe, which can be used for evaluation purposes only -->
<smile version="1.0" id="Asia" numsamples="10000" discsamples="10000">
&#x20;   <nodes>
&#x20;       <cpt id="asia" diagtype="observation" ranked="true">
&#x20;           <state id="no" />
&#x20;           <state id="yes" />
&#x20;           <probabilities>0.99 0.01</probabilities>
&#x20;       </cpt>
&#x20;       <cpt id="tub" diagtype="target">
&#x20;           <state id="no" label="F5" />
&#x20;           <state id="yes" label="F6" fault="true" />
&#x20;           <parents>asia</parents>
&#x20;           <probabilities>0.99 0.01 0.95 0.05</probabilities>
&#x20;       </cpt>
&#x20;       <cpt id="smoke" diagtype="observation" ranked="true">
&#x20;           <state id="no" />
&#x20;           <state id="yes" />
&#x20;           <probabilities>0.5 0.5</probabilities>
&#x20;       </cpt>
&#x20;       <cpt id="lung" diagtype="target">
&#x20;           <state id="no" label="F9" />
&#x20;           <state id="yes" label="F10" fault="true" />
&#x20;           <parents>smoke</parents>
&#x20;           <probabilities>0.99 0.01 0.9 0.1</probabilities>
&#x20;       </cpt>
&#x20;       <cpt id="either">
&#x20;           <state id="Nothing" />
&#x20;           <state id="CancerORTuberculosis" />
&#x20;           <parents>tub lung</parents>
&#x20;           <probabilities>1.00 0.0 1.00 0.0 1.00 0.0 0.0 1.0</probabilities>
&#x20;       </cpt>
&#x20;       <cpt id="xray" diagtype="observation" ranked="true">
&#x20;           <state id="Normal" />
&#x20;           <state id="Abnormal" />
&#x20;           <parents>either</parents>
&#x20;           <probabilities>0.95 0.05 0.02 0.98</probabilities>
&#x20;       </cpt>
&#x20;       <cpt id="bronc" diagtype="target">
&#x20;           <state id="Absent" label="F15" />
&#x20;           <state id="Present" label="F16" fault="true" />
&#x20;           <parents>smoke</parents>
&#x20;           <probabilities>0.7 0.3 0.4 0.6</probabilities>
&#x20;       </cpt>
&#x20;       <cpt id="dysp" diagtype="observation" ranked="true">
&#x20;           <state id="Absent" />
&#x20;           <state id="Present" />
&#x20;           <parents>either bronc</parents>
&#x20;           <probabilities>0.9 0.1 0.2 0.8 0.3 0.7 0.1 0.9</probabilities>
&#x20;       </cpt>
&#x20;   </nodes>
</smile>"""


TEST_WHITESPACE_MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<!-- This network was created in trial version of GeNIe, which can be used for evaluation purposes only -->
<smile version="1.0" id="Asia" numsamples="10000" discsamples="10000">
&#x20;   <nodes>
&#x20;       <cpt id="node 1" diagtype="observation" ranked="true">
&#x20;           <state id="no" />
&#x20;           <state id="yes" />
&#x20;           <probabilities>0.5 0.5</probabilities>
&#x20;       </cpt>
&#x20;       <cpt id="node 2" diagtype="target">
&#x20;           <state id="no" label="F5" />
&#x20;           <state id="yes" label="F6" fault="true" />
&#x20;           <parents>node 1</parents>
&#x20;           <probabilities>0.5 0.5 0.5 0.5</probabilities>
&#x20;       </cpt>
&#x20;   </nodes>
</smile>"""


def assert_models_equivalent(expected, got):
    assert set(expected.nodes()) == set(got.nodes())

    for node in expected.nodes():
        assert sorted(expected.get_parents(node)) == sorted(got.get_parents(node))

        cpds_expected = expected.get_cpds(node=node)
        cpds_got = got.get_cpds(node=node)

        assert cpds_expected == cpds_got


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("torch not installed")
        config.set_backend("torch")

    yield request.param

    config.set_backend("numpy")


@pytest.fixture
def reader(backend):
    return XDSLReader(string=TEST_FILE)


class TestXDSLReaderMethodsString:
    def test_whitespace_error(self, backend):
        with pytest.raises(ValueError):
            XDSLReader(string=TEST_WHITESPACE_MODEL)

    def test_get_variables(self, reader):
        var_expected = [
            "asia",
            "tub",
            "smoke",
            "lung",
            "either",
            "xray",
            "bronc",
            "dysp",
        ]

        assert reader.variables == var_expected

    def test_get_parents(self, reader):
        parents_expected = {
            "asia": [],
            "tub": ["asia"],
            "smoke": [],
            "lung": ["smoke"],
            "either": ["tub", "lung"],
            "xray": ["either"],
            "bronc": ["smoke"],
            "dysp": ["either", "bronc"],
        }

        parents = reader.variable_parents

        for variable in parents_expected:
            assert parents_expected[variable] == parents[variable]

    def test_get_states(self, reader):
        states_expected = {
            "asia": ["no", "yes"],
            "tub": ["no", "yes"],
            "smoke": ["no", "yes"],
            "lung": ["no", "yes"],
            "either": ["Nothing", "CancerORTuberculosis"],
            "xray": ["Normal", "Abnormal"],
            "bronc": ["Absent", "Present"],
            "dysp": ["Absent", "Present"],
        }

        states = reader.variable_states

        for variable in states_expected:
            assert states_expected[variable] == states[variable]

    def test_get_edges(self, reader):
        edges_expected = [
            ["asia", "tub"],
            ["smoke", "lung"],
            ["tub", "either"],
            ["lung", "either"],
            ["either", "xray"],
            ["smoke", "bronc"],
            ["either", "dysp"],
            ["bronc", "dysp"],
        ]

        assert sorted(reader.edge_list) == sorted(edges_expected)

    def test_get_values(self, reader):
        cpd_expected = {
            "asia": np.array([[0.99], [0.01]]),
            "tub": np.array([[0.99, 0.95], [0.01, 0.05]]),
            "smoke": np.array([[0.5], [0.5]]),
            "lung": np.array([[0.99, 0.9], [0.01, 0.1]]),
            "either": np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            "xray": np.array([[0.95, 0.02], [0.05, 0.98]]),
            "bronc": np.array([[0.7, 0.4], [0.3, 0.6]]),
            "dysp": np.array([[0.9, 0.2, 0.3, 0.1], [0.1, 0.8, 0.7, 0.9]]),
        }

        cpd = reader.variable_CPD

        for variable in cpd_expected:
            np_test.assert_array_equal(
                cpd_expected[variable],
                cpd[variable],
            )

    def test_model(self, reader):
        reader.read().check_model()


@pytest.fixture
def alarm_model_bn(backend):
    return load_model("bnlearn/alarm")


@pytest.fixture
def dummy_model(backend):
    model = DiscreteBayesianNetwork(
        [
            ("A", "C"),
            ("B", "C"),
            ("C", "D"),
        ]
    )

    cpd_a = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[0.92], [0.08]],
    )

    cpd_b = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.99], [0.01]],
    )

    cpd_c = TabularCPD(
        variable="C",
        variable_card=2,
        values=[
            [0.8, 0.75, 0.33, 0.99],
            [0.2, 0.25, 0.67, 0.01],
        ],
        evidence=["A", "B"],
        evidence_card=[2, 2],
    )

    cpd_d = TabularCPD(
        variable="D",
        variable_card=3,
        values=[
            [0.6, 0.4],
            [0.3, 0.4],
            [0.1, 0.2],
        ],
        evidence=["C"],
        evidence_card=[2],
    )

    model.add_cpds(
        cpd_a,
        cpd_b,
        cpd_c,
        cpd_d,
    )

    return model


@pytest.fixture
def model_with_whitespaces(backend):
    model = DiscreteBayesianNetwork()

    model.add_nodes_from(
        [
            "first node",
            "second node",
        ]
    )

    model.add_edges_from(
        [
            ("first node", "second node"),
        ]
    )

    cpd_a = TabularCPD(
        "first node",
        2,
        [[0.5], [0.5]],
    )

    cpd_b = TabularCPD(
        "second node",
        2,
        [[0.5, 0.5], [0.5, 0.5]],
        evidence=["first node"],
        evidence_card=[2],
    )

    model.add_cpds(
        cpd_a,
        cpd_b,
    )

    return model


class TestXDSLWriterMethods:
    def test_whitespace_warning(self, model_with_whitespaces):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            XDSLWriter(model_with_whitespaces)

    def test_writer_cpds(self, dummy_model, tmp_path):
        writer = XDSLWriter(dummy_model)
        filepath = tmp_path / "dummy_model.xdsl"

        writer.write(filename=str(filepath))

        with open(filepath) as f:
            reader = XDSLReader(f, state_name_type=int)

        model = reader.read()

        assert_models_equivalent(
            dummy_model,
            model,
        )

    def test_alarm_model(self, alarm_model_bn, tmp_path):
        filepath = tmp_path / "alarm_model.xdsl"

        XDSLWriter(alarm_model_bn).write(filename=str(filepath))

        with open(filepath) as f:
            file_text = f.read()

        alarm_model_bn_test = XDSLReader(string=file_text).read()

        assert_models_equivalent(
            alarm_model_bn,
            alarm_model_bn_test,
        )


class TestXDSLCommaWarning:
    def test_comma_state_name_warning(self, tmp_path):
        model = DiscreteBayesianNetwork([("A", "B")])

        cpd_a = TabularCPD(
            variable="A",
            variable_card=2,
            values=[
                [0.5],
                [0.5],
            ],
            state_names={
                "A": [
                    "state,1",
                    "state,2",
                ]
            },
        )

        cpd_b = TabularCPD(
            variable="B",
            variable_card=2,
            values=[
                [0.6, 0.4],
                [0.4, 0.6],
            ],
            evidence=["A"],
            evidence_card=[2],
            state_names={
                "B": [
                    "yes",
                    "no",
                ],
                "A": [
                    "state,1",
                    "state,2",
                ],
            },
        )

        model.add_cpds(
            cpd_a,
            cpd_b,
        )

        file_path = tmp_path / "comma_test.xdsl"

        with pytest.warns(
            UserWarning,
            match=(
                r"State name 'state,1' for variable 'A' "
                r"contains a comma"
            ),
        ):
            writer = XDSLWriter(model)
            writer.write(filename=str(file_path))

        reader = XDSLReader(str(file_path))

        loaded_model = reader.read()

        assert loaded_model.get_cpds("A").state_names["A"] == [
            "state,1",
            "state,2",
        ]

        assert loaded_model.get_cpds("B").state_names["A"] == [
            "state,1",
            "state,2",
        ]

        assert loaded_model.get_cpds("B").state_names["B"] == [
            "yes",
            "no",
        ]
