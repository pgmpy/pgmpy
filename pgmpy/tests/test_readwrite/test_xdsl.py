import os
import tempfile
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
    <nodes>
        <cpt id="asia" diagtype="observation" ranked="true">
            <state id="no" />
            <state id="yes" />
            <probabilities>0.99 0.01</probabilities>
        </cpt>
        <cpt id="tub" diagtype="target">
            <state id="no" label="F5" />
            <state id="yes" label="F6" fault="true" />
            <parents>asia</parents>
            <probabilities>0.99 0.01 0.95 0.05</probabilities>
        </cpt>
        <cpt id="smoke" diagtype="observation" ranked="true">
            <state id="no" />
            <state id="yes" />
            <probabilities>0.5 0.5</probabilities>
        </cpt>
        <cpt id="lung" diagtype="target">
            <state id="no" label="F9" />
            <state id="yes" label="F10" fault="true" />
            <parents>smoke</parents>
            <probabilities>0.99 0.01 0.9 0.1</probabilities>
        </cpt>
        <cpt id="either">
            <state id="Nothing" />
            <state id="CancerORTuberculosis" />
            <parents>tub lung</parents>
            <probabilities>1.00 0.0 1.00 0.0 1.00 0.0 0.0 1.0</probabilities>
        </cpt>
        <cpt id="xray" diagtype="observation" ranked="true">
            <state id="Normal" />
            <state id="Abnormal" />
            <parents>either</parents>
            <probabilities>0.95 0.05 0.02 0.98</probabilities>
        </cpt>
        <cpt id="bronc" diagtype="target">
            <state id="Absent" label="F15" />
            <state id="Present" label="F16" fault="true" />
            <parents>smoke</parents>
            <probabilities>0.7 0.3 0.4 0.6</probabilities>
        </cpt>
        <cpt id="dysp" diagtype="observation" ranked="true">
            <state id="Absent" />
            <state id="Present" />
            <parents>either bronc</parents>
            <probabilities>0.9 0.1 0.2 0.8 0.3 0.7 0.1 0.9</probabilities>
        </cpt>
    </nodes>
</smile>"""

TEST_WHITESPACE_MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<!-- This network was created in trial version of GeNIe, which can be used for evaluation purposes only -->
<smile version="1.0" id="Asia" numsamples="10000" discsamples="10000">
    <nodes>
        <cpt id="node 1" diagtype="observation" ranked="true">
            <state id="no" />
            <state id="yes" />
            <probabilities>0.5 0.5</probabilities>
        </cpt>
        <cpt id="node 2" diagtype="target">
            <state id="no" label="F5" />
            <state id="yes" label="F6" fault="true" />
            <parents>node 1</parents>
            <probabilities>0.5 0.5 0.5 0.5</probabilities>
        </cpt>
    </nodes>
</smile>"""


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
            np_test.assert_array_equal(cpd_expected[variable], cpd[variable])

    def test_model(self, reader):
        reader.read().check_model()


DUMMY_FILE = """<?xml version="1.0" encoding="UTF-8"?>
<smile version="1.0" id="dummy" numsamples="10000" discsamples="10000">
    <nodes>
        <cpt id="A" >
            <state id="yes" />
            <state id="no" />
            <probabilities>0.92 0.08</probabilities>
        </cpt>
        <cpt id="B" >
            <state id="high" />
            <state id="low" />
            <probabilities>0.99 0.01</probabilities>
        </cpt>
        <cpt id="C" >
            <state id="true" />
            <state id="false" />
            <parents>A B</parents>
            <probabilities>0.8 0.2 0.75 0.25 0.33 0.67 0.99 0.01</probabilities>
        </cpt>
        <cpt id="D" >
            <state id="big" />
            <state id="medium" />
            <state id="small" />
            <parents>C</parents>
            <probabilities>0.6 0.3 0.1 0.4 0.4 0.2</probabilities>
        </cpt>
    </nodes>
</smile>"""


@pytest.fixture
def alarm_model_bn(backend):
    return load_model("bnlearn/alarm")


@pytest.fixture
def dummy_model(backend):
    dummy_model = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("C", "D")])
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.92], [0.08]])
    cpd_b = TabularCPD(variable="B", variable_card=2, values=[[0.99], [0.01]])

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
        values=[[0.6, 0.4], [0.3, 0.4], [0.1, 0.2]],
        evidence=["C"],
        evidence_card=[2],
    )

    dummy_model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)  # testing without state names
    return dummy_model


@pytest.fixture
def model_with_whitespaces(backend):
    model_with_whitespaces = DiscreteBayesianNetwork()
    model_with_whitespaces.add_nodes_from(["first node", "second node"])
    model_with_whitespaces.add_edges_from([("first node", "second node")])
    cpd_a = TabularCPD("first node", 2, [[0.5], [0.5]])
    cpd_b = TabularCPD(
        "second node",
        2,
        [[0.5, 0.5], [0.5, 0.5]],
        evidence=["first node"],
        evidence_card=[2],
    )
    model_with_whitespaces.add_cpds(cpd_a, cpd_b)
    return model_with_whitespaces


def assert_models_equivalent(expected, got):
    assert set(expected.nodes()) == set(got.nodes())
    for node in expected.nodes():
        assert sorted(expected.get_parents(node)) == sorted(got.get_parents(node))
        cpds_expected = expected.get_cpds(node=node)
        cpds_got = got.get_cpds(node=node)
        assert cpds_expected == cpds_got


class TestXDSLWriterMethods:
    @pytest.fixture
    def writer_dummy(self, dummy_model):
        return XDSLWriter(dummy_model)

    def test_whitespace_warning(self, model_with_whitespaces):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            XDSLWriter(model_with_whitespaces)

    def test_writer_cpds(self, writer_dummy, dummy_model):
        writer_dummy.write(filename="dummy_model.xdsl")
        with open("dummy_model.xdsl") as f:
            reader = XDSLReader(f, state_name_type=int)
        model = reader.read()
        assert_models_equivalent(dummy_model, model)
        os.remove("dummy_model.xdsl")

    def test_alarm_model(self, alarm_model_bn):
        XDSLWriter(alarm_model_bn).write("alarm_model.xdsl")

        with open("alarm_model.xdsl") as f:
            file_text = f.read()
        alarm_model_bn_test = XDSLReader(string=file_text).read()
        assert_models_equivalent(alarm_model_bn, alarm_model_bn_test)

        os.remove("alarm_model.xdsl")


class TestXDSLCommaWarning:
    def test_comma_state_name_warning(self):
        # Create a model with state names containing commas
        model = DiscreteBayesianNetwork([("A", "B")])
        cpd_a = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"A": ["state,1", "state,2"]},
        )
        cpd_b = TabularCPD(
            variable="B",
            variable_card=2,
            values=[[0.6, 0.4], [0.4, 0.6]],
            evidence=["A"],
            evidence_card=[2],
            state_names={"B": ["yes", "no"], "A": ["state,1", "state,2"]},
        )
        model.add_cpds(cpd_a, cpd_b)

        # Test that warning is raised when writing
        with tempfile.NamedTemporaryFile(suffix=".xdsl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.warns(UserWarning, match="State name 'state,1' for variable 'A' contains a comma"):
                writer = XDSLWriter(model)
                writer.write(tmp_path)

            # Verify that the file can be loaded back with the same state names
            reader = XDSLReader(tmp_path)
            loaded_model = reader.read()

            # Check that the state names were preserved
            assert loaded_model.get_cpds("A").state_names["A"] == ["state,1", "state,2"]
            assert loaded_model.get_cpds("B").state_names["A"] == ["state,1", "state,2"]
            assert loaded_model.get_cpds("B").state_names["B"] == ["yes", "no"]
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
