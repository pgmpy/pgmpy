import os
import tempfile

import numpy as np
import numpy.testing as np_test
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter

TEST_FILE = """<?xml version="1.0"?>


<!--
    Bayesian network in XMLBIF v0.3 (BayesNet Interchange Format)
    Produced by JavaBayes (http://www.cs.cmu.edu/~javabayes/
    Output created Mon Aug 01 10:33:28 AEST 2016
-->



<!-- DTD for the XMLBIF 0.3 format -->
<!DOCTYPE BIF [
    <!ELEMENT BIF ( NETWORK )*>
          <!ATTLIST BIF VERSION CDATA #REQUIRED>
    <!ELEMENT NETWORK ( NAME, ( PROPERTY | VARIABLE | DEFINITION )* )>
    <!ELEMENT NAME (#PCDATA)>
    <!ELEMENT VARIABLE ( NAME, ( OUTCOME |  PROPERTY )* ) >
          <!ATTLIST VARIABLE TYPE (nature|decision|utility) "nature">
    <!ELEMENT OUTCOME (#PCDATA)>
    <!ELEMENT DEFINITION ( FOR | GIVEN | TABLE | PROPERTY )* >
    <!ELEMENT FOR (#PCDATA)>
    <!ELEMENT GIVEN (#PCDATA)>
    <!ELEMENT TABLE (#PCDATA)>
    <!ELEMENT PROPERTY (#PCDATA)>
]>


<BIF VERSION="0.3">
<NETWORK>
<NAME>Dog_Problem</NAME>

<!-- Variables -->
<VARIABLE TYPE="nature">
    <NAME>kid</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (100, 165)</PROPERTY>
</VARIABLE>

<VARIABLE TYPE="nature">
    <NAME>light_on</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (73, 165)</PROPERTY>
</VARIABLE>

<VARIABLE TYPE="nature">
    <NAME>bowel_problem</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (190, 69)</PROPERTY>
</VARIABLE>

<VARIABLE TYPE="nature">
    <NAME>dog_out</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (155, 165)</PROPERTY>
</VARIABLE>

<VARIABLE TYPE="nature">
    <NAME>hear_bark</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (154, 241)</PROPERTY>
</VARIABLE>

<VARIABLE TYPE="nature">
    <NAME>family_out</NAME>
    <OUTCOME>true</OUTCOME>
    <OUTCOME>false</OUTCOME>
    <PROPERTY>position = (112, 69)</PROPERTY>
</VARIABLE>


<!-- Probability distributions -->
<DEFINITION>
    <FOR>kid</FOR>
    <TABLE>0.3 0.7 </TABLE>
</DEFINITION>

<DEFINITION>
    <FOR>light_on</FOR>
    <GIVEN>family_out</GIVEN>
    <TABLE>0.6 0.4 0.05 0.95 </TABLE>
</DEFINITION>

<DEFINITION>
    <FOR>bowel_problem</FOR>
    <TABLE>0.01 0.99 </TABLE>
</DEFINITION>

<DEFINITION>
    <FOR>dog_out</FOR>
    <GIVEN>bowel_problem</GIVEN>
    <GIVEN>family_out</GIVEN>
    <TABLE>0.99 0.01 0.97 0.03 0.9 0.1 0.3 0.7 </TABLE>
</DEFINITION>

<DEFINITION>
    <FOR>hear_bark</FOR>
    <GIVEN>dog_out</GIVEN>
    <TABLE>0.7 0.3 0.01 0.99 </TABLE>
</DEFINITION>

<DEFINITION>
    <FOR>family_out</FOR>
    <TABLE>0.15 0.85 </TABLE>
</DEFINITION>


</NETWORK>
</BIF>"""


@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    if request.param == "torch":
        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("torch not installed")
        config.set_backend("torch")
    yield request.param
    config.set_backend("numpy")


class TestXMLBIFReaderMethods:
    @pytest.fixture
    def reader(self, backend):
        return XMLBIFReader(string=TEST_FILE)

    def test_get_variables(self, reader):
        var_expected = [
            "kid",
            "light_on",
            "bowel_problem",
            "dog_out",
            "hear_bark",
            "family_out",
        ]
        assert reader.variables == var_expected

    def test_get_states(self, reader):
        states_expected = {
            "bowel_problem": ["true", "false"],
            "dog_out": ["true", "false"],
            "family_out": ["true", "false"],
            "hear_bark": ["true", "false"],
            "kid": ["true", "false"],
            "light_on": ["true", "false"],
        }
        states = reader.variable_states
        for variable in states_expected:
            assert states_expected[variable] == states[variable]

    def test_get_parents(self, reader):
        parents_expected = {
            "bowel_problem": [],
            "dog_out": ["bowel_problem", "family_out"],
            "family_out": [],
            "hear_bark": ["dog_out"],
            "kid": [],
            "light_on": ["family_out"],
        }
        parents = reader.variable_parents
        for variable in parents_expected:
            assert parents_expected[variable] == parents[variable]

    def test_get_edges(self, reader):
        edges_expected = [
            ["family_out", "dog_out"],
            ["bowel_problem", "dog_out"],
            ["family_out", "light_on"],
            ["dog_out", "hear_bark"],
        ]
        assert sorted(reader.edge_list) == sorted(edges_expected)

    def test_get_values(self, reader):
        cpd_expected = {
            "bowel_problem": np.array([[0.01], [0.99]]),
            "dog_out": np.array([[0.99, 0.97, 0.9, 0.3], [0.01, 0.03, 0.1, 0.7]]),
            "family_out": np.array([[0.15], [0.85]]),
            "hear_bark": np.array([[0.7, 0.01], [0.3, 0.99]]),
            "kid": np.array([[0.3], [0.7]]),
            "light_on": np.array([[0.6, 0.05], [0.4, 0.95]]),
        }
        cpd = reader.variable_CPD
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable], cpd[variable])

    def test_get_property(self, reader):
        property_expected = {
            "bowel_problem": ["position = (190, 69)"],
            "dog_out": ["position = (155, 165)"],
            "family_out": ["position = (112, 69)"],
            "hear_bark": ["position = (154, 241)"],
            "kid": ["position = (100, 165)"],
            "light_on": ["position = (73, 165)"],
        }
        prop = reader.variable_property
        for variable in property_expected:
            assert property_expected[variable] == prop[variable]

    def test_model(self, reader):
        reader.read().check_model()

    def test_make_valid_state_name(self):
        model = DiscreteBayesianNetwork()
        writer = XMLBIFWriter(model)

        valid_state = "valid_state"
        assert writer._make_valid_state_name(valid_state) == valid_state

        invalid_state = "invalid-state@123"
        expected_fixed = "invalid_state_123"
        with pytest.warns(
            UserWarning, match=f"State name '{invalid_state}' for variable 'unknown' was changed to '{expected_fixed}'"
        ):
            result = writer._make_valid_state_name(invalid_state)

        assert result == expected_fixed


class TestXMLBIFReaderMethodsFile:
    @pytest.fixture
    def reader(self, backend):
        with open("dog_problem.xml", "w") as fout:
            fout.write(TEST_FILE)
        yield XMLBIFReader("dog_problem.xml")
        os.remove("dog_problem.xml")

    def test_get_variables(self, reader):
        var_expected = [
            "kid",
            "light_on",
            "bowel_problem",
            "dog_out",
            "hear_bark",
            "family_out",
        ]
        assert reader.variables == var_expected

    def test_get_states(self, reader):
        states_expected = {
            "bowel_problem": ["true", "false"],
            "dog_out": ["true", "false"],
            "family_out": ["true", "false"],
            "hear_bark": ["true", "false"],
            "kid": ["true", "false"],
            "light_on": ["true", "false"],
        }
        states = reader.variable_states
        for variable in states_expected:
            assert states_expected[variable] == states[variable]

    def test_get_parents(self, reader):
        parents_expected = {
            "bowel_problem": [],
            "dog_out": ["bowel_problem", "family_out"],
            "family_out": [],
            "hear_bark": ["dog_out"],
            "kid": [],
            "light_on": ["family_out"],
        }
        parents = reader.variable_parents
        for variable in parents_expected:
            assert parents_expected[variable] == parents[variable]

    def test_get_edges(self, reader):
        edges_expected = [
            ["family_out", "dog_out"],
            ["bowel_problem", "dog_out"],
            ["family_out", "light_on"],
            ["dog_out", "hear_bark"],
        ]
        assert sorted(reader.edge_list) == sorted(edges_expected)

    def test_get_values(self, reader):
        cpd_expected = {
            "bowel_problem": np.array([[0.01], [0.99]]),
            "dog_out": np.array([[0.99, 0.97, 0.9, 0.3], [0.01, 0.03, 0.1, 0.7]]),
            "family_out": np.array([[0.15], [0.85]]),
            "hear_bark": np.array([[0.7, 0.01], [0.3, 0.99]]),
            "kid": np.array([[0.3], [0.7]]),
            "light_on": np.array([[0.6, 0.05], [0.4, 0.95]]),
        }
        cpd = reader.variable_CPD
        for variable in cpd_expected:
            np_test.assert_array_equal(cpd_expected[variable], cpd[variable])

    def test_get_property(self, reader):
        property_expected = {
            "bowel_problem": ["position = (190, 69)"],
            "dog_out": ["position = (155, 165)"],
            "family_out": ["position = (112, 69)"],
            "hear_bark": ["position = (154, 241)"],
            "kid": ["position = (100, 165)"],
            "light_on": ["position = (73, 165)"],
        }
        prop = reader.variable_property
        for variable in property_expected:
            assert property_expected[variable] == prop[variable]

    def test_model(self, reader):
        reader.read().check_model()


class TestXMLBIFWriterMethodsString:
    @pytest.fixture
    def expected_model(self, backend):
        reader = XMLBIFReader(string=TEST_FILE)
        return reader.read()

    @pytest.fixture
    def writer(self, expected_model):
        return XMLBIFWriter(expected_model)

    @pytest.fixture
    def model_stateless(self, backend):
        model_stateless = DiscreteBayesianNetwork([("D", "G"), ("I", "G"), ("G", "L"), ("I", "S")])
        cpd_d = TabularCPD(variable="D", variable_card=2, values=[[0.6], [0.4]])
        cpd_i = TabularCPD(variable="I", variable_card=2, values=[[0.7], [0.3]])

        cpd_g = TabularCPD(
            variable="G",
            variable_card=3,
            values=[
                [0.3, 0.05, 0.9, 0.5],
                [0.4, 0.25, 0.08, 0.3],
                [0.3, 0.7, 0.02, 0.2],
            ],
            evidence=["I", "D"],
            evidence_card=[2, 2],
        )

        cpd_l = TabularCPD(
            variable="L",
            variable_card=2,
            values=[[0.1, 0.4, 0.99], [0.9, 0.6, 0.01]],
            evidence=["G"],
            evidence_card=[3],
        )

        cpd_s = TabularCPD(
            variable="S",
            variable_card=2,
            values=[[0.95, 0.2], [0.05, 0.8]],
            evidence=["I"],
            evidence_card=[2],
        )

        model_stateless.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
        return model_stateless

    @pytest.fixture
    def writer_stateless(self, model_stateless):
        return XMLBIFWriter(model_stateless)

    def test_write_xmlbif_statefull(self, writer, expected_model):
        writer.write("dog_problem_output.xbif")
        with open("dog_problem_output.xbif") as f:
            file_text = f.read()
        reader = XMLBIFReader(string=file_text, state_name_type=str)
        model = reader.read()
        self.assert_models_equivelent(expected_model, model)
        os.remove("dog_problem_output.xbif")

    def test_write_xmlbif_stateless(self, writer_stateless, model_stateless):
        writer_stateless.write("grade_problem_output.xbif")
        with open("grade_problem_output.xbif") as f:
            reader = XMLBIFReader(f, state_name_type=int)
        model = reader.read()
        self.assert_models_equivelent(model_stateless, model)
        assert {"D": [0, 1]} == model.get_cpds("D").state_names
        os.remove("grade_problem_output.xbif")

    def assert_models_equivelent(self, expected, got):
        assert set(expected.nodes()) == set(got.nodes())
        for node in expected.nodes():
            assert sorted(expected.get_parents(node)) == sorted(got.get_parents(node))
            cpds_expected = expected.get_cpds(node=node)
            cpds_got = got.get_cpds(node=node)
            assert cpds_expected == cpds_got

    def test_comma_state_name_warning(self, backend):
        # Create a simple model with state names containing commas
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
        with tempfile.NamedTemporaryFile(suffix=".xmlbif", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.warns(UserWarning, match="State name 'state,1' for variable 'A' was changed to 'state_1'"):
                writer = XMLBIFWriter(model)
                writer.write(tmp_path)

            # The file should still be loadable but with modified state names
            reader = XMLBIFReader(tmp_path)
            loaded_model = reader.read()

            # Check that the state names were modified to be valid XMLBIF identifiers
            # Commas should be replaced with underscores, but no leading underscore needed
            assert loaded_model.get_cpds("A").state_names["A"] == ["state_1", "state_2"]
            assert loaded_model.get_cpds("B").state_names["A"] == ["state_1", "state_2"]
            assert loaded_model.get_cpds("B").state_names["B"] == ["yes", "no"]
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
