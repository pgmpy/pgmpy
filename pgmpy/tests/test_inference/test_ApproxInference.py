import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import ApproxInference, VariableElimination
from pgmpy.models import DynamicBayesianNetwork as DBN


@pytest.fixture(
    params=[
        "numpy",
        pytest.param(
            "torch",
            marks=pytest.mark.skipif(
                not _check_soft_dependencies("torch", severity="none"),
                reason="PyTorch is not installed",
            ),
        ),
    ]
)
def backend(request):
    config.set_backend(request.param)
    try:
        yield request.param
    finally:
        config.set_backend("numpy")


class TestApproxInferenceBN:
    @pytest.fixture(autouse=True)
    def setUp(self, backend):
        self.alarm_model = load_model("bnlearn/alarm")
        self.infer_alarm = ApproxInference(self.alarm_model)
        self.alarm_ve = VariableElimination(self.alarm_model)
        self.samples = self.alarm_model.simulate(int(1e4), seed=42)

    @pytest.mark.parametrize("evidence", [None, {"PVSAT": "LOW"}])
    def test_query(self, evidence):
        samples = self.samples if evidence is None else self.samples[self.samples.PVSAT == "LOW"]
        for variables, joint in [(["HISTORY"], True), (["HISTORY", "CVP"], True), (["HISTORY", "CVP"], False)]:
            ve_results = self.alarm_ve.query(variables=variables, evidence=evidence, joint=joint)
            for s, atol in ((None, 0.01), (samples, 0.02)):
                query_results = self.infer_alarm.query(
                    variables=variables, evidence=evidence, samples=s, joint=joint, seed=42
                )
                if joint:
                    assert query_results.__eq__(ve_results, atol=atol)
                else:
                    assert all(query_results[var].__eq__(ve_results[var], atol=atol) for var in variables)

    def test_virtual_evidence(self):
        virtual_evid = [TabularCPD("PAP", 3, [[0.2], [0.3], [0.5]], state_names={"PAP": ["LOW", "NORMAL", "HIGH"]})]
        for evidence in (None, {"PVSAT": "LOW"}):
            query_results = self.infer_alarm.query(
                variables=["HISTORY"], evidence=evidence, virtual_evidence=virtual_evid, seed=42
            )
            ve_results = self.alarm_ve.query(variables=["HISTORY"], evidence=evidence, virtual_evidence=virtual_evid)
            assert query_results.__eq__(ve_results, atol=0.01)


class TestApproxInferenceDBN:
    @pytest.fixture(autouse=True)
    def setUp(self, backend):
        self.model = DBN()
        self.model.add_edges_from([(("Z", 0), ("X", 0)), (("X", 0), ("Y", 0)), (("Z", 0), ("Z", 1))])
        z_start_cpd = TabularCPD(("Z", 0), 2, [[0.5], [0.5]])
        x_i_cpd = TabularCPD(("X", 0), 2, [[0.6, 0.9], [0.4, 0.1]], evidence=[("Z", 0)], evidence_card=[2])
        y_i_cpd = TabularCPD(("Y", 0), 2, [[0.2, 0.3], [0.8, 0.7]], evidence=[("X", 0)], evidence_card=[2])
        z_trans_cpd = TabularCPD(("Z", 1), 2, [[0.4, 0.7], [0.6, 0.3]], evidence=[("Z", 0)], evidence_card=[2])
        self.model.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        self.model.initialize_initial_state()
        self.infer = ApproxInference(self.model)

    def test_query(self):
        cases = [
            ([("Y", 1)], {}, [0.2259, 0.7741]),
            ([("Y", 0), ("Y", 1)], {}, [0.0510, 0.1763, 0.1698, 0.6029]),
            ([("Y", 1), ("Y", 5)], {}, [0.0476, 0.1732, 0.1762, 0.6030]),
            ([("Y", 4)], {"evidence": {("Y", 2): 0}}, [0.2232, 0.7768]),
            ([("Y", 4)], {"virtual_evidence": [TabularCPD(("Y", 2), 2, [[0.2], [0.8]])]}, [0.2205, 0.7795]),
        ]
        for variables, kwargs, values in cases:
            result = self.infer.query(variables, seed=42, **kwargs)
            assert result.__eq__(DiscreteFactor(variables, [2] * len(variables), values), atol=0.01)

        # Evidence in a later time slice than the query variable.
        assert self.infer.query([("Y", 0)], evidence={("Y", 1): 0}) is not None
        assert self.infer.query([("Y", 0)], virtual_evidence=[TabularCPD(("Y", 1), 2, [[0.2], [0.8]])]) is not None
