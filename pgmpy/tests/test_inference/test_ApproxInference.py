import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import ApproxInference, VariableElimination
from pgmpy.models import DynamicBayesianNetwork as DBN


@pytest.fixture
def alarm_setup():
    alarm_model = load_model("bnlearn/alarm")
    infer_alarm = ApproxInference(alarm_model)
    alarm_ve = VariableElimination(alarm_model)
    samples = alarm_model.simulate(int(1e4))
    return infer_alarm, alarm_ve, samples


@pytest.fixture
def dbn_setup():
    model = DBN()
    model.add_edges_from([(("Z", 0), ("X", 0)), (("X", 0), ("Y", 0)), (("Z", 0), ("Z", 1))])
    z_start_cpd = TabularCPD(("Z", 0), 2, [[0.5], [0.5]])
    x_i_cpd = TabularCPD(
        ("X", 0),
        2,
        [[0.6, 0.9], [0.4, 0.1]],
        evidence=[("Z", 0)],
        evidence_card=[2],
    )
    y_i_cpd = TabularCPD(
        ("Y", 0),
        2,
        [[0.2, 0.3], [0.8, 0.7]],
        evidence=[("X", 0)],
        evidence_card=[2],
    )
    z_trans_cpd = TabularCPD(
        ("Z", 1),
        2,
        [[0.4, 0.7], [0.6, 0.3]],
        evidence=[("Z", 0)],
        evidence_card=[2],
    )
    model.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
    model.initialize_initial_state()
    infer = ApproxInference(model)
    return infer


@pytest.fixture
def dbn_torch_setup():
    config.set_backend("torch")
    model = DBN()
    model.add_edges_from([(("Z", 0), ("X", 0)), (("X", 0), ("Y", 0)), (("Z", 0), ("Z", 1))])
    z_start_cpd = TabularCPD(("Z", 0), 2, [[0.5], [0.5]])
    x_i_cpd = TabularCPD(
        ("X", 0),
        2,
        [[0.6, 0.9], [0.4, 0.1]],
        evidence=[("Z", 0)],
        evidence_card=[2],
    )
    y_i_cpd = TabularCPD(
        ("Y", 0),
        2,
        [[0.2, 0.3], [0.8, 0.7]],
        evidence=[("X", 0)],
        evidence_card=[2],
    )
    z_trans_cpd = TabularCPD(
        ("Z", 1),
        2,
        [[0.4, 0.7], [0.6, 0.3]],
        evidence=[("Z", 0)],
        evidence_card=[2],
    )
    model.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
    model.initialize_initial_state()
    infer = ApproxInference(model)
    yield infer
    # teardown
    config.set_backend("numpy")


class TestApproxInferenceBN:
    def test_query_marg(self, alarm_setup):
        infer_alarm, alarm_ve, samples = alarm_setup
        query_results = infer_alarm.query(variables=["HISTORY"])
        ve_results = alarm_ve.query(variables=["HISTORY"])
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY"], samples=samples)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], joint=True)
        ve_results = alarm_ve.query(variables=["HISTORY", "CVP"], joint=True)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], samples=samples, joint=True)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], joint=False)
        ve_results = alarm_ve.query(variables=["HISTORY", "CVP"], joint=False)
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], samples=samples, joint=False)
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

    def test_query_evidence(self, alarm_setup):
        infer_alarm, alarm_ve, samples = alarm_setup
        query_results = infer_alarm.query(variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True)
        ve_results = alarm_ve.query(variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            samples=samples[samples.PVSAT == "LOW"],
            joint=True,
            seed=42,
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True)
        ve_results = alarm_ve.query(variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=samples[samples.PVSAT == "LOW"],
            joint=True,
            seed=42,
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False)
        ve_results = alarm_ve.query(variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False)
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=samples[samples.PVSAT == "LOW"],
            joint=False,
            seed=42,
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

    def test_virtual_evidence(self, alarm_setup):
        infer_alarm, alarm_ve, samples = alarm_setup
        virtual_evid = TabularCPD(
            "PAP",
            3,
            [[0.2], [0.3], [0.5]],
            state_names={"PAP": ["LOW", "NORMAL", "HIGH"]},
        )
        query_results = infer_alarm.query(variables=["HISTORY"], virtual_evidence=[virtual_evid])
        ve_results = alarm_ve.query(variables=["HISTORY"], virtual_evidence=[virtual_evid])
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        ve_results = alarm_ve.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        assert query_results.__eq__(ve_results, atol=0.01)


class TestApproxInferenceDBN:
    def test_inference(self, dbn_setup):
        infer = dbn_setup
        res1 = infer.query([("Y", 1)], seed=42)
        expected1 = DiscreteFactor([("Y", 1)], [2], [0.2259, 0.7741])
        assert res1.__eq__(expected1, atol=0.01)
        res2 = infer.query([("Y", 0), ("Y", 1)], seed=42)
        expected2 = DiscreteFactor([("Y", 0), ("Y", 1)], [2, 2], [0.0510, 0.1763, 0.1698, 0.6029])
        assert res2.__eq__(expected2, atol=0.01)
        res3 = infer.query([("Y", 1), ("Y", 5)], seed=42)
        expected3 = DiscreteFactor([("Y", 1), ("Y", 5)], [2, 2], [0.0476, 0.1732, 0.1762, 0.6030])
        assert res3.__eq__(expected3, atol=0.01)

    def test_evidence(self, dbn_setup):
        infer = dbn_setup
        res1 = infer.query([("Y", 4)], evidence={("Y", 2): 0})
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2232, 0.7768])
        assert res1.__eq__(expected1, atol=0.01)

        # Case where evidence has higher time slice than query variable (covers line 176)
        res2 = infer.query([("Y", 0)], evidence={("Y", 1): 0})
        assert res2 is not None

    def test_virtual_evidence(self, dbn_setup):
        infer = dbn_setup
        res1 = infer.query([("Y", 4)], virtual_evidence=[TabularCPD(("Y", 2), 2, [[0.2], [0.8]])])
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2205, 0.7795])
        assert res1.__eq__(expected1, atol=0.01)

        # Case where virtual evidence has higher time slice than query variable (covers line 179)
        res2 = infer.query([("Y", 0)], virtual_evidence=[TabularCPD(("Y", 1), 2, [[0.2], [0.8]])])
        assert res2 is not None


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestApproxInferenceBNTorch:
    def test_query_marg(self, alarm_setup):
        infer_alarm, alarm_ve, samples = alarm_setup
        query_results = infer_alarm.query(variables=["HISTORY"])
        ve_results = alarm_ve.query(variables=["HISTORY"])
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY"], samples=samples)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], joint=True)
        ve_results = alarm_ve.query(variables=["HISTORY", "CVP"], joint=True)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], samples=samples, joint=True)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], joint=False)
        ve_results = alarm_ve.query(variables=["HISTORY", "CVP"], joint=False)
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], samples=samples, joint=False)
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

    def test_query_evidence(self, alarm_setup):
        infer_alarm, alarm_ve, samples = alarm_setup
        query_results = infer_alarm.query(variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True, seed=42)
        ve_results = alarm_ve.query(variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            samples=samples[samples.PVSAT == "LOW"],
            joint=True,
            seed=42,
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True, seed=42)
        ve_results = alarm_ve.query(variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=samples[samples.PVSAT == "LOW"],
            joint=True,
            seed=42,
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False, seed=42)
        ve_results = alarm_ve.query(variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False)
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=samples[samples.PVSAT == "LOW"],
            joint=False,
            seed=42,
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

    def test_virtual_evidence(self, alarm_setup):
        infer_alarm, alarm_ve, _ = alarm_setup
        virtual_evid = TabularCPD(
            "PAP",
            3,
            [[0.2], [0.3], [0.5]],
            state_names={"PAP": ["LOW", "NORMAL", "HIGH"]},
        )
        query_results = infer_alarm.query(variables=["HISTORY"], virtual_evidence=[virtual_evid])
        ve_results = alarm_ve.query(variables=["HISTORY"], virtual_evidence=[virtual_evid])
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        ve_results = alarm_ve.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        assert query_results.__eq__(ve_results, atol=0.01)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestApproxInferenceDBNTorch:
    def test_inference(self, dbn_torch_setup):
        infer = dbn_torch_setup
        res1 = infer.query([("Y", 1)], seed=42)
        expected1 = DiscreteFactor([("Y", 1)], [2], [0.2259, 0.7741])
        assert res1.__eq__(expected1, atol=0.01)
        res2 = infer.query([("Y", 0), ("Y", 1)], seed=42)
        expected2 = DiscreteFactor([("Y", 0), ("Y", 1)], [2, 2], [0.0510, 0.1763, 0.1698, 0.6029])
        assert res2.__eq__(expected2, atol=0.01)
        res3 = infer.query([("Y", 1), ("Y", 5)], seed=42)
        expected3 = DiscreteFactor([("Y", 1), ("Y", 5)], [2, 2], [0.0476, 0.1732, 0.1762, 0.6030])
        assert res3.__eq__(expected3, atol=0.01)

    def test_evidence(self, dbn_torch_setup):
        infer = dbn_torch_setup
        res1 = infer.query([("Y", 4)], evidence={("Y", 2): 0})
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2232, 0.7768])
        assert res1.__eq__(expected1, atol=0.01)

        # Case where evidence has higher time slice than query variable (covers line 176)
        res2 = infer.query([("Y", 0)], evidence={("Y", 1): 0})
        assert res2 is not None

    def test_virtual_evidence(self, dbn_torch_setup):
        infer = dbn_torch_setup
        res1 = infer.query([("Y", 4)], virtual_evidence=[TabularCPD(("Y", 2), 2, [[0.2], [0.8]])])
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2205, 0.7795])
        assert res1.__eq__(expected1, atol=0.01)

        # Case where virtual evidence has higher time slice than query variable (covers line 179)
        res2 = infer.query([("Y", 0)], virtual_evidence=[TabularCPD(("Y", 1), 2, [[0.2], [0.8]])])
        assert res2 is not None
