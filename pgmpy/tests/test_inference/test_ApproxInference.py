import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy import config
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import ApproxInference, VariableElimination
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.utils import get_example_model


@pytest.fixture
def torch_backend():
    config.set_backend("torch")
    yield
    config.set_backend("numpy")


@pytest.fixture
def alarm_model():
    return get_example_model("alarm")


@pytest.fixture
def alarm_model_torch(torch_backend):
    return get_example_model("alarm")


@pytest.fixture
def infer_alarm(alarm_model):
    return ApproxInference(alarm_model)


@pytest.fixture
def infer_alarm_torch(alarm_model_torch):
    return ApproxInference(alarm_model_torch)


@pytest.fixture
def alarm_variable_elimination(alarm_model):
    return VariableElimination(alarm_model)


@pytest.fixture
def alarm_variable_elimination_torch(alarm_model_torch):
    return VariableElimination(alarm_model_torch)


@pytest.fixture
def alarm_samples(alarm_model):
    samples = alarm_model.simulate(int(1e4))
    return samples


@pytest.fixture
def alarm_samples_torch(alarm_model_torch):
    samples = alarm_model_torch.simulate(int(1e4))
    return samples


def _build_approx_inference_dbn_inference():
    model = DBN()
    model.add_edges_from(
        [(("Z", 0), ("X", 0)), (("X", 0), ("Y", 0)), (("Z", 0), ("Z", 1))]
    )
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
def approx_inference_dbn_inference():
    return _build_approx_inference_dbn_inference()


@pytest.fixture
def approx_inference_dbn_inference_torch(torch_backend):
    return _build_approx_inference_dbn_inference()


class TestApproxInferenceBN:

    def test_query_marg(self, infer_alarm, alarm_variable_elimination, alarm_samples):
        query_results = infer_alarm.query(variables=["HISTORY"])
        ve_results = alarm_variable_elimination.query(variables=["HISTORY"])
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY"], samples=alarm_samples)
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], joint=True)
        ve_results = alarm_variable_elimination.query(
            variables=["HISTORY", "CVP"], joint=True
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"], samples=alarm_samples, joint=True
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(variables=["HISTORY", "CVP"], joint=False)
        ve_results = alarm_variable_elimination.query(
            variables=["HISTORY", "CVP"], joint=False
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"], samples=alarm_samples, joint=False
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

    def test_query_evidence(
        self, infer_alarm, alarm_variable_elimination, alarm_samples
    ):
        query_results = infer_alarm.query(
            variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True
        )
        ve_results = alarm_variable_elimination.query(
            variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            samples=alarm_samples[alarm_samples.PVSAT == "LOW"],
            joint=True,
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
        )
        ve_results = alarm_variable_elimination.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=alarm_samples[alarm_samples.PVSAT == "LOW"],
            joint=True,
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False
        )
        ve_results = alarm_variable_elimination.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=alarm_samples[alarm_samples.PVSAT == "LOW"],
            joint=False,
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

    def test_virtual_evidence(self, infer_alarm, alarm_variable_elimination):
        virtual_evid = TabularCPD(
            "PAP",
            3,
            [[0.2], [0.3], [0.5]],
            state_names={"PAP": ["LOW", "NORMAL", "HIGH"]},
        )
        query_results = infer_alarm.query(
            variables=["HISTORY"], virtual_evidence=[virtual_evid]
        )
        ve_results = alarm_variable_elimination.query(
            variables=["HISTORY"], virtual_evidence=[virtual_evid]
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        ve_results = alarm_variable_elimination.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        assert query_results.__eq__(ve_results, atol=0.01)


class TestApproxInferenceDBN:
    def test_inference(self, approx_inference_dbn_inference):
        res1 = approx_inference_dbn_inference.query([("Y", 1)], seed=42)
        expected1 = DiscreteFactor([("Y", 1)], [2], [0.2259, 0.7741])
        assert res1.__eq__(expected1, atol=0.01)
        res2 = approx_inference_dbn_inference.query([("Y", 0), ("Y", 1)], seed=42)
        expected2 = DiscreteFactor(
            [("Y", 0), ("Y", 1)], [2, 2], [0.0510, 0.1763, 0.1698, 0.6029]
        )
        assert res2.__eq__(expected2, atol=0.01)
        res3 = approx_inference_dbn_inference.query([("Y", 1), ("Y", 5)], seed=42)
        expected3 = DiscreteFactor(
            [("Y", 1), ("Y", 5)], [2, 2], [0.0476, 0.1732, 0.1762, 0.6030]
        )
        assert res3.__eq__(expected3, atol=0.01)

    def test_evidence(self, approx_inference_dbn_inference):
        res1 = approx_inference_dbn_inference.query([("Y", 4)], evidence={("Y", 2): 0})
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2232, 0.7768])
        assert res1.__eq__(expected1, atol=0.01)

    def test_virtual_evidence(self, approx_inference_dbn_inference):
        res1 = approx_inference_dbn_inference.query(
            [("Y", 4)], virtual_evidence=[TabularCPD(("Y", 2), 2, [[0.2], [0.8]])]
        )
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2205, 0.7795])
        assert res1.__eq__(expected1, atol=0.01)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestApproxInferenceBNTorch:
    def test_query_marg(
        self, infer_alarm_torch, alarm_variable_elimination_torch, alarm_samples_torch
    ):
        query_results = infer_alarm_torch.query(variables=["HISTORY"])
        ve_results = alarm_variable_elimination_torch.query(variables=["HISTORY"])
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY"], samples=alarm_samples_torch
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY", "CVP"], joint=True
        )
        ve_results = alarm_variable_elimination_torch.query(
            variables=["HISTORY", "CVP"], joint=True
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY", "CVP"], samples=alarm_samples_torch, joint=True
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY", "CVP"], joint=False
        )
        ve_results = alarm_variable_elimination_torch.query(
            variables=["HISTORY", "CVP"], joint=False
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY", "CVP"], samples=alarm_samples_torch, joint=False
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

    def test_query_evidence(
        self, infer_alarm_torch, alarm_variable_elimination_torch, alarm_samples_torch
    ):
        query_results = infer_alarm_torch.query(
            variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True
        )
        ve_results = alarm_variable_elimination_torch.query(
            variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            samples=alarm_samples_torch[alarm_samples_torch.PVSAT == "LOW"],
            joint=True,
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
        )
        ve_results = alarm_variable_elimination_torch.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=alarm_samples_torch[alarm_samples_torch.PVSAT == "LOW"],
            joint=True,
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False
        )
        ve_results = alarm_variable_elimination_torch.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=alarm_samples_torch[alarm_samples_torch.PVSAT == "LOW"],
            joint=False,
        )
        for var in ["HISTORY", "CVP"]:
            assert query_results[var].__eq__(ve_results[var], atol=0.01)

    def test_virtual_evidence(
        self, infer_alarm_torch, alarm_variable_elimination_torch
    ):
        virtual_evid = TabularCPD(
            "PAP",
            3,
            [[0.2], [0.3], [0.5]],
            state_names={"PAP": ["LOW", "NORMAL", "HIGH"]},
        )
        query_results = infer_alarm_torch.query(
            variables=["HISTORY"], virtual_evidence=[virtual_evid]
        )
        ve_results = alarm_variable_elimination_torch.query(
            variables=["HISTORY"], virtual_evidence=[virtual_evid]
        )
        assert query_results.__eq__(ve_results, atol=0.01)

        query_results = infer_alarm_torch.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        ve_results = alarm_variable_elimination_torch.query(
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
    def test_inference(self, approx_inference_dbn_inference_torch):
        res1 = approx_inference_dbn_inference_torch.query([("Y", 1)], seed=42)
        expected1 = DiscreteFactor([("Y", 1)], [2], [0.2259, 0.7741])
        assert res1.__eq__(expected1, atol=0.01)
        res2 = approx_inference_dbn_inference_torch.query([("Y", 0), ("Y", 1)], seed=42)
        expected2 = DiscreteFactor(
            [("Y", 0), ("Y", 1)], [2, 2], [0.0510, 0.1763, 0.1698, 0.6029]
        )
        assert res2.__eq__(expected2, atol=0.01)
        res3 = approx_inference_dbn_inference_torch.query([("Y", 1), ("Y", 5)], seed=42)
        expected3 = DiscreteFactor(
            [("Y", 1), ("Y", 5)], [2, 2], [0.0476, 0.1732, 0.1762, 0.6030]
        )
        assert res3.__eq__(expected3, atol=0.01)

    def test_evidence(self, approx_inference_dbn_inference_torch):
        res1 = approx_inference_dbn_inference_torch.query(
            [("Y", 4)], evidence={("Y", 2): 0}
        )
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2232, 0.7768])
        assert res1.__eq__(expected1, atol=0.01)

    def test_virtual_evidence(self, approx_inference_dbn_inference_torch):
        res1 = approx_inference_dbn_inference_torch.query(
            [("Y", 4)], virtual_evidence=[TabularCPD(("Y", 2), 2, [[0.2], [0.8]])]
        )
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2205, 0.7795])
        assert res1.__eq__(expected1, atol=0.01)
