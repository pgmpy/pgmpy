import unittest

from pgmpy import config
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import ApproxInference, VariableElimination
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.utils import get_example_model


class TestApproxInferenceBN(unittest.TestCase):
    def setUp(self):
        self.alarm_model = get_example_model("alarm")
        self.infer_alarm = ApproxInference(self.alarm_model)
        self.alarm_ve = VariableElimination(self.alarm_model)
        self.samples = self.alarm_model.simulate(int(1e4))

    def test_query_marg(self):
        query_results = self.infer_alarm.query(variables=["HISTORY"])
        ve_results = self.alarm_ve.query(variables=["HISTORY"])
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY"], samples=self.samples
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(variables=["HISTORY", "CVP"], joint=True)
        ve_results = self.alarm_ve.query(variables=["HISTORY", "CVP"], joint=True)
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], samples=self.samples, joint=True
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], joint=False
        )
        ve_results = self.alarm_ve.query(variables=["HISTORY", "CVP"], joint=False)
        for var in ["HISTORY", "CVP"]:
            self.assertTrue(query_results[var].__eq__(ve_results[var], atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], samples=self.samples, joint=False
        )
        for var in ["HISTORY", "CVP"]:
            self.assertTrue(query_results[var].__eq__(ve_results[var], atol=0.01))

    def test_query_evidence(self):
        query_results = self.infer_alarm.query(
            variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            samples=self.samples[self.samples.PVSAT == "LOW"],
            joint=True,
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=self.samples[self.samples.PVSAT == "LOW"],
            joint=True,
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False
        )
        for var in ["HISTORY", "CVP"]:
            self.assertTrue(query_results[var].__eq__(ve_results[var], atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=self.samples[self.samples.PVSAT == "LOW"],
            joint=False,
        )
        for var in ["HISTORY", "CVP"]:
            self.assertTrue(query_results[var].__eq__(ve_results[var], atol=0.01))

    def test_virtual_evidence(self):
        virtual_evid = TabularCPD(
            "PAP",
            3,
            [[0.2], [0.3], [0.5]],
            state_names={"PAP": ["LOW", "NORMAL", "HIGH"]},
        )
        query_results = self.infer_alarm.query(
            variables=["HISTORY"], virtual_evidence=[virtual_evid]
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY"], virtual_evidence=[virtual_evid]
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))


class TestApproxInferenceDBN(unittest.TestCase):
    def setUp(self):
        self.model = DBN()
        self.model.add_edges_from(
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
        self.model.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        self.model.initialize_initial_state()
        self.infer = ApproxInference(self.model)

    def test_inference(self):
        res1 = self.infer.query([("Y", 1)], seed=42)
        expected1 = DiscreteFactor([("Y", 1)], [2], [0.2259, 0.7741])
        self.assertTrue(res1.__eq__(expected1, atol=0.01))
        res2 = self.infer.query([("Y", 0), ("Y", 1)], seed=42)
        expected2 = DiscreteFactor(
            [("Y", 0), ("Y", 1)], [2, 2], [0.0510, 0.1763, 0.1698, 0.6029]
        )
        self.assertTrue(res2.__eq__(expected2, atol=0.01))
        res3 = self.infer.query([("Y", 1), ("Y", 5)], seed=42)
        expected3 = DiscreteFactor(
            [("Y", 1), ("Y", 5)], [2, 2], [0.0476, 0.1732, 0.1762, 0.6030]
        )
        self.assertTrue(res3.__eq__(expected3, atol=0.01))

    def test_evidence(self):
        res1 = self.infer.query([("Y", 4)], evidence={("Y", 2): 0})
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2232, 0.7768])
        self.assertTrue(res1.__eq__(expected1, atol=0.01))

    def test_virtual_evidence(self):
        res1 = self.infer.query(
            [("Y", 4)], virtual_evidence=[TabularCPD(("Y", 2), 2, [[0.2], [0.8]])]
        )
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2205, 0.7795])
        self.assertTrue(res1.__eq__(expected1, atol=0.01))


class TestApproxInferenceBNTorch(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.alarm_model = get_example_model("alarm")
        self.infer_alarm = ApproxInference(self.alarm_model)
        self.alarm_ve = VariableElimination(self.alarm_model)
        self.samples = self.alarm_model.simulate(int(1e4))

    def test_query_marg(self):
        query_results = self.infer_alarm.query(variables=["HISTORY"])
        ve_results = self.alarm_ve.query(variables=["HISTORY"])
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY"], samples=self.samples
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(variables=["HISTORY", "CVP"], joint=True)
        ve_results = self.alarm_ve.query(variables=["HISTORY", "CVP"], joint=True)
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], samples=self.samples, joint=True
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], joint=False
        )
        ve_results = self.alarm_ve.query(variables=["HISTORY", "CVP"], joint=False)
        for var in ["HISTORY", "CVP"]:
            self.assertTrue(query_results[var].__eq__(ve_results[var], atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], samples=self.samples, joint=False
        )
        for var in ["HISTORY", "CVP"]:
            self.assertTrue(query_results[var].__eq__(ve_results[var], atol=0.01))

    def test_query_evidence(self):
        query_results = self.infer_alarm.query(
            variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY"], evidence={"PVSAT": "LOW"}, joint=True
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            samples=self.samples[self.samples.PVSAT == "LOW"],
            joint=True,
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=self.samples[self.samples.PVSAT == "LOW"],
            joint=True,
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=False
        )
        for var in ["HISTORY", "CVP"]:
            self.assertTrue(query_results[var].__eq__(ve_results[var], atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"],
            evidence={"PVSAT": "LOW"},
            samples=self.samples[self.samples.PVSAT == "LOW"],
            joint=False,
        )
        for var in ["HISTORY", "CVP"]:
            self.assertTrue(query_results[var].__eq__(ve_results[var], atol=0.01))

    def test_virtual_evidence(self):
        virtual_evid = TabularCPD(
            "PAP",
            3,
            [[0.2], [0.3], [0.5]],
            state_names={"PAP": ["LOW", "NORMAL", "HIGH"]},
        )
        query_results = self.infer_alarm.query(
            variables=["HISTORY"], virtual_evidence=[virtual_evid]
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY"], virtual_evidence=[virtual_evid]
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY"],
            evidence={"PVSAT": "LOW"},
            virtual_evidence=[virtual_evid],
            joint=True,
        )
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

    def tearDown(self):
        config.set_backend("numpy")


class TestApproxInferenceDBN(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")

        self.model = DBN()
        self.model.add_edges_from(
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
        self.model.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        self.model.initialize_initial_state()
        self.infer = ApproxInference(self.model)

    def test_inference(self):
        res1 = self.infer.query([("Y", 1)], seed=42)
        expected1 = DiscreteFactor([("Y", 1)], [2], [0.2259, 0.7741])
        self.assertTrue(res1.__eq__(expected1, atol=0.01))
        res2 = self.infer.query([("Y", 0), ("Y", 1)], seed=42)
        expected2 = DiscreteFactor(
            [("Y", 0), ("Y", 1)], [2, 2], [0.0510, 0.1763, 0.1698, 0.6029]
        )
        self.assertTrue(res2.__eq__(expected2, atol=0.01))
        res3 = self.infer.query([("Y", 1), ("Y", 5)], seed=42)
        expected3 = DiscreteFactor(
            [("Y", 1), ("Y", 5)], [2, 2], [0.0476, 0.1732, 0.1762, 0.6030]
        )
        self.assertTrue(res3.__eq__(expected3, atol=0.01))

    def test_evidence(self):
        res1 = self.infer.query([("Y", 4)], evidence={("Y", 2): 0})
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2232, 0.7768])
        self.assertTrue(res1.__eq__(expected1, atol=0.01))

    def test_virtual_evidence(self):
        res1 = self.infer.query(
            [("Y", 4)], virtual_evidence=[TabularCPD(("Y", 2), 2, [[0.2], [0.8]])]
        )
        expected1 = DiscreteFactor([("Y", 4)], [2], [0.2205, 0.7795])
        self.assertTrue(res1.__eq__(expected1, atol=0.01))

    def tearDown(self):
        config.set_backend("numpy")
