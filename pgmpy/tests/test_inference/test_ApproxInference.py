import unittest

from pgmpy.utils import get_example_model
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import ApproxInference, VariableElimination


class TestApproxInference(unittest.TestCase):
    def setUp(self):
        self.alarm_model = get_example_model("alarm")
        self.infer_alarm = ApproxInference(self.alarm_model)
        self.alarm_ve = VariableElimination(self.alarm_model)

    def test_query_marg(self):
        query_results = self.infer_alarm.query(variables=["HISTORY"])
        ve_results = self.alarm_ve.query(variables=["HISTORY"])
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(variables=["HISTORY", "CVP"], joint=True)
        ve_results = self.alarm_ve.query(variables=["HISTORY", "CVP"], joint=True)
        self.assertTrue(query_results.__eq__(ve_results, atol=0.01))

        query_results = self.infer_alarm.query(
            variables=["HISTORY", "CVP"], joint=False
        )
        ve_results = self.alarm_ve.query(variables=["HISTORY", "CVP"], joint=False)
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
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
        )
        ve_results = self.alarm_ve.query(
            variables=["HISTORY", "CVP"], evidence={"PVSAT": "LOW"}, joint=True
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
