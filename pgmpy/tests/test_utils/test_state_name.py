import unittest
import itertools

import numpy as np
import numpy.testing as np_test

from pgmpy.factors import Factor
from pgmpy.factors import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.inference import Inference
from pgmpy.utils import StateNameInit
from pgmpy.utils import StateNameDecorator


class TestStateNameInit(unittest.TestCase):

	def setUp(self):
		self.sn2 = {'grade': ['A', 'B', 'F'], 'diff':['high', 'low'], 
					'intel':['poor', 'good', 'very good']}
		self.sn1 = {'speed': ['low', 'medium', 'high'],
             			'switch': ['on', 'off'],
             			'time': ['day', 'night']}

		self.phi1 = Factor(['speed', 'switch', 'time'],
                     [3, 2, 2], np.ones(12))
		self.phi2 = Factor(['speed', 'switch', 'time'],
                     [3, 2, 2], np.ones(12), state_names=self.sn1)

		self.cpd1 = TabularCPD('grade',3,[[0.1,0.1,0.1,0.1,0.1,0.1],
                                [0.1,0.1,0.1,0.1,0.1,0.1],
                                [0.8,0.8,0.8,0.8,0.8,0.8]],
                                evidence=['diff', 'intel'], evidence_card=[2,3])
		self.cpd2 = TabularCPD('grade',3,[[0.1,0.1,0.1,0.1,0.1,0.1],
                                [0.1,0.1,0.1,0.1,0.1,0.1],
                                [0.8,0.8,0.8,0.8,0.8,0.8]],
                                evidence=['diff', 'intel'], evidence_card=[2,3],
                                state_names=self.sn2)

		student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
		diff_cpd = TabularCPD('diff', 2, [[0.2, 0.8]])
		intel_cpd = TabularCPD('intel', 2, [[0.3, 0.7]])
		grade_cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1],
    	                                     [0.1, 0.1, 0.1, 0.1],
                          	               [0.8, 0.8, 0.8, 0.8]],
                          	  evidence=['diff', 'intel'], evidence_card=[2, 2])
		student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
		self.model1 = Inference(student)
		self.model2 = Inference(student, state_names=self.sn2)

	def test_factor_init(self):
		self.assertEqual(self.phi1.state_names, None)
		self.assertEqual(self.phi2.state_names, self.sn1)

	def test_cpd_init(self):
		self.assertEqual(self.cpd1.state_names, None)
		self.assertEqual(self.cpd2.state_names, self.sn2)

	def test_inference_init(self):
		self.assertEqual(self.model1.state_names, None)
		self.assertEqual(self.model2.state_names, self.sn2)
