import unittest
from pgmpy.models import MarkovModel
from pgmpy.inference.Sampling import BayesianModelSampling
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD


class TestBayesianModelSampling(unittest.TestCase):
    def setUp(self):
        self.bayesian_model = BayesianModel([('A', 'J'), ('R', 'J'), ('J', 'Q'),
                                             ('J', 'L'), ('G', 'L')])
        cpd_a = TabularCPD('A', 2, [[0.2], [0.8]])
        cpd_r = TabularCPD('R', 2, [[0.4], [0.6]])
        cpd_j = TabularCPD('J', 2,
                           [[0.9, 0.6, 0.7, 0.1],
                            [0.1, 0.4, 0.3, 0.9]],
                           ['R', 'A'], [2, 2])
        cpd_q = TabularCPD('Q', 2,
                           [[0.9, 0.2],
                            [0.1, 0.8]],
                           ['J'], [2])
        cpd_l = TabularCPD('L', 2,
                           [[0.9, 0.45, 0.8, 0.1],
                            [0.1, 0.55, 0.2, 0.9]],
                           ['G', 'J'], [2, 2])
        cpd_g = TabularCPD('G', 2, [[0.6], [0.4]])
        self.bayesian_model.add_cpds(cpd_a, cpd_g, cpd_j, cpd_l, cpd_q, cpd_r)
        self.sampling_inference = BayesianModelSampling(self.bayesian_model)
        self.markov_model = MarkovModel()

    def test_init(self):
        with self.assertRaises(TypeError):
            BayesianModelSampling(self.markov_model)

    def test_forward_sample(self):
        sample = self.sampling_inference.forward_sample(25)
        self.assertEquals(len(sample), 25)
        self.assertEquals(len(sample.columns), 6)
        self.assertIn('A', sample.columns)
        self.assertIn('J', sample.columns)
        self.assertIn('R', sample.columns)
        self.assertIn('Q', sample.columns)
        self.assertIn('G', sample.columns)
        self.assertIn('L', sample.columns)
        self.assertEquals(set(sample.A), {'A_0', 'A_1', None})
        self.assertTrue(set(sample.J).issubset({'J_0', 'J_1'}))
        self.assertTrue(set(sample.R).issubset({'R_0', 'R_1'}))
        self.assertTrue(set(sample.Q).issubset({'Q_0', 'Q_1'}))
        self.assertTrue(set(sample.G).issubset({'G_0', 'G_1'}))
        self.assertTrue(set(sample.L).issubset({'L_0', 'L_1'}))

    def test_rejection_sample_basic(self):
        sample = self.sampling_inference.rejection_sample({'A': 'A_0',
                                                           'J': 'J_1',
                                                           'R': 'R_0'}, 25)
        self.assertEquals(len(sample), 25)
        self.assertEquals(len(sample.columns), 3)
        self.assertIn('Q', sample.columns)
        self.assertIn('G', sample.columns)
        self.assertIn('L', sample.columns)
        self.assertEquals(set(sample.Q), {'Q_0', 'Q_1', None})
        self.assertTrue(set(sample.G).issubset({'G_0', 'G_1'}))
        self.assertTrue(set(sample.L).issubset({'L_0', 'L_1'}))

    def tearDown(self):
        del self.sampling_inference
        del self.bayesian_model
        del self.markov_model
