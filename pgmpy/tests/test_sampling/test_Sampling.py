import unittest

from mock import MagicMock, patch

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD, State
from pgmpy.models import BayesianModel, MarkovModel
from pgmpy.sampling import BayesianModelSampling, GibbsSampling


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
        self.assertTrue(set(sample.A).issubset({0, 1}))
        self.assertTrue(set(sample.J).issubset({0, 1}))
        self.assertTrue(set(sample.R).issubset({0, 1}))
        self.assertTrue(set(sample.Q).issubset({0, 1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

    def test_rejection_sample_basic(self):
        sample = self.sampling_inference.rejection_sample([State('A', 1), State('J', 1), State('R', 1)], 25)
        self.assertEquals(len(sample), 25)
        self.assertEquals(len(sample.columns), 6)
        self.assertIn('A', sample.columns)
        self.assertIn('J', sample.columns)
        self.assertIn('R', sample.columns)
        self.assertIn('Q', sample.columns)
        self.assertIn('G', sample.columns)
        self.assertIn('L', sample.columns)
        self.assertTrue(set(sample.A).issubset({1}))
        self.assertTrue(set(sample.J).issubset({1}))
        self.assertTrue(set(sample.R).issubset({1}))
        self.assertTrue(set(sample.Q).issubset({0, 1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

    @patch("pgmpy.sampling.BayesianModelSampling.forward_sample", autospec=True)
    def test_rejection_sample_less_arg(self, forward_sample):
        sample = self.sampling_inference.rejection_sample(size=5)
        forward_sample.assert_called_once_with(self.sampling_inference, 5)
        self.assertEqual(sample, forward_sample.return_value)

    def test_likelihood_weighted_sample(self):
        sample = self.sampling_inference.likelihood_weighted_sample([State('A', 0), State('J', 1), State('R', 0)], 25)
        self.assertEquals(len(sample), 25)
        self.assertEquals(len(sample.columns), 7)
        self.assertIn('A', sample.columns)
        self.assertIn('J', sample.columns)
        self.assertIn('R', sample.columns)
        self.assertIn('Q', sample.columns)
        self.assertIn('G', sample.columns)
        self.assertIn('L', sample.columns)
        self.assertIn('_weight', sample.columns)
        self.assertTrue(set(sample.A).issubset({0, 1}))
        self.assertTrue(set(sample.J).issubset({0, 1}))
        self.assertTrue(set(sample.R).issubset({0, 1}))
        self.assertTrue(set(sample.Q).issubset({0, 1}))
        self.assertTrue(set(sample.G).issubset({0, 1}))
        self.assertTrue(set(sample.L).issubset({0, 1}))

    def tearDown(self):
        del self.sampling_inference
        del self.bayesian_model
        del self.markov_model


class TestGibbsSampling(unittest.TestCase):
    def setUp(self):
        # A test Bayesian model
        diff_cpd = TabularCPD('diff', 2, [[0.6], [0.4]])
        intel_cpd = TabularCPD('intel', 2, [[0.7], [0.3]])
        grade_cpd = TabularCPD('grade', 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
                               evidence=['diff', 'intel'], evidence_card=[2, 2])
        self.bayesian_model = BayesianModel()
        self.bayesian_model.add_nodes_from(['diff', 'intel', 'grade'])
        self.bayesian_model.add_edges_from([('diff', 'grade'), ('intel', 'grade')])
        self.bayesian_model.add_cpds(diff_cpd, intel_cpd, grade_cpd)

        # A test Markov model
        self.markov_model = MarkovModel([('A', 'B'), ('C', 'B'), ('B', 'D')])
        factor_ab = DiscreteFactor(['A', 'B'], [2, 3], [1, 2, 3, 4, 5, 6])
        factor_cb = DiscreteFactor(['C', 'B'], [4, 3], [3, 1, 4, 5, 7, 8, 1, 3, 10, 4, 5, 6])
        factor_bd = DiscreteFactor(['B', 'D'], [3, 2], [5, 7, 2, 1, 9, 3])
        self.markov_model.add_factors(factor_ab, factor_cb, factor_bd)

        self.gibbs = GibbsSampling(self.bayesian_model)

    def tearDown(self):
        del self.bayesian_model
        del self.markov_model

    @patch('pgmpy.sampling.GibbsSampling._get_kernel_from_bayesian_model', autospec=True)
    @patch('pgmpy.models.MarkovChain.__init__', autospec=True)
    def test_init_bayesian_model(self, init, get_kernel):
        model = MagicMock(spec_set=BayesianModel)
        gibbs = GibbsSampling(model)
        init.assert_called_once_with(gibbs)
        get_kernel.assert_called_once_with(gibbs, model)

    @patch('pgmpy.sampling.GibbsSampling._get_kernel_from_markov_model', autospec=True)
    def test_init_markov_model(self, get_kernel):
        model = MagicMock(spec_set=MarkovModel)
        gibbs = GibbsSampling(model)
        get_kernel.assert_called_once_with(gibbs, model)

    def test_get_kernel_from_bayesian_model(self):
        gibbs = GibbsSampling()
        gibbs._get_kernel_from_bayesian_model(self.bayesian_model)
        self.assertListEqual(list(gibbs.variables), self.bayesian_model.nodes())
        self.assertDictEqual(gibbs.cardinalities, {'diff': 2, 'intel': 2, 'grade': 3})

    def test_get_kernel_from_markov_model(self):
        gibbs = GibbsSampling()
        gibbs._get_kernel_from_markov_model(self.markov_model)
        self.assertListEqual(list(gibbs.variables), self.markov_model.nodes())
        self.assertDictEqual(gibbs.cardinalities, {'A': 2, 'B': 3, 'C': 4, 'D': 2})

    def test_sample(self):
        start_state = [State('diff', 0), State('intel', 0), State('grade', 0)]
        sample = self.gibbs.sample(start_state, 2)
        self.assertEquals(len(sample), 2)
        self.assertEquals(len(sample.columns), 3)
        self.assertIn('diff', sample.columns)
        self.assertIn('intel', sample.columns)
        self.assertIn('grade', sample.columns)
        self.assertTrue(set(sample['diff']).issubset({0, 1}))
        self.assertTrue(set(sample['intel']).issubset({0, 1}))
        self.assertTrue(set(sample['grade']).issubset({0, 1, 2}))

    @patch("pgmpy.sampling.GibbsSampling.random_state", autospec=True)
    def test_sample_less_arg(self, random_state):
        self.gibbs.state = None
        random_state.return_value = [State('diff', 0), State('intel', 0), State('grade', 0)]
        sample = self.gibbs.sample(size=2)
        random_state.assert_called_once_with(self.gibbs)
        self.assertEqual(len(sample), 2)

    def test_generate_sample(self):
        start_state = [State('diff', 0), State('intel', 0), State('grade', 0)]
        gen = self.gibbs.generate_sample(start_state, 2)
        samples = [sample for sample in gen]
        self.assertEqual(len(samples), 2)
        self.assertEqual({samples[0][0].var, samples[0][1].var, samples[0][2].var}, {'diff', 'intel', 'grade'})
        self.assertEqual({samples[1][0].var, samples[1][1].var, samples[1][2].var}, {'diff', 'intel', 'grade'})

    @patch("pgmpy.sampling.GibbsSampling.random_state", autospec=True)
    def test_generate_sample_less_arg(self, random_state):
        self.gibbs.state = None
        gen = self.gibbs.generate_sample(size=2)
        samples = [sample for sample in gen]
        random_state.assert_called_once_with(self.gibbs)
        self.assertEqual(len(samples), 2)
