import unittest

import pandas as pd
import numpy as np

from pgmpy.models import SEM
from pgmpy.estimators import SEMEstimator


class TestSEMEstimator(unittest.TestCase):
    def setUp(self):
        self.custom = SEM(ebunch=[('a', 'b'),
                                  ('b', 'c')],
                          latents=[],
                          err_corr={})
        a = np.random.randn(10**3)
        b = a + np.random.normal(loc=0, scale=0.1, size=10**3)
        c = b + np.random.normal(loc=0, scale=0.2, size=10**3)
        self.custom_data = pd.DataFrame({'a': a, 'b': b, 'c': c})
        self.custom_data -= self.custom_data.mean(axis=0)

        self.demo = SEM(ebunch=[('xi1', 'x1'),
                                ('xi1', 'x2'),
                                ('xi1', 'x3'),
                                ('xi1', 'eta1'),
                                ('eta1', 'y1'),
                                ('eta1', 'y2'),
                                ('eta1', 'y3'),
                                ('eta1', 'y4'),
                                ('eta1', 'eta2'),
                                ('xi1', 'eta2'),
                                ('eta2', 'y5'),
                                ('eta2', 'y6'),
                                ('eta2', 'y7'),
                                ('eta2', 'y8')],
                        latents=['xi1', 'eta1', 'eta2'],
                        err_corr={'y1': {'y5'},
                                  'y2': {'y6', 'y4'},
                                  'y3': {'y7'},
                                  'y4': {'y8', 'y2'},
                                  'y5': {'y1'},
                                  'y6': {'y2', 'y8'},
                                  'y7': {'y3'},
                                  'y8': {'y4', 'y6'}})

        self.demo_data = pd.read_csv('pgmpy/tests/test_estimators/testdata/democracy1989a.csv',
                                     index_col=0, header=0)

        self.union = SEM(ebunch=[('yrsmill', 'unionsen'),
                                 ('age', 'laboract'),
                                 ('age', 'deferenc'),
                                 ('deferenc', 'laboract'),
                                 ('deferenc', 'unionsen'),
                                 ('laboract', 'unionsen')],
                         latents=[],
                         err_corr={'yrsmill': {'age'},
                                   'age': {'yrsmill'}})

        self.union_data = pd.read_csv('pgmpy/tests/test_estimators/testdata/union1989b.csv',
                                      index_col=0, header=0)

    def test_get_init_values(self):
        demo_estimator = SEMEstimator(self.demo)
        for method in ['random', 'std']:
            init_values = demo_estimator.get_init_values(data=self.demo_data, method=method)

            m, n, p, q = len(self.demo.eta), len(self.demo.xi), len(self.demo.y), len(self.demo.x)
            self.assertEqual(init_values['B'].shape, (m, m))
            self.assertEqual(init_values['gamma'].shape, (m, n))
            self.assertEqual(init_values['wedge_y'].shape, (p, m))
            self.assertEqual(init_values['wedge_x'].shape, (q, n))
            self.assertEqual(init_values['theta_e'].shape, (p, p))
            self.assertEqual(init_values['theta_del'].shape, (q, q))
            self.assertEqual(init_values['psi'].shape, (m, m))
            self.assertEqual(init_values['phi'].shape, (n, n))

            union_estimator = SEMEstimator(self.union)
            init_values = union_estimator.get_init_values(data=self.union_data, method=method)
            m, n, p, q = len(self.union.eta), len(self.union.xi), len(self.union.y), len(self.union.x)
            self.assertEqual(init_values['B'].shape, (m, m))
            self.assertEqual(init_values['gamma'].shape, (m, n))
            self.assertEqual(init_values['wedge_y'].shape, (p, m))
            self.assertEqual(init_values['wedge_x'].shape, (q, n))
            self.assertEqual(init_values['theta_e'].shape, (p, p))
            self.assertEqual(init_values['theta_del'].shape, (q, q))
            self.assertEqual(init_values['psi'].shape, (m, m))
            self.assertEqual(init_values['phi'].shape, (n, n))

    @unittest.skip
    def test_demo_estimator_random_init(self):
        estimator = SEMEstimator(self.demo)
        summary = estimator.fit(self.demo_data, method='ml')

    def test_union_estimator_random_init(self):
        estimator = SEMEstimator(self.union)
        summary = estimator.fit(self.union_data, method='ml', opt='adam', max_iter=10**6, exit_delta=1e-1)

    def test_custom_estimator_random_init(self):
        estimator = SEMEstimator(self.custom)
        summary = estimator.fit(self.custom_data, method='ml', max_iter=10**6, opt='adam')
        summary = estimator.fit(self.custom_data, method='uls', max_iter=10**6, opt='adam')
        summary = estimator.fit(self.custom_data, method='gls', max_iter=10**6, opt='adam', W=np.ones((3, 3)))

    def test_union_estimator_std_init(self):
        estimator = SEMEstimator(self.union)
        summary = estimator.fit(self.union_data, method='ml', opt='adam', init_values ='std',
                                max_iter=10**6, exit_delta=1e-1)

    def test_custom_estimator_std_init(self):
        estimator = SEMEstimator(self.custom)
        summary = estimator.fit(self.custom_data, method='ml', init_values='std', max_iter=10**6, opt='adam')

