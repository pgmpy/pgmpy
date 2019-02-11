import unittest

import pandas as pd

from pgmpy.models import SEM
from pgmpy.estimators import SEMEstimator


class TestSEMEstimator(unittest.TestCase):
    def setUp(self):
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

        self.union = SEM(ebunch=[('x1', 'y3'),
                                 ('x2', 'y2'),
                                 ('x2', 'y1'),
                                 ('y1', 'y2'),
                                 ('y1', 'y3'),
                                 ('y2', 'y3')],
                         latents=[],
                         err_corr={'x1': {'x2'},
                                   'x2': {'x1'}})

        self.union_data = pd.read_csv('pgmpy/tests/test_estimators/testdata/union1989b.csv',
                                      index_col=0, header=0)

    @unittest.skip
    def test_demo_estimator(self):
        estimator = SEMEstimator(self.demo)
        B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, psi = estimator.fit(self.demo_data, method='ols')

    def test_union_estimator(self):
        estimator = SEMEstimator(self.union)
        B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, psi = estimator.fit(self.union_data, method='ols')
