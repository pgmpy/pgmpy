import unittest

import pandas as pd

from pgmpy.models import SEM
from pgmpy.estimators import SEMEstimator


class TestSEMEstimator(unittest.TestCase):
    def setUp(self):
        self.G = SEM(ebunch=[('xi1', 'x1'),
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

        self.data = pd.read_csv('pgmpy/tests/test_estimators/testdata/PoliticalDemocracy.csv',
                                index_col=0, header=0)

    # def test_ml_estimator(self):
    #     estimator = SEMEstimator(self.G)
    #     B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, psi = estimator.fit(self.data, method='ols')
