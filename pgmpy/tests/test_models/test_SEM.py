import unittest

from pgmpy.models import SEM

class TestSEMInit(unittest.TestCase):
    def setUp(self):
        self.G = SEM(ebunch=[('psi1', 'x1'),
                             ('psi1', 'x2'),
                             ('psi1', 'x3'),
                             ('psi1', 'eta1'),
                             ('eta1', 'y1'),
                             ('eta1', 'y2'),
                             ('eta1', 'y3'),
                             ('eta1', 'y4'),
                             ('eta1', 'eta2'),
                             ('psi1', 'eta2'),
                             ('eta2', 'y5'),
                             ('eta2', 'y6'),
                             ('eta2', 'y7'),
                             ('eta2', 'y8')],
                     latents=['psi1', 'eta1', 'eta2'],
                     err_corr={'y1': {'y5'},
                               'y2': {'y6', 'y4'},
                               'y3': {'y7'},
                               'y4': {'y8', 'y2'},
                               'y5': {'y1'},
                               'y6': {'y2', 'y8'},
                               'y7': {'y3'},
                               'y8': {'y4', 'y6'}})

    def test_init(self):
        self.assertSetEqual(self.G.latents, {'psi1', 'eta1', 'eta2'})
        self.assertSetEqual(self.G.observed, {'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                                              'y4', 'y5', 'y6', 'y7', 'y8'})

        self.assertTrue(self.G.node['psi1']['latent'])
        self.assertTrue(self.G.node['eta1']['latent'])
        self.assertTrue(self.G.node['eta2']['latent'])
        self.assertFalse(self.G.node['y1']['latent'])
        self.assertFalse(self.G.node['y5']['latent'])
        self.assertFalse(self.G.node['x1']['latent'])

        self.assertSetEqual(self.G.err_corr['y1'], {'y5'})
        self.assertSetEqual(self.G.err_corr['y4'], {'y8', 'y2'})
        self.assertSetEqual(self.G.err_corr['y8'], {'y4', 'y6'})

