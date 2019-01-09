import unittest

import numpy as np
import numpy.testing as npt

from pgmpy.models import SEM

class TestSEMInit(unittest.TestCase):
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

    def test_init(self):
        self.assertSetEqual(self.G.latents, {'xi1', 'eta1', 'eta2'})
        self.assertSetEqual(self.G.observed, {'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                                              'y4', 'y5', 'y6', 'y7', 'y8'})

        self.assertListEqual(sorted(self.G.graph.nodes()),
                             ['eta1', 'eta2', 'x1', 'x2', 'x3', 'xi1', 'y1', 'y2',
                              'y3', 'y4', 'y5', 'y6', 'y7', 'y8'])
        self.assertListEqual(sorted(self.G.graph.edges()),
                             [('eta1', 'eta2'), ('eta1', 'y1'), ('eta1', 'y2'),
                              ('eta1', 'y3'), ('eta1', 'y4'), ('eta2', 'y5'),
                              ('eta2', 'y6'), ('eta2', 'y7'), ('eta2', 'y8'),
                              ('xi1', 'eta1'), ('xi1', 'eta2'), ('xi1', 'x1'),
                              ('xi1', 'x2'), ('xi1', 'x3')])

        self.assertTrue(self.G.graph.node['xi1']['latent'])
        self.assertTrue(self.G.graph.node['eta1']['latent'])
        self.assertTrue(self.G.graph.node['eta2']['latent'])
        self.assertFalse(self.G.graph.node['y1']['latent'])
        self.assertFalse(self.G.graph.node['y5']['latent'])
        self.assertFalse(self.G.graph.node['x1']['latent'])

        self.assertListEqual(sorted(self.G.latent_struct.nodes()),
                             ['eta1', 'eta2', 'xi1'])
        self.assertListEqual(sorted(self.G.latent_struct.edges()),
                             [('eta1', 'eta2'), ('xi1', 'eta1'), ('xi1', 'eta2')])

        self.assertListEqual(sorted(self.G.eta), ['eta1', 'eta2'])
        self.assertListEqual(sorted(self.G.xi), ['xi1'])
        self.assertListEqual(sorted(self.G.y), ['y1', 'y2', 'y3', 'y4',
                                                'y5', 'y6', 'y7', 'y8'])
        self.assertListEqual(sorted(self.G.x), ['x1', 'x2', 'x3'])

        self.assertListEqual(sorted(self.G.err_graph.nodes()),
                             ['eta1', 'eta2', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                              'y4', 'y5', 'y6', 'y7', 'y8'])
        self.assertListEqual(sorted(self.G.err_graph.edges()),
                             [('y1', 'y5'), ('y2', 'y4'), ('y2', 'y6'),
                              ('y3', 'y7'), ('y4', 'y8'), ('y6', 'y8')])

    def test_get_masks(self):
        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
        theta_del_mask, psi_mask) = self.G.get_masks()

        # TODO: This might fail if the order of variables in self.x, self.y, self.eta, self.xi
        #       etc changes. Make these tests robust to that.
        npt.assert_equal(B_mask, np.array([[0., 0.],
                                           [1., 0.]]))
        npt.assert_equal(gamma_mask, np.array([[1., 1.]]))
        npt.assert_equal(wedge_y_mask, np.array([[1., 1., 1., 1., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 1., 1., 1., 1.]]))
        npt.assert_equal(wedge_x_mask, np.array([[1., 1., 1.]]))
        npt.assert_equal(phi_mask, np.array([[1.]]))
        npt.assert_equal(theta_e_mask,
                         np.array([[0., 0., 0., 0., 1., 0., 0., 0.],
                                   [0., 0., 0., 1., 0., 1., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 1., 0.],
                                   [0., 1., 0., 0., 0., 0., 0., 1.],
                                   [1., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 1., 0., 0., 0., 0., 0., 1.],
                                   [0., 0., 1., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 1., 0., 1., 0., 0.]]))
        npt.assert_equal(theta_del_mask,
                         np.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]))
        npt.assert_equal(psi_mask,
                         np.array([[0., 0.],
                                   [0., 0.]]))

