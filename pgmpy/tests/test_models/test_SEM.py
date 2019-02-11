import unittest

import numpy as np
import networkx as nx
import numpy.testing as npt

from pgmpy.models import SEM

class TestSEMInit(unittest.TestCase):
    def setUp(self):
        # TODO: Add tests for fixed parameters in error correlations.
        self.lisrel = SEM(ebunch=[('xi1', 'x1'),
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

        self.non_lisrel = SEM(ebunch=[('xi1', 'eta1'),
                                      ('xi1', 'y1'),
                                      ('xi1', 'y4'),
                                      ('xi1', 'x1'),
                                      ('xi1', 'x2'),
                                      ('y4', 'y1'),
                                      ('y1', 'eta2'),
                                      ('eta2', 'y5'),
                                      ('y1', 'eta1'),
                                      ('eta1', 'y2'),
                                      ('eta1', 'y3')],
                              latents=['xi1', 'eta1', 'eta2'],
                              err_corr={'y1': {'y2'},
                                        'y2': {'y1', 'y3'},
                                        'y3': {'y2'}})

        self.lisrel_params = SEM(ebunch=[('xi1', 'x1', 0.1),
                                         ('xi1', 'x2', 0.2),
                                         ('xi1', 'x3', 0.3),
                                         ('xi1', 'eta1'),
                                         ('eta1', 'y1', 0.5),
                                         ('eta1', 'y2'),
                                         ('eta1', 'y3', 0.7),
                                         ('eta1', 'y4'),
                                         ('eta1', 'eta2', 0.9),
                                         ('xi1', 'eta2', 1.0),
                                         ('eta2', 'y5', 1.1),
                                         ('eta2', 'y6'),
                                         ('eta2', 'y7', 1.3),
                                         ('eta2', 'y8', 1.4)],
                                 latents=['xi1', 'eta1', 'eta2'],
                                 err_corr={'y1': {'y5'},
                                           'y2': {'y6', 'y4'},
                                           'y3': {'y7'},
                                           'y4': {'y8', 'y2'},
                                           'y5': {'y1'},
                                           'y6': {'y2', 'y8'},
                                           'y7': {'y3'},
                                           'y8': {'y4', 'y6'}})

        self.non_lisrel_params = SEM(ebunch=[('xi1', 'eta1'),
                                             ('xi1', 'y1', 0.2),
                                             ('xi1', 'y4'),
                                             ('xi1', 'x1'),
                                             ('xi1', 'x2', 0.5),
                                             ('y4', 'y1', 0.6),
                                             ('y1', 'eta2', 0.7),
                                             ('eta2', 'y5'),
                                             ('y1', 'eta1'),
                                             ('eta1', 'y2', 1.0),
                                             ('eta1', 'y3', 1.1)],
                                     latents=['xi1', 'eta1', 'eta2'],
                                     err_corr={'y1': {'y2'},
                                               'y2': {'y1', 'y3'},
                                               'y3': {'y2'}})

    def test_lisrel_init(self):
        self.assertSetEqual(self.lisrel.latents, {'xi1', 'eta1', 'eta2'})
        self.assertSetEqual(self.lisrel.observed, {'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                                                   'y4', 'y5', 'y6', 'y7', 'y8'})

        self.assertListEqual(sorted(self.lisrel.graph.nodes()),
                             ['eta1', 'eta2', 'x1', 'x2', 'x3', 'xi1', 'y1', 'y2',
                              'y3', 'y4', 'y5', 'y6', 'y7', 'y8'])
        self.assertListEqual(sorted(self.lisrel.graph.edges()),
                             sorted([('eta1', 'eta2'), ('eta1', 'y1'), ('eta1', 'y2'),
                                     ('eta1', 'y3'), ('eta1', 'y4'), ('eta2', 'y5'),
                                     ('eta2', 'y6'), ('eta2', 'y7'), ('eta2', 'y8'),
                                     ('xi1', 'eta1'), ('xi1', 'eta2'), ('xi1', 'x1'),
                                     ('xi1', 'x2'), ('xi1', 'x3')]))

        self.assertTrue(self.lisrel.graph.node['xi1']['latent'])
        self.assertTrue(self.lisrel.graph.node['eta1']['latent'])
        self.assertTrue(self.lisrel.graph.node['eta2']['latent'])
        self.assertFalse(self.lisrel.graph.node['y1']['latent'])
        self.assertFalse(self.lisrel.graph.node['y5']['latent'])
        self.assertFalse(self.lisrel.graph.node['x1']['latent'])

        self.assertDictEqual(self.lisrel.graph.edges[('xi1', 'x1')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('xi1', 'x2')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('xi1', 'x3')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('xi1', 'eta1')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('eta1', 'y1')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('eta1', 'y2')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('eta1', 'y3')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('eta1', 'y4')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('eta1', 'eta2')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('xi1', 'eta2')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('eta2', 'y5')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('eta2', 'y6')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('eta2', 'y7')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel.graph.edges[('eta2', 'y8')], {'weight': np.NaN})

        self.assertListEqual(sorted(self.lisrel.latent_struct.nodes()),
                             ['eta1', 'eta2', 'xi1'])
        self.assertListEqual(sorted(self.lisrel.latent_struct.edges()),
                             [('eta1', 'eta2'), ('xi1', 'eta1'), ('xi1', 'eta2')])

        self.assertListEqual(sorted(self.lisrel.eta), ['eta1', 'eta2'])
        self.assertListEqual(sorted(self.lisrel.xi), ['xi1'])
        self.assertListEqual(sorted(self.lisrel.y), ['y1', 'y2', 'y3', 'y4',
                                                     'y5', 'y6', 'y7', 'y8'])
        self.assertListEqual(sorted(self.lisrel.x), ['x1', 'x2', 'x3'])

        self.assertListEqual(sorted(self.lisrel.err_graph.nodes()),
                             ['eta1', 'eta2', 'x1', 'x2', 'x3', 'xi1', 'y1', 'y2', 'y3',
                              'y4', 'y5', 'y6', 'y7', 'y8'])
        npt.assert_equal(nx.to_numpy_matrix(self.lisrel.err_graph, nodelist=sorted(self.lisrel.err_graph.nodes()), weight=None),
                         np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
                                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.]]))

        for edge in self.lisrel.err_graph.edges():
            self.assertDictEqual(self.lisrel.err_graph.edges[edge], {'weight': np.NaN})

    def test_non_lisrel_init(self):
        self.assertSetEqual(self.non_lisrel.latents, {'eta1', 'eta2', 'xi1', '_l_y1', '_l_y4'})
        self.assertSetEqual(self.non_lisrel.observed, {'x1', 'x2', 'y1', 'y2', 'y3', 'y4', 'y5'})

        self.assertListEqual(sorted(self.non_lisrel.graph.nodes()),
                             ['_l_y1', '_l_y4', 'eta1', 'eta2', 'x1',
                              'x2', 'xi1', 'y1', 'y2', 'y3', 'y4', 'y5'])
        self.assertListEqual(sorted(self.non_lisrel.graph.edges()),
                             [('_l_y1', 'eta1'), ('_l_y1', 'eta2'), ('_l_y1', 'y1'),
                              ('_l_y4', '_l_y1'), ('_l_y4', 'y4'), ('eta1', 'y2'), ('eta1', 'y3'),
                              ('eta2', 'y5'), ('xi1', '_l_y1'), ('xi1', '_l_y4'), ('xi1', 'eta1'),
                              ('xi1', 'x1'), ('xi1', 'x2')])

        self.assertDictEqual(self.non_lisrel.graph.edges[('_l_y1', 'y1')], {'weight': 1.0})
        self.assertDictEqual(self.non_lisrel.graph.edges[('_l_y4', 'y4')], {'weight': 1.0})

        self.assertTrue(self.non_lisrel.graph.node['_l_y1']['latent'])
        self.assertTrue(self.non_lisrel.graph.node['_l_y4']['latent'])
        self.assertTrue(self.non_lisrel.graph.node['xi1']['latent'])
        self.assertTrue(self.non_lisrel.graph.node['eta1']['latent'])
        self.assertTrue(self.non_lisrel.graph.node['eta2']['latent'])
        self.assertFalse(self.non_lisrel.graph.node['x1']['latent'])
        self.assertFalse(self.non_lisrel.graph.node['x2']['latent'])
        self.assertFalse(self.non_lisrel.graph.node['y1']['latent'])
        self.assertFalse(self.non_lisrel.graph.node['y2']['latent'])
        self.assertFalse(self.non_lisrel.graph.node['y3']['latent'])
        self.assertFalse(self.non_lisrel.graph.node['y4']['latent'])

        self.assertListEqual(sorted(self.non_lisrel.latent_struct.nodes()),
                             ['_l_y1', '_l_y4', 'eta1', 'eta2', 'xi1'])
        self.assertListEqual(sorted(self.non_lisrel.latent_struct.edges()),
                             [('_l_y1', 'eta1'), ('_l_y1', 'eta2'), ('_l_y4', '_l_y1'),
                              ('xi1', '_l_y1'), ('xi1', '_l_y4'), ('xi1', 'eta1')])

        self.assertListEqual(sorted(self.non_lisrel.eta), ['_l_y1', '_l_y4', 'eta1', 'eta2'])
        self.assertListEqual(sorted(self.non_lisrel.xi), ['xi1'])
        self.assertListEqual(sorted(self.non_lisrel.y), ['y1', 'y2', 'y3', 'y4', 'y5'])
        self.assertListEqual(sorted(self.non_lisrel.x), ['x1', 'x2'])

        self.assertListEqual(sorted(self.non_lisrel.err_graph.nodes()),
                             sorted(['_l_y1', '_l_y4', 'eta1', 'eta2', 'x1', 'x2', 'xi1', 'y1', 'y2',
                                     'y3', 'y4', 'y5']))
        npt.assert_equal(nx.to_numpy_matrix(self.non_lisrel.err_graph,
                                            nodelist=sorted(self.non_lisrel.err_graph.nodes()),
                                            weight=None),
                         np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))

        for edge in self.non_lisrel.err_graph.edges():
            self.assertDictEqual(self.non_lisrel.err_graph.edges[edge], {'weight': np.NaN})

    def test_lisrel_param_init(self):
        self.assertDictEqual(self.lisrel_params.graph.edges[('xi1', 'x1')], {'weight': 0.1})
        self.assertDictEqual(self.lisrel_params.graph.edges[('xi1', 'x2')], {'weight': 0.2})
        self.assertDictEqual(self.lisrel_params.graph.edges[('xi1', 'x3')], {'weight': 0.3})
        self.assertDictEqual(self.lisrel_params.graph.edges[('xi1', 'eta1')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel_params.graph.edges[('eta1', 'y1')], {'weight': 0.5})
        self.assertDictEqual(self.lisrel_params.graph.edges[('eta1', 'y2')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel_params.graph.edges[('eta1', 'y3')], {'weight': 0.7})
        self.assertDictEqual(self.lisrel_params.graph.edges[('eta1', 'y4')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel_params.graph.edges[('eta1', 'eta2')], {'weight': 0.9})
        self.assertDictEqual(self.lisrel_params.graph.edges[('xi1', 'eta2')], {'weight': 1.0})
        self.assertDictEqual(self.lisrel_params.graph.edges[('eta2', 'y5')], {'weight': 1.1})
        self.assertDictEqual(self.lisrel_params.graph.edges[('eta2', 'y6')], {'weight': np.NaN})
        self.assertDictEqual(self.lisrel_params.graph.edges[('eta2', 'y7')], {'weight': 1.3})
        self.assertDictEqual(self.lisrel_params.graph.edges[('eta2', 'y8')], {'weight': 1.4})

    def test_non_lisrel_param_init(self):
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('xi1', 'eta1')], {'weight': np.NaN})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('xi1', '_l_y1')], {'weight': 0.2})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('xi1', '_l_y4')], {'weight': np.NaN})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('xi1', 'x1')], {'weight': np.NaN})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('xi1', 'x2')], {'weight': 0.5})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('_l_y4', '_l_y1')], {'weight': 0.6})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('_l_y1', 'eta2')], {'weight': 0.7})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('eta2', 'y5')], {'weight': np.NaN})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('_l_y1', 'eta1')], {'weight': np.NaN})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('eta1', 'y2')], {'weight': 1.0})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('eta1', 'y3')], {'weight': 1.1})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('_l_y1', 'y1')], {'weight': 1.0})
        self.assertDictEqual(self.non_lisrel_params.graph.edges[('_l_y4', 'y4')], {'weight': 1.0})

    def test_lisrel_get_fixed_masks(self):
        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
         theta_del_mask, psi_mask) = self.lisrel.get_fixed_masks(sort_vars=True)

        npt.assert_equal(B_mask, np.array([[0., 0.],
                                           [0., 0.]]))
        npt.assert_equal(gamma_mask, np.array([[0.],
                                               [0.]]))
        npt.assert_equal(wedge_y_mask, np.array([[0., 0.],
                                                 [0., 0.],
                                                 [0., 0.],
                                                 [0., 0.],
                                                 [0., 0.],
                                                 [0., 0.],
                                                 [0., 0.],
                                                 [0., 0.]]))
        npt.assert_equal(wedge_x_mask, np.array([[0.],
                                                 [0.],
                                                 [0.]]))

        npt.assert_equal(phi_mask, np.array([[0.]]))
        npt.assert_equal(theta_e_mask,
                         np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.]]))
        npt.assert_equal(theta_del_mask,
                         np.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]))
        npt.assert_equal(psi_mask,
                         np.array([[0., 0.],
                                   [0., 0.]]))

    def test_lisrel_get_masks(self):
        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
         theta_del_mask, psi_mask) = self.lisrel.get_masks(sort_vars=True)

        npt.assert_equal(B_mask, np.array([[0., 0.],
                                           [1., 0.]]))
        npt.assert_equal(gamma_mask, np.array([[1.],
                                               [1.]]))
        npt.assert_equal(wedge_y_mask, np.array([[1., 0.],
                                                 [1., 0.],
                                                 [1., 0.],
                                                 [1., 0.],
                                                 [0., 1.],
                                                 [0., 1.],
                                                 [0., 1.],
                                                 [0., 1.]]))
        npt.assert_equal(wedge_x_mask, np.array([[1.],
                                                 [1.],
                                                 [1.]]))

        npt.assert_equal(phi_mask, np.array([[0.]]))
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

    def test_non_lisrel_get_fixed_masks(self):
        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
         theta_del_mask, psi_mask) = self.non_lisrel.get_fixed_masks(sort_vars=True)

        npt.assert_equal(B_mask, np.array([[0., 0., 0., 0.],
                                           [0., 0., 0., 0.],
                                           [0., 0., 0., 0.],
                                           [0., 0., 0., 0.]]))
        npt.assert_equal(gamma_mask, np.array([[0.],
                                               [0.],
                                               [0.],
                                               [0.]]))
        npt.assert_equal(wedge_y_mask, np.array([[1., 0., 0., 0.],
                                                 [0., 0., 0., 0.],
                                                 [0., 0., 0., 0.],
                                                 [0., 1., 0., 0.],
                                                 [0., 0., 0., 0.]]))
        npt.assert_equal(wedge_x_mask, np.array([[0.],
                                                 [0.]]))

        npt.assert_equal(phi_mask, np.array([[0.]]))

        npt.assert_equal(theta_e_mask, np.array([[0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.]]))
        npt.assert_equal(theta_del_mask, np.array([[0., 0.],
                                                   [0., 0.]]))
        npt.assert_equal(psi_mask, np.array([[0., 0., 0., 0.],
                                             [0., 0., 0., 0.],
                                             [0., 0., 0., 0.],
                                             [0., 0., 0., 0.]]))

    def test_non_lisrel_get_masks(self):
        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
         theta_del_mask, psi_mask) = self.non_lisrel.get_masks(sort_vars=True)

        npt.assert_equal(B_mask, np.array([[0., 1., 0., 0.],
                                           [0., 0., 0., 0.],
                                           [1., 0., 0., 0.],
                                           [1., 0., 0., 0.]]))
        npt.assert_equal(gamma_mask, np.array([[1.],
                                               [1.],
                                               [1.],
                                               [0.]]))
        npt.assert_equal(wedge_y_mask, np.array([[0., 0., 0., 0.],
                                                 [0., 0., 1., 0.],
                                                 [0., 0., 1., 0.],
                                                 [0., 0., 0., 0.],
                                                 [0., 0., 0., 1.]]))
        npt.assert_equal(wedge_x_mask, np.array([[1.],
                                                 [1.]]))

        npt.assert_equal(phi_mask, np.array([[0.]]))

        npt.assert_equal(theta_e_mask, np.array([[0., 1., 0., 0., 0.],
                                                 [1., 0., 1., 0., 0.],
                                                 [0., 1., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.]]))
        npt.assert_equal(theta_del_mask, np.array([[0., 0.],
                                                   [0., 0.]]))
        npt.assert_equal(psi_mask, np.array([[0., 0., 0., 0.],
                                             [0., 0., 0., 0.],
                                             [0., 0., 0., 0.],
                                             [0., 0., 0., 0.]]))

    def test_lisrel_fixed_param_get_fixed_masks(self):
        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
         theta_del_mask, psi_mask) = self.lisrel_params.get_fixed_masks(sort_vars=True)

        npt.assert_equal(B_mask, np.array([[0. , 0.],
                                           [0.9, 0.]]))
        npt.assert_equal(gamma_mask, np.array([[0.],
                                               [1.]]))
        npt.assert_equal(wedge_y_mask, np.array([[0.5, 0. ],
                                                 [0. , 0. ],
                                                 [0.7, 0. ],
                                                 [0. , 0. ],
                                                 [0. , 1.1],
                                                 [0. , 0. ],
                                                 [0. , 1.3],
                                                 [0. , 1.4]]))
        npt.assert_equal(wedge_x_mask, np.array([[0.1],
                                                 [0.2],
                                                 [0.3]]))

        npt.assert_equal(phi_mask, np.array([[0.]]))
        npt.assert_equal(theta_e_mask,
                         np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.]]))
        npt.assert_equal(theta_del_mask,
                         np.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]))
        npt.assert_equal(psi_mask,
                         np.array([[0., 0.],
                                   [0., 0.]]))

    def test_lisrel_fixed_param_get_masks(self):
        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
         theta_del_mask, psi_mask) = self.lisrel_params.get_masks(sort_vars=True)

        npt.assert_equal(B_mask, np.array([[0., 0.],
                                           [0., 0.]]))
        npt.assert_equal(gamma_mask, np.array([[1.],
                                               [0.]]))
        npt.assert_equal(wedge_y_mask, np.array([[0., 0.],
                                                 [1., 0.],
                                                 [0., 0.],
                                                 [1., 0.],
                                                 [0., 0.],
                                                 [0., 1.],
                                                 [0., 0.],
                                                 [0., 0.]]))
        npt.assert_equal(wedge_x_mask, np.array([[0.],
                                                 [0.],
                                                 [0.]]))

        npt.assert_equal(phi_mask, np.array([[0.]]))
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

    def test_non_lisrel_param_get_fixed_masks(self):
        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
         theta_del_mask, psi_mask) = self.non_lisrel_params.get_fixed_masks(sort_vars=True)

        npt.assert_equal(B_mask, np.array([[0. , 0.6, 0., 0.],
                                           [0. , 0. , 0., 0.],
                                           [0. , 0. , 0., 0.],
                                           [0.7, 0. , 0., 0.]]))
        npt.assert_equal(gamma_mask, np.array([[0.2],
                                               [0. ],
                                               [0. ],
                                               [0. ]]))
        npt.assert_equal(wedge_y_mask, np.array([[1., 0., 0. , 0.],
                                                 [0., 0., 1. , 0.],
                                                 [0., 0., 1.1, 0.],
                                                 [0., 1., 0. , 0.],
                                                 [0., 0., 0. , 0.]]))
        npt.assert_equal(wedge_x_mask, np.array([[0.],
                                                 [0.5]]))

        npt.assert_equal(phi_mask, np.array([[0.]]))

        npt.assert_equal(theta_e_mask, np.array([[0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.]]))
        npt.assert_equal(theta_del_mask, np.array([[0., 0.],
                                                   [0., 0.]]))
        npt.assert_equal(psi_mask, np.array([[0., 0., 0., 0.],
                                             [0., 0., 0., 0.],
                                             [0., 0., 0., 0.],
                                             [0., 0., 0., 0.]]))

    def test_non_lisrel_params_get_masks(self):
        (B_mask, gamma_mask, wedge_y_mask, wedge_x_mask, phi_mask, theta_e_mask,
         theta_del_mask, psi_mask) = self.non_lisrel_params.get_masks(sort_vars=True)

        npt.assert_equal(B_mask, np.array([[0., 0., 0., 0.],
                                           [0., 0., 0., 0.],
                                           [1., 0., 0., 0.],
                                           [0., 0., 0., 0.]]))
        npt.assert_equal(gamma_mask, np.array([[0.],
                                               [1.],
                                               [1.],
                                               [0.]]))
        npt.assert_equal(wedge_y_mask, np.array([[0., 0., 0., 0.],
                                                 [0., 0., 0., 0.],
                                                 [0., 0., 0., 0.],
                                                 [0., 0., 0., 0.],
                                                 [0., 0., 0., 1.]]))
        npt.assert_equal(wedge_x_mask, np.array([[1.],
                                                 [0.]]))

        npt.assert_equal(phi_mask, np.array([[0.]]))

        npt.assert_equal(theta_e_mask, np.array([[0., 1., 0., 0., 0.],
                                                 [1., 0., 1., 0., 0.],
                                                 [0., 1., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.]]))
        npt.assert_equal(theta_del_mask, np.array([[0., 0.],
                                                   [0., 0.]]))
        npt.assert_equal(psi_mask, np.array([[0., 0., 0., 0.],
                                             [0., 0., 0., 0.],
                                             [0., 0., 0., 0.],
                                             [0., 0., 0., 0.]]))

    def test_iv_transformations(self):
        graph, err_graph = self.lisrel._iv_transformations('y1', 'y2')
        self.assertTrue(('y1', 'y2') in err_graph.edges)
        self.assertFalse(('eta1', 'y2') in graph.edges)

        graph, err_graph = self.lisrel._iv_transformations('y1', 'y3')
        self.assertTrue(('y1', 'y3') in err_graph.edges)
        self.assertFalse(('eta1', 'y3') in graph.edges)

        graph, err_graph = self.lisrel._iv_transformations('x1', 'y1', indicators={'xi1': 'x1'})
        self.assertTrue(('eta1', 'y1') in err_graph.edges)
        self.assertTrue(('x1', 'y1') in err_graph.edges)
        self.assertFalse(('eta1', 'y1') in graph.edges)

        graph, err_graph = self.lisrel._iv_transformations('x1', 'y5', indicators={'xi1': 'x1', 'eta1': 'y1'})
        self.assertTrue(('y1', 'y5') in err_graph.edges)
        self.assertTrue(('eta2', 'y5') in err_graph.edges)
        self.assertTrue(('x1', 'y5') in err_graph.edges)
        self.assertFalse(('eta2', 'y5') in graph.edges)
