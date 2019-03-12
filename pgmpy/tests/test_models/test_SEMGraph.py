import unittest

import numpy as np
import networkx as nx
import numpy.testing as npt

from pgmpy.models import SEMGraph

import unittest
import unittest

import numpy as np
import networkx as nx
import numpy.testing as npt

from pgmpy.models import SEMGraph


class TestSEMGraph(unittest.TestCase):
    def setUp(self):
        self.demo = SEMGraph(ebunch=[('xi1', 'x1'),
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
                             err_corr=[('y1', 'y5'),
                                       ('y2', 'y6'),
                                       ('y2', 'y4'),
                                       ('y3', 'y7'),
                                       ('y4', 'y8'),
                                       ('y6', 'y8')])

        self.union = SEMGraph(ebunch=[('yrsmill', 'unionsen'),
                                      ('age', 'laboract'),
                                      ('age', 'deferenc'),
                                      ('deferenc', 'laboract'),
                                      ('deferenc', 'unionsen'),
                                      ('laboract', 'unionsen')],
                              latents=[],
                              err_corr=[('yrsmill', 'age')])

        self.demo_params = SEMGraph(ebunch=[('xi1', 'x1', 0.4),
                                            ('xi1', 'x2', 0.5),
                                            ('xi1', 'x3', 0.6),
                                            ('xi1', 'eta1', 0.3),
                                            ('eta1', 'y1', 1.1),
                                            ('eta1', 'y2', 1.2),
                                            ('eta1', 'y3', 1.3),
                                            ('eta1', 'y4', 1.4),
                                            ('eta1', 'eta2', 0.1),
                                            ('xi1', 'eta2', 0.2),
                                            ('eta2', 'y5', 0.7),
                                            ('eta2', 'y6', 0.8),
                                            ('eta2', 'y7', 0.9),
                                            ('eta2', 'y8', 1.0)],
                                    latents=['xi1', 'eta1', 'eta2'],
                                    err_corr=[('y1', 'y5', 1.5),
                                              ('y2', 'y6', 1.6),
                                              ('y2', 'y4', 1.9),
                                              ('y3', 'y7', 1.7),
                                              ('y4', 'y8', 1.8),
                                              ('y6', 'y8', 2.0)],
                                    err_var={'y1': 2.1, 'y2': 2.2, 'y3': 2.3, 'y4': 2.4,
                                             'y5': 2.5, 'y6': 2.6, 'y7': 2.7, 'y8': 2.8,
                                             'x1': 3.1, 'x2': 3.2, 'x3': 3.3, 'eta1': 2.9,
                                             'eta2': 3.0, 'xi1': 3.4})

        self.custom = SEMGraph(ebunch=[('xi1', 'eta1'),
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
                               err_corr=[('y1', 'y2'),
                                         ('y2', 'y3')],
                               err_var={})

    def test_demo_init(self):
        self.assertSetEqual(self.demo.latents, {'xi1', 'eta1', 'eta2'})
        self.assertSetEqual(self.demo.observed, {'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                                                 'y4', 'y5', 'y6', 'y7', 'y8'})
        self.assertListEqual(sorted(self.demo.graph.nodes()),
                             ['eta1', 'eta2', 'x1', 'x2', 'x3', 'xi1', 'y1', 'y2', 'y3',
                              'y4', 'y5', 'y6', 'y7', 'y8'])
        self.assertListEqual(sorted(self.demo.graph.edges()),
                             sorted([('eta1', 'eta2'), ('eta1', 'y1'), ('eta1', 'y2'),
                                     ('eta1', 'y3'), ('eta1', 'y4'), ('eta2', 'y5'),
                                     ('eta2', 'y6'), ('eta2', 'y7'), ('eta2', 'y8'),
                                     ('xi1', 'eta1'), ('xi1', 'eta2'), ('xi1', 'x1'),
                                     ('xi1', 'x2'), ('xi1', 'x3')]))

        self.assertDictEqual(self.demo.graph.edges[('xi1', 'x1')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('xi1', 'x2')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('xi1', 'x3')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('xi1', 'eta1')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('eta1', 'y1')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('eta1', 'y2')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('eta1', 'y3')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('eta1', 'y4')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('eta1', 'eta2')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('xi1', 'eta2')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('eta2', 'y5')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('eta2', 'y6')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('eta2', 'y7')], {'weight': np.NaN})
        self.assertDictEqual(self.demo.graph.edges[('eta2', 'y8')], {'weight': np.NaN})

        npt.assert_equal(nx.to_numpy_matrix(self.demo.err_graph,
                                            nodelist=sorted(self.demo.err_graph.nodes()), weight=None),
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

        for edge in self.demo.err_graph.edges():
            self.assertDictEqual(self.demo.err_graph.edges[edge], {'weight': np.NaN})
        for node in self.demo.err_graph.nodes():
            self.assertDictEqual(self.demo.err_graph.nodes[node], {'weight': np.NaN})

    def test_union_init(self):
        self.assertSetEqual(self.union.latents, set())
        self.assertSetEqual(self.union.observed, {'yrsmill', 'unionsen', 'age', 'laboract', 'deferenc'})

        self.assertListEqual(sorted(self.union.graph.nodes()),
                             sorted(['yrsmill', 'unionsen', 'age', 'laboract', 'deferenc']))
        self.assertListEqual(sorted(self.union.graph.edges()),
                             sorted([('yrsmill', 'unionsen'), ('age', 'laboract'), ('age', 'deferenc'),
                                     ('deferenc', 'laboract'),('deferenc', 'unionsen'),
                                     ('laboract', 'unionsen')]))

        self.assertDictEqual(self.union.graph.edges[('yrsmill', 'unionsen')], {'weight': np.NaN})
        self.assertDictEqual(self.union.graph.edges[('age', 'laboract')], {'weight': np.NaN})
        self.assertDictEqual(self.union.graph.edges[('age', 'deferenc')], {'weight': np.NaN})
        self.assertDictEqual(self.union.graph.edges[('deferenc', 'laboract')], {'weight': np.NaN})
        self.assertDictEqual(self.union.graph.edges[('deferenc', 'unionsen')], {'weight': np.NaN})
        self.assertDictEqual(self.union.graph.edges[('laboract', 'unionsen')], {'weight': np.NaN})

        npt.assert_equal(nx.to_numpy_matrix(self.union.err_graph, nodelist=sorted(self.union.err_graph.nodes()),
                                            weight=None),
                         np.array([[0., 0., 0., 0., 1.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.],
                                   [1., 0., 0., 0., 0.]]))

        for edge in self.union.err_graph.edges():
            self.assertDictEqual(self.union.err_graph.edges[edge], {'weight': np.NaN})
        for node in self.union.err_graph.nodes():
            self.assertDictEqual(self.union.err_graph.nodes[node], {'weight': np.NaN})

    def test_demo_param_init(self):
        self.assertDictEqual(self.demo_params.graph.edges[('xi1', 'x1')], {'weight': 0.4})
        self.assertDictEqual(self.demo_params.graph.edges[('xi1', 'x2')], {'weight': 0.5})
        self.assertDictEqual(self.demo_params.graph.edges[('xi1', 'x3')], {'weight': 0.6})
        self.assertDictEqual(self.demo_params.graph.edges[('xi1', 'eta1')], {'weight': 0.3})
        self.assertDictEqual(self.demo_params.graph.edges[('eta1', 'y1')], {'weight': 1.1})
        self.assertDictEqual(self.demo_params.graph.edges[('eta1', 'y2')], {'weight': 1.2})
        self.assertDictEqual(self.demo_params.graph.edges[('eta1', 'y3')], {'weight': 1.3})
        self.assertDictEqual(self.demo_params.graph.edges[('eta1', 'y4')], {'weight': 1.4})
        self.assertDictEqual(self.demo_params.graph.edges[('eta1', 'eta2')], {'weight': 0.1})
        self.assertDictEqual(self.demo_params.graph.edges[('xi1', 'eta2')], {'weight': 0.2})
        self.assertDictEqual(self.demo_params.graph.edges[('eta2', 'y5')], {'weight': 0.7})
        self.assertDictEqual(self.demo_params.graph.edges[('eta2', 'y6')], {'weight': 0.8})
        self.assertDictEqual(self.demo_params.graph.edges[('eta2', 'y7')], {'weight': 0.9})
        self.assertDictEqual(self.demo_params.graph.edges[('eta2', 'y8')], {'weight': 1.0})

        self.assertDictEqual(self.demo_params.err_graph.edges[('y1', 'y5')], {'weight': 1.5})
        self.assertDictEqual(self.demo_params.err_graph.edges[('y2', 'y6')], {'weight': 1.6})
        self.assertDictEqual(self.demo_params.err_graph.edges[('y2', 'y4')], {'weight': 1.9})
        self.assertDictEqual(self.demo_params.err_graph.edges[('y3', 'y7')], {'weight': 1.7})
        self.assertDictEqual(self.demo_params.err_graph.edges[('y4', 'y8')], {'weight': 1.8})
        self.assertDictEqual(self.demo_params.err_graph.edges[('y6', 'y8')], {'weight': 2.0})

        self.assertDictEqual(self.demo_params.err_graph.nodes['y1'], {'weight': 2.1})
        self.assertDictEqual(self.demo_params.err_graph.nodes['y2'], {'weight': 2.2})
        self.assertDictEqual(self.demo_params.err_graph.nodes['y3'], {'weight': 2.3})
        self.assertDictEqual(self.demo_params.err_graph.nodes['y4'], {'weight': 2.4})
        self.assertDictEqual(self.demo_params.err_graph.nodes['y5'], {'weight': 2.5})
        self.assertDictEqual(self.demo_params.err_graph.nodes['y6'], {'weight': 2.6})
        self.assertDictEqual(self.demo_params.err_graph.nodes['y7'], {'weight': 2.7})
        self.assertDictEqual(self.demo_params.err_graph.nodes['y8'], {'weight': 2.8})
        self.assertDictEqual(self.demo_params.err_graph.nodes['x1'], {'weight': 3.1})
        self.assertDictEqual(self.demo_params.err_graph.nodes['x2'], {'weight': 3.2})
        self.assertDictEqual(self.demo_params.err_graph.nodes['x3'], {'weight': 3.3})
        self.assertDictEqual(self.demo_params.err_graph.nodes['eta1'], {'weight': 2.9})
        self.assertDictEqual(self.demo_params.err_graph.nodes['eta2'], {'weight': 3.0})

    def test_get_full_graph_struct(self):
        full_struct = self.union._get_full_graph_struct()
        self.assertFalse(set(full_struct.nodes()) -
                         set(['yrsmill', 'unionsen', 'age', 'laboract', 'deferenc',
                                 '.yrsmill', '.unionsen', '.age', '.laboract', '.deferenc',
                                 '..ageyrsmill', '..yrsmillage']))
        self.assertFalse(set(full_struct.edges()) -
                         set([('yrsmill', 'unionsen'), ('age', 'laboract'), ('age', 'deferenc'),
                              ('deferenc', 'laboract'),('deferenc', 'unionsen'),
                              ('laboract', 'unionsen'), ('.yrsmill', 'yrsmill'), ('.unionsen', 'unionsen'),
                              ('.age', 'age'), ('.laboract', 'laboract'), ('.deferenc', 'deferenc'),
                              ('..ageyrsmill', '.age'), ('..ageyrsmill', '.yrsmill'),
                              ('..yrsmillage', '.age'), ('..yrsmillage', '.yrsmill')]))

    def test_active_trail_nodes(self):
        demo_nodes = ['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8']
        for node in demo_nodes:
            self.assertSetEqual(self.demo.active_trail_nodes(node, struct='full')[node], set(demo_nodes))

        union_nodes = self.union.graph.nodes()
        active_trails = self.union.active_trail_nodes(list(union_nodes), struct='full')
        for node in union_nodes:
            self.assertSetEqual(active_trails[node], set(union_nodes))

        self.assertSetEqual(self.union.active_trail_nodes(
            'age', observed=['laboract', 'deferenc', 'unionsen'])['age'], {'age', 'yrsmill'})

    def test_get_scaling_indicators(self):
        demo_scaling_indicators = self.demo.get_scaling_indicators()
        self.assertTrue(demo_scaling_indicators['eta1'] in ['y1', 'y2', 'y3', 'y4'])
        self.assertTrue(demo_scaling_indicators['eta2'] in ['y5', 'y6', 'y7', 'y8'])
        self.assertTrue(demo_scaling_indicators['xi1'] in ['x1', 'x2', 'x3'])

        union_scaling_indicators = self.union.get_scaling_indicators()
        self.assertDictEqual(union_scaling_indicators, dict())

        custom_scaling_indicators = self.custom.get_scaling_indicators()
        self.assertTrue(custom_scaling_indicators['xi1'] in ['x1', 'x2', 'y1', 'y4'])
        self.assertTrue(custom_scaling_indicators['eta1'] in ['y2', 'y3'])
        self.assertTrue(custom_scaling_indicators['eta2'] in ['y5'])

    def test_to_lisrel(self):
        demo_lisrel = self.demo.to_lisrel()
        union_lisrel = self.union.to_lisrel()
        demo_params_lisrel = self.demo_params.to_lisrel()
        custom_lisrel = self.custom.to_lisrel()

        demo_graph = demo_lisrel.to_SEMGraph()
        union_graph = union_lisrel.to_SEMGraph()
        demo_params_graph = demo_params_lisrel.to_SEMGraph()
        custom_graph = custom_lisrel.to_SEMGraph()

        # Test demo
        self.assertSetEqual(set(self.demo.graph.nodes()), set(demo_graph.graph.nodes()))
        self.assertSetEqual(set(self.demo.graph.edges()), set(demo_graph.graph.edges()))
        self.assertSetEqual(set(self.demo.err_graph.nodes()), set(demo_graph.err_graph.nodes()))
        self.assertSetEqual(set(self.demo.err_graph.edges()), set(demo_graph.err_graph.edges()))
        self.assertSetEqual(set(self.demo.full_graph_struct.nodes()),
                            set(demo_graph.full_graph_struct.nodes()))
        self.assertSetEqual(set(self.demo.full_graph_struct.edges()),
                            set(demo_graph.full_graph_struct.edges()))
        self.assertSetEqual(self.demo.latents, demo_graph.latents)
        self.assertSetEqual(self.demo.observed, demo_graph.observed)

        # Test union
        self.assertSetEqual(set(self.union.graph.nodes()), set(union_graph.graph.nodes()))
        self.assertSetEqual(set(self.union.graph.edges()), set(union_graph.graph.edges()))
        self.assertSetEqual(set(self.union.err_graph.nodes()), set(union_graph.err_graph.nodes()))
        self.assertSetEqual(set(self.union.err_graph.edges()), set(union_graph.err_graph.edges()))
        self.assertSetEqual(set(self.union.full_graph_struct.nodes()),
                            set(union_graph.full_graph_struct.nodes()))
        self.assertSetEqual(set(self.union.full_graph_struct.edges()),
                            set(union_graph.full_graph_struct.edges()))
        self.assertSetEqual(self.union.latents, union_graph.latents)
        self.assertSetEqual(self.union.observed, union_graph.observed)

        # Test demo_params
        self.assertSetEqual(set(self.demo_params.graph.nodes()),
                            set(demo_params_graph.graph.nodes()))
        self.assertSetEqual(set(self.demo_params.graph.edges()),
                            set(demo_params_graph.graph.edges()))
        self.assertSetEqual(set(self.demo_params.err_graph.nodes()),
                            set(demo_params_graph.err_graph.nodes()))
        self.assertSetEqual(set(self.demo_params.err_graph.edges()),
                            set(demo_params_graph.err_graph.edges()))
        self.assertSetEqual(set(self.demo_params.full_graph_struct.nodes()),
                            set(demo_params_graph.full_graph_struct.nodes()))
        self.assertSetEqual(set(self.demo_params.full_graph_struct.edges()),
                            set(demo_params_graph.full_graph_struct.edges()))
        self.assertSetEqual(self.demo_params.latents, demo_params_graph.latents)
        self.assertSetEqual(self.demo_params.observed, demo_params_graph.observed)

        # Test demo
        self.assertSetEqual(set(self.custom.graph.nodes()), set(custom_graph.graph.nodes()))
        self.assertSetEqual(set(self.custom.graph.edges()), set(custom_graph.graph.edges()))
        self.assertSetEqual(set(self.custom.err_graph.nodes()), set(custom_graph.err_graph.nodes()))
        self.assertSetEqual(set(self.custom.err_graph.edges()), set(custom_graph.err_graph.edges()))
        self.assertSetEqual(set(self.custom.full_graph_struct.nodes()),
                            set(custom_graph.full_graph_struct.nodes()))
        self.assertSetEqual(set(self.custom.full_graph_struct.edges()),
                            set(custom_graph.full_graph_struct.edges()))
        self.assertSetEqual(self.custom.latents, custom_graph.latents)
        self.assertSetEqual(self.custom.observed, custom_graph.observed)
