import unittest

import numpy as np
import networkx as nx
import numpy.testing as npt

from pgmpy.models import SEM, SEMGraph, SEMLISREL


class TestSEM(unittest.TestCase):
    def test_from_graph(self):
        self.demo = SEM.from_graph(ebunch=[('xi1', 'x1'),
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

    def test_from_lavaan(self):
        self.assertRaises(NotImplementedError, SEM.from_lavaan, "Doesn't matter what I write here")

    def test_from_lisrel(self):
        pass # TODO: Add this test when done writing the tests for SEMLISREL

    def test_from_ram(self):
        pass # TODO: Add this.


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
        npt.assert_array_equal(nx.to_numpy_matrix(self.demo.err_graph,
                                                  nodelist=sorted(self.demo.err_graph.nodes())),
                               nx.to_numpy_matrix(demo_graph,
                                                  nodelist=sorted(demo_graph.err_graph.nodes())))

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
        npt.assert_array_equal(nx.to_numpy_matrix(self.union.err_graph,
                                                  nodelist=sorted(self.union.err_graph.nodes())),
                               nx.to_numpy_matrix(union_graph,
                                                  nodelist=sorted(union_graph.err_graph.nodes())))

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
        npt.assert_array_equal(nx.to_numpy_matrix(self.demo_params.err_graph,
                                                  nodelist=sorted(self.demo_params.err_graph.nodes()),
                                                  weight=None),
                               nx.to_numpy_matrix(demo_graph.err_graph,
                                                  nodelist=sorted(demo_params_graph.err_graph.nodes()),
                                                  weight=None))

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
        npt.assert_array_equal(nx.to_numpy_matrix(self.custom.err_graph,
                                                  nodelist=sorted(self.custom.err_graph.nodes())),
                               nx.to_numpy_matrix(custom_graph,
                                                  nodelist=sorted(custom_graph.err_graph.nodes())))

        self.assertSetEqual(set(self.custom.full_graph_struct.nodes()),
                            set(custom_graph.full_graph_struct.nodes()))
        self.assertSetEqual(set(self.custom.full_graph_struct.edges()),
                            set(custom_graph.full_graph_struct.edges()))

        self.assertSetEqual(self.custom.latents, custom_graph.latents)
        self.assertSetEqual(self.custom.observed, custom_graph.observed)

    def test_iv_transformations_demo(self):
        scale = {'eta1': 'y1', 'eta2': 'y5', 'xi1': 'x1'}

        self.assertRaises(ValueError, self.demo._iv_transformations, 'x1', 'y1', scale)

        for y in ['y2', 'y3', 'y4']:
            full_graph, dependent_var = self.demo._iv_transformations(X='eta1', Y=y,
                                                                      scaling_indicators=scale)
            self.assertEqual(dependent_var, y)
            self.assertTrue(('.y1', y) in full_graph.edges)
            self.assertFalse(('eta1', y) in full_graph.edges)

        for y in ['y6', 'y7', 'y8']:
            full_graph, dependent_var = self.demo._iv_transformations(X='eta2', Y=y,
                                                                      scaling_indicators=scale)
            self.assertEqual(dependent_var, y)
            self.assertTrue(('.y5', y) in full_graph.edges)
            self.assertFalse(('eta2', y) in full_graph.edges)

        full_graph, dependent_var = self.demo._iv_transformations(X='xi1', Y='eta1', scaling_indicators=scale)
        self.assertEqual(dependent_var, 'y1')
        self.assertTrue(('.eta1', 'y1') in full_graph.edges())
        self.assertTrue(('.x1', 'y1') in full_graph.edges())
        self.assertFalse(('xi1', 'eta1') in full_graph.edges())

        full_graph, dependent_var = self.demo._iv_transformations(X='xi1', Y='eta2', scaling_indicators=scale)
        self.assertEqual(dependent_var, 'y5')
        self.assertTrue(('.y1', 'y5') in full_graph.edges())
        self.assertTrue(('.eta2', 'y5') in full_graph.edges())
        self.assertTrue(('.x1', 'y5') in full_graph.edges())
        self.assertFalse(('eta1', 'eta2') in full_graph.edges())
        self.assertFalse(('xi1', 'eta2') in full_graph.edges())

        full_graph, dependent_var = self.demo._iv_transformations(X='eta1', Y='eta2', scaling_indicators=scale)
        self.assertEqual(dependent_var, 'y5')
        self.assertTrue(('.y1', 'y5') in full_graph.edges())
        self.assertTrue(('.eta2', 'y5') in full_graph.edges())
        self.assertTrue(('.x1', 'y5') in full_graph.edges())
        self.assertFalse(('eta1', 'eta2') in full_graph.edges())
        self.assertFalse(('xi1', 'eta2') in full_graph.edges())

    def test_iv_transformations_union(self):
        scale = {}
        for u, v in self.union.graph.edges():
            full_graph, dependent_var = self.union._iv_transformations(u, v, scaling_indicators=scale)
            self.assertFalse((u, v) in full_graph.edges())
            self.assertEqual(dependent_var, v)

    def test_get_ivs_demo(self):
        scale = {'eta1': 'y1', 'eta2': 'y5', 'xi1': 'x1'}

        self.assertSetEqual(self.demo.get_ivs('eta1', 'y2', scaling_indicators=scale),
                            {'x1', 'x2', 'x3', 'y3', 'y7', 'y8'})
        self.assertSetEqual(self.demo.get_ivs('eta1', 'y3', scaling_indicators=scale),
                            {'x1', 'x2', 'x3', 'y2', 'y4', 'y6', 'y8'})
        self.assertSetEqual(self.demo.get_ivs('eta1', 'y4', scaling_indicators=scale),
                            {'x1', 'x2', 'x3', 'y3', 'y6', 'y7'})

        self.assertSetEqual(self.demo.get_ivs('eta2', 'y6', scaling_indicators=scale),
                            {'x1', 'x2', 'x3', 'y3', 'y4', 'y7'})
        self.assertSetEqual(self.demo.get_ivs('eta2', 'y7', scaling_indicators=scale),
                            {'x1', 'x2', 'x3', 'y2', 'y4', 'y6', 'y8'})
        self.assertSetEqual(self.demo.get_ivs('eta2', 'y8', scaling_indicators=scale),
                            {'x1', 'x2', 'x3', 'y2', 'y3', 'y7'})

        self.assertSetEqual(self.demo.get_ivs('xi1', 'x2', scaling_indicators=scale),
                            {'x3', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8'})
        self.assertSetEqual(self.demo.get_ivs('xi1', 'x3', scaling_indicators=scale),
                            {'x2', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8'})

        self.assertSetEqual(self.demo.get_ivs('xi1', 'eta1', scaling_indicators=scale),
                            {'x2', 'x3'})
        self.assertSetEqual(self.demo.get_ivs('xi1', 'eta2', scaling_indicators=scale),
                            {'x2', 'x3', 'y2', 'y3', 'y4'})
        self.assertSetEqual(self.demo.get_ivs('eta1', 'eta2', scaling_indicators=scale),
                            {'x2', 'x3', 'y2', 'y3', 'y4'})

    def test_get_conditional_ivs_demo(self):
        scale = {'eta1': 'y1', 'eta2': 'y5', 'xi1': 'x1'}

        self.assertEqual(self.demo.get_conditional_ivs('eta1', 'y2', scaling_indicators=scale), [])
        self.assertEqual(self.demo.get_conditional_ivs('eta1', 'y3', scaling_indicators=scale), [])
        self.assertEqual(self.demo.get_conditional_ivs('eta1', 'y4', scaling_indicators=scale), [])

        self.assertEqual(self.demo.get_conditional_ivs('eta2', 'y6', scaling_indicators=scale), [])
        self.assertEqual(self.demo.get_conditional_ivs('eta2', 'y7', scaling_indicators=scale), [])
        self.assertEqual(self.demo.get_conditional_ivs('eta2', 'y8', scaling_indicators=scale), [])

        self.assertEqual(self.demo.get_conditional_ivs('xi1', 'x2', scaling_indicators=scale), [])
        self.assertEqual(self.demo.get_conditional_ivs('xi1', 'x3', scaling_indicators=scale), [])

        self.assertEqual(self.demo.get_conditional_ivs('xi1', 'eta1', scaling_indicators=scale), [])
        self.assertEqual(self.demo.get_conditional_ivs('xi1', 'eta2', scaling_indicators=scale), [])
        self.assertEqual(self.demo.get_conditional_ivs('eta1', 'eta2', scaling_indicators=scale), [])

    def test_get_ivs_union(self):
        scale = {}
        self.assertSetEqual(self.union.get_ivs('yrsmill', 'unionsen', scaling_indicators=scale), set())
        self.assertSetEqual(self.union.get_ivs('deferenc', 'unionsen', scaling_indicators=scale), set())
        self.assertSetEqual(self.union.get_ivs('laboract', 'unionsen', scaling_indicators=scale), set())
        self.assertSetEqual(self.union.get_ivs('deferenc', 'laboract', scaling_indicators=scale), set())
        self.assertSetEqual(self.union.get_ivs('age', 'laboract', scaling_indicators=scale), {'yrsmill'})
        self.assertSetEqual(self.union.get_ivs('age', 'deferenc', scaling_indicators=scale), {'yrsmill'})

    def test_get_conditional_ivs_union(self):
        self.assertEqual(self.union.get_conditional_ivs('yrsmill', 'unionsen'),
                         [('age', {'laboract', 'deferenc'})])
        # This case wouldn't have conditonal IV if the Total effect between `deferenc` and 
        # `unionsen` needs to be computed because one of the conditional variable lies on the 
        # effect path.
        self.assertEqual(self.union.get_conditional_ivs('deferenc', 'unionsen'),
                         [('age', {'yrsmill', 'laboract'})])
        self.assertEqual(self.union.get_conditional_ivs('laboract', 'unionsen'),
                         [('age', {'yrsmill', 'deferenc'})])
        self.assertEqual(self.union.get_conditional_ivs('deferenc', 'laboract'), [])

        self.assertEqual(self.union.get_conditional_ivs('age', 'laboract'), [('yrsmill', {'deferenc'})])
        self.assertEqual(self.union.get_conditional_ivs('age', 'deferenc'), [])

    def test_iv_transformations_custom(self):
        scale_custom = {'eta1': 'y2', 'eta2': 'y5', 'xi1': 'x1'}

        full_graph, var = self.custom._iv_transformations('xi1', 'x2', scaling_indicators=scale_custom)
        self.assertEqual(var, 'x2')
        self.assertTrue(('.x1', 'x2') in full_graph.edges())
        self.assertFalse(('xi1', 'x2') in full_graph.edges())

        full_graph, var = self.custom._iv_transformations('xi1', 'y4', scaling_indicators=scale_custom)
        self.assertEqual(var, 'y4')
        self.assertTrue(('.x1', 'y4') in full_graph.edges())
        self.assertFalse(('xi1', 'y4') in full_graph.edges())

        full_graph, var = self.custom._iv_transformations('xi1', 'y1', scaling_indicators=scale_custom)
        self.assertEqual(var, 'y1')
        self.assertTrue(('.x1', 'y1') in full_graph.edges())
        self.assertFalse(('xi1', 'y1') in full_graph.edges())
        self.assertFalse(('y4', 'y1') in full_graph.edges())

        full_graph, var = self.custom._iv_transformations('xi1', 'eta1', scaling_indicators=scale_custom)
        self.assertEqual(var, 'y2')
        self.assertTrue(('.eta1', 'y2') in full_graph.edges())
        self.assertTrue(('.x1', 'y2') in full_graph.edges())
        self.assertFalse(('y1', 'eta1') in full_graph.edges())
        self.assertFalse(('xi1', 'eta1') in full_graph.edges())

        full_graph, var = self.custom._iv_transformations('y1', 'eta1', scaling_indicators=scale_custom)
        self.assertEqual(var, 'y2')
        self.assertTrue(('.eta1', 'y2') in full_graph.edges())
        self.assertTrue(('.x1', 'y2') in full_graph.edges())
        self.assertFalse(('y1', 'eta1') in full_graph.edges())
        self.assertFalse(('xi1', 'eta1') in full_graph.edges())

        full_graph, var = self.custom._iv_transformations('y1', 'eta2', scaling_indicators=scale_custom)
        self.assertEqual(var, 'y5')
        self.assertTrue(('.eta2', 'y5') in full_graph.edges())
        self.assertFalse(('y1', 'eta2') in full_graph.edges())

        full_graph, var = self.custom._iv_transformations('y4', 'y1', scaling_indicators=scale_custom)
        self.assertEqual(var, 'y1')
        self.assertFalse(('y4', 'y1') in full_graph.edges())

        full_graph, var = self.custom._iv_transformations('eta1', 'y3', scaling_indicators=scale_custom)
        self.assertEqual(var, 'y3')
        self.assertTrue(('.y2', 'y3') in full_graph.edges())
        self.assertFalse(('eta1', 'y3') in full_graph.edges())

    def test_get_ivs_custom(self):
        scale_custom = {'eta1': 'y2', 'eta2': 'y5', 'xi1': 'x1'}

        self.assertSetEqual(self.custom.get_ivs('xi1', 'x2', scaling_indicators=scale_custom),
                            {'y1', 'y2', 'y3', 'y4', 'y5'})
        self.assertSetEqual(self.custom.get_ivs('xi1', 'y4', scaling_indicators=scale_custom),
                            {'x2'})
        self.assertSetEqual(self.custom.get_ivs('xi1', 'y1', scaling_indicators=scale_custom),
                            {'x2', 'y4'})
        self.assertSetEqual(self.custom.get_ivs('xi1', 'eta1', scaling_indicators=scale_custom),
                            {'x2', 'y4'})
        self.assertSetEqual(self.custom.get_ivs('y1', 'eta1', scaling_indicators=scale_custom),
                           {'x2', 'y4'})
        self.assertSetEqual(self.custom.get_ivs('y1', 'eta2', scaling_indicators=scale_custom),
                            {'x1', 'x2', 'y2', 'y3', 'y4'})
        self.assertSetEqual(self.custom.get_ivs('y4', 'y1', scaling_indicators=scale_custom),
                            set())
        self.assertSetEqual(self.custom.get_ivs('eta1', 'y3', scaling_indicators=scale_custom),
                           {'x1', 'x2', 'y4'})

    def test_small_model_ivs(self):
        model1 = SEMGraph(ebunch=[('X', 'Y'), ('I', 'X'), ('W', 'I')],
                          latents=[],
                          err_corr=[('W', 'Y')],
                          err_var={})
        self.assertEqual(model1.get_conditional_ivs('X', 'Y'), [('I', {'W'})])

        model2 = SEMGraph(ebunch=[('x', 'y'), ('z', 'x'), ('w', 'z'), ('w', 'u'), ('u', 'x'), ('u', 'y')],
                          latents=['u'])
        self.assertEqual(model2.get_conditional_ivs('x', 'y'), [('z', {'w'})])

        model3 = SEMGraph(ebunch=[('x', 'y'), ('u', 'x'), ('u', 'y'), ('z', 'x')],
                          latents=['u'])
        self.assertEqual(model3.get_ivs('x', 'y'), {'z'})

        # The conditional algo shouldn't work in this case because the variable z is already
        # an IV and nearest separator doesn't work because the ancestral graph is disconnected.
        model4 = SEMGraph(ebunch=[('x', 'y'), ('z', 'x'), ('u', 'x'), ('u', 'y')])
        self.assertEqual(model4.get_conditional_ivs('x', 'y'), [])


class TESTSEMLISREL(unittest.TestCase):
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
        self.demo_lisrel = self.demo.to_lisrel()

