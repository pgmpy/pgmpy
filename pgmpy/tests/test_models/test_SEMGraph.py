import unittest

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
