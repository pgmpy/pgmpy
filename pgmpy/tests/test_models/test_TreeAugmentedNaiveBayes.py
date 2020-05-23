import unittest

import pandas as pd

from pgmpy.models import TreeAugmentedNaiveBayes

class TestTreeAugmentedNaiveBayes(unittest.TestCase):
    def setUp(self):
        self.model1 = TreeAugmentedNaiveBayes()

    def test_learn_structure(self):
        values = pd.DataFrame(
            data = [[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]],
            columns=['A', 'B', 'C', 'D']
        )

        self.model1.learn_structure(values, class_node='A', root_node='D')

        self.assertCountEqual(self.model1.nodes(), ["A", "B", "C", "D"])
        self.assertCountEqual(
            self.model1.edges(), [("A", "B"), ("A", "C"), ("A", "D"), ("D", "B"), ("D", "C")]
        )
        self.assertTrue(self.model1.has_edge("A", "B"))
        self.assertTrue(self.model1.has_edge("A", "C"))
        self.assertTrue(self.model1.has_edge("A", "D"))
        self.assertTrue(self.model1.has_edge("D", "B"))
        self.assertTrue(self.model1.has_edge("D", "C"))

    def tearDown(self):
        del self.model1
