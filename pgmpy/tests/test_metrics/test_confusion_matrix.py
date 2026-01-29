import unittest

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import ConfusionMatrix


class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.true_dag = DAG([('A', 'B'), ('B', 'C'), ('A', 'C')])
        self.est_dag = DAG([('A', 'B'), ('C', 'B')])
        self.perfect_dag = DAG([('A', 'B'), ('B', 'C'), ('A', 'C')])
        self.empty_dag = DAG()
        self.empty_dag.add_nodes_from(['A', 'B', 'C'])

    def test_basic_adjacency_metrics(self):
        """Test basic adjacency metrics computation."""
        cm = ConfusionMatrix()
        result = cm.evaluate(self.true_dag, self.est_dag)

        self.assertIn('adjacency_precision', result)
        self.assertIn('adjacency_recall', result)
        self.assertIn('adjacency_f1', result)
        self.assertIn('adjacency_npv', result)
        self.assertIn('adjacency_specificity', result)
        self.assertIn('adjacency_confusion_matrix', result)

        adj_cm = result['adjacency_confusion_matrix']
        for key in ['tp', 'fp', 'fn', 'tn']:
            self.assertGreaterEqual(adj_cm[key], 0)
            self.assertIsInstance(adj_cm[key], int)

        for metric in ['adjacency_precision', 'adjacency_recall', 'adjacency_f1',
                      'adjacency_npv', 'adjacency_specificity']:
            self.assertGreaterEqual(result[metric], 0.0)
            self.assertLessEqual(result[metric], 1.0)

    def test_perfect_match(self):
        """Test metrics when estimated graph perfectly matches true graph."""
        cm = ConfusionMatrix()
        result = cm.evaluate(self.true_dag, self.perfect_dag)

        self.assertEqual(result['adjacency_precision'], 1.0)
        self.assertEqual(result['adjacency_recall'], 1.0)
        self.assertEqual(result['adjacency_f1'], 1.0)

        adj_cm = result['adjacency_confusion_matrix']
        self.assertEqual(adj_cm['fp'], 0)
        self.assertEqual(adj_cm['fn'], 0)

    def test_empty_graphs(self):
        """Test metrics with empty graphs."""
        cm = ConfusionMatrix()
        result = cm.evaluate(self.empty_dag, self.empty_dag)

        adj_cm = result['adjacency_confusion_matrix']
        self.assertEqual(adj_cm['tp'], 0)
        self.assertEqual(adj_cm['fp'], 0)
        self.assertEqual(adj_cm['fn'], 0)

        n_nodes = len(self.empty_dag.nodes())
        max_edges = n_nodes * (n_nodes - 1) // 2
        self.assertEqual(adj_cm['tn'], max_edges)

    def test_orientation_metrics(self):
        """Test orientation metrics for DAGs."""
        cm = ConfusionMatrix()
        result = cm.evaluate(self.true_dag, self.est_dag)

        self.assertIn('orientation_precision', result)
        self.assertIn('orientation_recall', result)
        self.assertIn('orientation_confusion_matrix', result)

        self.assertGreaterEqual(result['orientation_precision'], 0.0)
        self.assertLessEqual(result['orientation_precision'], 1.0)
        self.assertGreaterEqual(result['orientation_recall'], 0.0)
        self.assertLessEqual(result['orientation_recall'], 1.0)

    def test_selective_metrics(self):
        """Test computation of selective metrics."""
        cm = ConfusionMatrix(metrics=['precision', 'recall'])
        result = cm.evaluate(self.true_dag, self.est_dag)

        self.assertIn('adjacency_precision', result)
        self.assertIn('adjacency_recall', result)
        self.assertNotIn('adjacency_f1', result)
        self.assertNotIn('adjacency_npv', result)
        self.assertNotIn('adjacency_specificity', result)

    def test_pdag_support(self):
        """Test that PDAGs are supported."""
        pdag = PDAG()
        pdag.add_nodes_from(['A', 'B', 'C'])
        pdag.add_edges_from([('A', 'B'), ('B', 'C')])

        cm = ConfusionMatrix()
        result = cm.evaluate(pdag, pdag)

        self.assertIn('adjacency_precision', result)
        self.assertIn('adjacency_recall', result)
        self.assertNotIn('orientation_precision', result)

    def test_different_nodes_error(self):
        """Test error when graphs have different nodes."""
        other_dag = DAG([('X', 'Y')])
        cm = ConfusionMatrix()

        with self.assertRaises(ValueError):
            cm.evaluate(self.true_dag, other_dag)

    def test_edge_case_empty_estimated(self):
        """Test edge case where estimated graph is empty."""
        cm = ConfusionMatrix()
        result = cm.evaluate(self.true_dag, self.empty_dag)

        self.assertEqual(result['adjacency_precision'], 0.0)
        self.assertEqual(result['adjacency_recall'], 0.0)

    def test_confusion_matrix_values(self):
        """Test specific confusion matrix values for known graphs."""
        cm = ConfusionMatrix()
        result = cm.evaluate(self.true_dag, self.est_dag)

        adj_cm = result['adjacency_confusion_matrix']

        self.assertEqual(adj_cm['tp'], 2)
        self.assertEqual(adj_cm['fn'], 1)
        self.assertEqual(adj_cm['fp'], 0)


if __name__ == '__main__':
    unittest.main()
