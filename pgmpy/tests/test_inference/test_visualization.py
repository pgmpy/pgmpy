import unittest
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.base import DAG
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.inference.visualization import plot_causal_graph


class TestVisualization(unittest.TestCase):
    def setUp(self):
        # Create a simple DAG
        self.dag = DAG([("X", "Y"), ("Z", "X"), ("Z", "Y")])
        
        # Create a Bayesian Network
        self.bn = DiscreteBayesianNetwork([("X", "Y"), ("Z", "X"), ("Z", "Y")])
        
        # Create a CausalInference object
        self.causal_inference = CausalInference(self.dag)
        
        # Create a temporary directory for output files
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
        plt.close('all')
    
    def test_plot_causal_graph_with_dag(self):
        """Test plotting with a DAG object"""
        output_file = os.path.join(self.test_dir, "dag_plot.png")
        ax = plot_causal_graph(self.dag, output_file=output_file)
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue(os.path.exists(output_file))
    
    def test_plot_causal_graph_with_bn(self):
        """Test plotting with a BayesianNetwork object"""
        output_file = os.path.join(self.test_dir, "bn_plot.png")
        ax = plot_causal_graph(self.bn, output_file=output_file)
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue(os.path.exists(output_file))
    
    def test_plot_causal_graph_with_causal_inference(self):
        """Test plotting with a CausalInference object"""
        output_file = os.path.join(self.test_dir, "causal_inference_plot.png")
        ax = plot_causal_graph(self.causal_inference, output_file=output_file)
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue(os.path.exists(output_file))
    
    def test_plot_causal_graph_with_custom_positions(self):
        """Test plotting with custom node positions"""
        pos = {"X": (0, 0), "Y": (1, 0), "Z": (0.5, 0.5)}
        output_file = os.path.join(self.test_dir, "custom_pos_plot.png")
        ax = plot_causal_graph(self.dag, node_pos=pos, output_file=output_file)
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue(os.path.exists(output_file))
    
    def test_plot_causal_graph_with_different_layouts(self):
        """Test plotting with different layout algorithms"""
        layouts = ["circular", "kamada_kawai", "shell", "spring"]
        
        for layout in layouts:
            output_file = os.path.join(self.test_dir, f"{layout}_plot.png")
            ax = plot_causal_graph(self.dag, node_pos=layout, output_file=output_file)
            self.assertIsInstance(ax, plt.Axes)
            self.assertTrue(os.path.exists(output_file))
    
    def test_plot_causal_graph_with_custom_styling(self):
        """Test plotting with custom styling options"""
        output_file = os.path.join(self.test_dir, "custom_style_plot.png")
        ax = plot_causal_graph(
            self.dag,
            node_color="lightgreen",
            node_size=800,
            font_size=12,
            edge_color="red",
            arrowsize=30,
            width=2.0,
            output_file=output_file
        )
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue(os.path.exists(output_file))
    
    def test_plot_causal_graph_invalid_model(self):
        """Test plotting with an invalid model type"""
        with self.assertRaises(TypeError):
            plot_causal_graph({"X": "Y"})  # Dictionary is not a valid model
    
    def test_plot_causal_graph_invalid_layout(self):
        """Test plotting with an invalid layout name"""
        with self.assertRaises(ValueError):
            plot_causal_graph(self.dag, node_pos="invalid_layout")


if __name__ == "__main__":
    unittest.main()