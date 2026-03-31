"""
Tests for base classes - BaseSimulator, BaseMetric, etc.
"""

import pytest
import networkx as nx
from pgmpy.benchmark.simulators.base import BaseSimulator, SimulationOutput
from pgmpy.benchmark.metrics.base import BaseMetric
import pandas as pd


class ConcreteSimulator(BaseSimulator):
    """Concrete implementation of BaseSimulator for testing."""
    
    def simulate(self, seed=None):
        """Simple implementation for testing."""
        dag = nx.DiGraph()
        dag.add_nodes_from(range(self.n_nodes))
        data = pd.DataFrame(
            [[0.0] * self.n_nodes for _ in range(self.n_samples)]
        )
        return SimulationOutput(dag=dag, data=data, metadata={})
    
    def get_name(self):
        """Return simulator name."""
        return "TestSimulator"


class ConcreteMetric(BaseMetric):
    """Concrete implementation of BaseMetric for testing."""
    
    def compute(self, pred_dag, true_dag):
        """Simple implementation for testing."""
        return 0.5


class TestSimulationOutput:
    """Test SimulationOutput dataclass."""
    
    def test_output_creation(self):
        """Test creating SimulationOutput."""
        dag = nx.DiGraph([(0, 1)])
        data = pd.DataFrame([[1, 2], [3, 4]])
        output = SimulationOutput(dag=dag, data=data, metadata={"key": "value"})
        
        assert output.dag.number_of_edges() == 1
        assert output.data.shape == (2, 2)
        assert output.metadata["key"] == "value"


class TestBaseSimulator:
    """Test BaseSimulator abstract base class."""
    
    def test_cannot_instantiate_abstract(self):
        """Test that BaseSimulator cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseSimulator(n_nodes=5, n_samples=10)
    
    def test_concrete_implementation(self):
        """Test concrete simulator implementation."""
        sim = ConcreteSimulator(n_nodes=5, n_samples=10)
        assert sim.n_nodes == 5
        assert sim.n_samples == 10
    
    def test_validate_n_nodes(self):
        """Test n_nodes validation."""
        with pytest.raises(ValueError):
            ConcreteSimulator(n_nodes=0, n_samples=10)
        
        with pytest.raises(ValueError):
            ConcreteSimulator(n_nodes=-1, n_samples=10)
    
    def test_validate_n_samples(self):
        """Test n_samples validation."""
        with pytest.raises(ValueError):
            ConcreteSimulator(n_nodes=5, n_samples=-1)
        
        # n_samples=0 should be OK
        sim = ConcreteSimulator(n_nodes=5, n_samples=0)
        assert sim.n_samples == 0
    
    def test_simulate_method(self):
        """Test that simulate method works."""
        sim = ConcreteSimulator(n_nodes=3, n_samples=5)
        output = sim.simulate(seed=42)
        
        assert isinstance(output, SimulationOutput)
        assert output.dag.number_of_nodes() == 3
        assert output.data.shape[0] == 5
    
    def test_get_name_method(self):
        """Test get_name method."""
        sim = ConcreteSimulator(n_nodes=5, n_samples=10)
        name = sim.get_name()
        assert name == "TestSimulator"


class TestBaseMetric:
    """Test BaseMetric abstract base class."""
    
    def test_cannot_instantiate_abstract(self):
        """Test that BaseMetric cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseMetric(name="test")
    
    def test_concrete_implementation(self):
        """Test concrete metric implementation."""
        metric = ConcreteMetric(name="TestMetric")
        assert metric.name == "TestMetric"
    
    def test_compute_method(self):
        """Test that compute method works."""
        metric = ConcreteMetric(name="TestMetric")
        
        dag1 = nx.DiGraph([(0, 1)])
        dag2 = nx.DiGraph([(0, 1)])
        
        result = metric.compute(dag1, dag2)
        assert result == 0.5
    
    def test_get_name_method(self):
        """Test get_name method."""
        metric = ConcreteMetric(name="TestMetric")
        assert metric.get_name() == "TestMetric"


class TestBaseSimulatorIntegration:
    """Integration tests for BaseSimulator."""
    
    def test_simulate_with_various_sizes(self):
        """Test simulator with various graph sizes."""
        for n_nodes in [1, 5, 10, 50]:
            sim = ConcreteSimulator(n_nodes=n_nodes, n_samples=100)
            output = sim.simulate()
            
            assert output.dag.number_of_nodes() == n_nodes
            assert output.data.shape[1] == n_nodes
    
    def test_simulate_with_various_samples(self):
        """Test simulator with various sample counts."""
        for n_samples in [0, 1, 10, 100]:
            sim = ConcreteSimulator(n_nodes=5, n_samples=n_samples)
            output = sim.simulate()
            
            assert output.data.shape[0] == n_samples


class TestBaseMetricIntegration:
    """Integration tests for BaseMetric."""
    
    def test_metric_with_various_graphs(self):
        """Test metric with various graph types."""
        metric = ConcreteMetric(name="TestMetric")
        
        # Empty graph
        g_empty = nx.DiGraph()
        g_empty.add_nodes_from(range(5))
        result = metric.compute(g_empty, g_empty)
        assert result == 0.5
        
        # Complete graph
        g_complete = nx.complete_graph(5, create_using=nx.DiGraph)
        result = metric.compute(g_complete, g_empty)
        assert result == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
