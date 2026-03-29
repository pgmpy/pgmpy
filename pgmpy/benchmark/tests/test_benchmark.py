"""
Unit tests for pgmpy.benchmark — Cognitive Benchmarking Framework.

Tests for simulators, metrics, runner, and semantic components.
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from pgmpy.benchmark.base import BaseSimulator, SimulationOutput
from pgmpy.benchmark.simulators import (
    ErdosRenyiSimulator,
    ScaleFreeSimulator,
    LinearGaussianSEM,
)
from pgmpy.benchmark.metrics import (
    SHDMetric,
    PrecisionRecallMetric,
    OrientationMetric,
    shd,
    precision_recall,
    orientation_f1,
    MetricsRegistry,
)
from pgmpy.benchmark.runner import BenchmarkRunner, BenchmarkRun, BenchmarkResults


class TestSimulators:
    """Test data simulators."""
    
    def test_erdos_renyi_simulator_basic(self):
        """Test ErdosRenyiSimulator produces valid output."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=100, seed=42)
        output = sim.simulate()
        
        assert isinstance(output, SimulationOutput)
        assert isinstance(output.dag, nx.DiGraph)
        assert isinstance(output.data, pd.DataFrame)
        assert output.data.shape[0] == 100  # n_samples
        assert output.data.shape[1] == 5    # n_nodes
        assert len(output.dag.nodes()) == 5
    
    def test_erdos_renyi_deterministic(self):
        """Test ErdosRenyiSimulator with fixed seed is deterministic."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=50, seed=42)
        output1 = sim.simulate()
        output2 = sim.simulate()
        
        # Data should be the same
        assert np.allclose(output1.data.values, output2.data.values)
    
    def test_erdos_renyi_params(self):
        """Test parameter retrieval."""
        sim = ErdosRenyiSimulator(n_nodes=10, edge_prob=0.2, n_samples=500)
        params = sim.get_params()
        
        assert params["n_nodes"] == 10
        assert params["edge_prob"] == 0.2
        assert params["n_samples"] == 500
    
    def test_scale_free_simulator(self):
        """Test ScaleFreeSimulator."""
        sim = ScaleFreeSimulator(n_nodes=5, alpha=1.0, n_samples=100, seed=42)
        output = sim.simulate()
        
        assert isinstance(output.dag, nx.DiGraph)
        assert isinstance(output.data, pd.DataFrame)
        assert output.data.shape == (100, 5)
    
    def test_linear_gaussian_sem(self):
        """Test LinearGaussianSEM."""
        sim = LinearGaussianSEM(n_nodes=5, n_samples=100, sparsity=0.3, seed=42)
        output = sim.simulate()
        
        assert isinstance(output.dag, nx.DiGraph)
        assert isinstance(output.data, pd.DataFrame)
        assert output.data.shape == (100, 5)


class TestMetrics:
    """Test metrics engine."""
    
    @pytest.fixture
    def simple_dags(self):
        """Create simple test DAGs."""
        # True DAG: A -> B -> C
        true_dag = nx.DiGraph()
        true_dag.add_edges_from([("A", "B"), ("B", "C")])
        
        # Estimated DAG: A -> B, B -> C, A -> C (extra edge)
        est_dag = nx.DiGraph()
        est_dag.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        
        return true_dag, est_dag
    
    def test_shd_metric(self, simple_dags):
        """Test SHD metric computation."""
        true_dag, est_dag = simple_dags
        
        metric = SHDMetric()
        result = metric.compute(est_dag, true_dag)
        
        assert result.name == "SHD"
        assert result.value == 1.0  # 1 extra edge
        assert result.metadata["extra"] == 1
        assert result.metadata["missing"] == 0
    
    def test_precision_recall_metric(self, simple_dags):
        """Test precision/recall metric."""
        true_dag, est_dag = simple_dags
        
        metric = PrecisionRecallMetric()
        result = metric.compute(est_dag, true_dag)
        
        assert result.name == "PrecisionRecall"
        assert result.metadata["precision"] == 2 / 3  # 2TP / 3est
        assert result.metadata["recall"] == 1.0        # 2TP / 2true
    
    def test_orientation_metric(self, simple_dags):
        """Test orientation F1 metric."""
        true_dag, est_dag = simple_dags
        
        metric = OrientationMetric()
        result = metric.compute(est_dag, true_dag)
        
        assert result.name == "OrientationF1"
        # Both (A-B) and (B-C) correctly oriented
        assert result.value == 1.0
    
    def test_shd_convenience_function(self, simple_dags):
        """Test SHD convenience function."""
        true_dag, est_dag = simple_dags
        result = shd(est_dag, true_dag)
        
        assert isinstance(result.value, float)
        assert result.value >= 0
    
    def test_metrics_registry(self):
        """Test MetricsRegistry."""
        registry = MetricsRegistry()
        
        # Check that default metrics are registered
        assert "shd" in registry.list_metrics()
        assert "precision_recall" in registry.list_metrics()
        
        # Retrieve a metric
        metric = registry.get("shd")
        assert isinstance(metric, SHDMetric)


class TestBenchmarkRunner:
    """Test BenchmarkRunner orchestrator."""
    
    @pytest.fixture
    def mock_method(self):
        """Simple mock method for testing."""
        class MockMethod:
            def __call__(self, data):
                # Return a simple DAG based on data
                dag = nx.DiGraph()
                dag.add_edges_from([("X0", "X1"), ("X1", "X2")])
                return dag
        
        return MockMethod()
    
    def test_benchmark_runner_basic(self, mock_method):
        """Test BenchmarkRunner with basic configuration."""
        sim = ErdosRenyiSimulator(n_nodes=3, edge_prob=0.3, n_samples=50, seed=42)
        
        runner = BenchmarkRunner(
            simulators=[sim],
            methods=[mock_method],
            metrics=[shd, precision_recall],
            n_runs=2,
            n_jobs=1,
            verbose=0,
        )
        
        results = runner.run()
        
        assert isinstance(results, BenchmarkResults)
        assert len(results.runs) == 2  # 1 simulator * 1 method * 2 runs
        assert all(isinstance(run, BenchmarkRun) for run in results.runs)
    
    def test_benchmark_run_structure(self, mock_method):
        """Test structure of individual BenchmarkRun."""
        sim = ErdosRenyiSimulator(n_nodes=3, edge_prob=0.3, n_samples=50, seed=42)
        
        runner = BenchmarkRunner(
            simulators=[sim],
            methods=[mock_method],
            metrics=[shd],
            n_runs=1,
            n_jobs=1,
        )
        
        results = runner.run()
        run = results.runs[0]
        
        assert run.run_id == 0
        assert "MockMethod" in run.method_name
        assert "ErdosRenyi" in run.simulator_name
        assert "SHD" in run.metrics
        assert run.execution_time >= 0
    
    def test_benchmark_summary(self, mock_method):
        """Test results summary generation."""
        sim = ErdosRenyiSimulator(n_nodes=3, edge_prob=0.3, n_samples=50, seed=42)
        
        runner = BenchmarkRunner(
            simulators=[sim],
            methods=[mock_method],
            metrics=[shd],
            n_runs=2,
            n_jobs=1,
        )
        
        results = runner.run()
        summary = results.summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1  # 1 (simulator, method) pair
        assert "n_runs" in summary.columns
        assert summary.iloc[0]["n_runs"] == 2
    
    def test_benchmark_export_json(self, tmp_path, mock_method):
        """Test JSON export."""
        sim = ErdosRenyiSimulator(n_nodes=3, edge_prob=0.3, n_samples=50, seed=42)
        
        runner = BenchmarkRunner(
            simulators=[sim],
            methods=[mock_method],
            metrics=[shd],
            n_runs=1,
            n_jobs=1,
        )
        
        results = runner.run()
        json_file = tmp_path / "results.json"
        results.to_json(str(json_file))
        
        assert json_file.exists()
        
        import json
        with open(json_file) as f:
            data = json.load(f)
        
        assert "runs" in data
        assert len(data["runs"]) == 1


class TestIntegration:
    """Integration tests."""
    
    def test_full_benchmark_pipeline(self):
        """Test complete benchmark pipeline."""
        # Simple mock method
        class SimpleMethod:
            def __call__(self, data):
                dag = nx.DiGraph()
                dag.add_edges_from([("X0", "X1")])
                return dag
        
        simple_method = SimpleMethod()
        
        # Define benchmark
        runner = BenchmarkRunner(
            simulators=[
                ErdosRenyiSimulator(n_nodes=4, edge_prob=0.3, n_samples=100, seed=42),
            ],
            methods=[simple_method],
            metrics=[shd, precision_recall, orientation_f1],
            n_runs=2,
            n_jobs=1,
            verbose=0,
        )
        
        # Run benchmark
        results = runner.run()
        
        # Check results
        assert len(results.runs) == 2
        assert all(isinstance(r, BenchmarkRun) for r in results.runs)
        
        # Check summary
        summary = results.summary()
        assert len(summary) > 0
        assert "SHD_mean" in summary.columns or "SHD" in summary.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
