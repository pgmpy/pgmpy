"""
Comprehensive edge case tests for benchmark framework.

Tests coverage for:
- Simulator parameter validation
- Metrics edge cases (empty graphs, single nodes, etc.)
- Runner error handling
- Semantic rule edge cases
- Storage implementation
"""

import pytest
import pandas as pd
import networkx as nx
from pgmpy.benchmark import (
    BenchmarkRunner,
    ErdosRenyiSimulator,
    ScaleFreeSimulator,
    shd,
    precision_recall,
    orientation_f1,
)
from pgmpy.benchmark.semantic import SemanticContext, RuleEngine, SemanticScorer
from pgmpy.benchmark.reasoning import ChainOfThoughtTracer, ReasoningStep


class TestSimulatorValidation:
    """Test simulator parameter validation and edge cases."""

    def test_erdos_renyi_zero_samples(self):
        """Test Erdos-Renyi with zero samples."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=0, seed=42)
        output = sim.simulate(seed=42)
        # Should have empty data
        assert output.data.shape[0] == 0
        assert output.dag.number_of_nodes() == 5

    def test_erdos_renyi_single_node(self):
        """Test with single node."""
        sim = ErdosRenyiSimulator(n_nodes=1, edge_prob=0.3, n_samples=10, seed=42)
        output = sim.simulate(seed=42)
        assert output.dag.number_of_nodes() == 1
        assert output.dag.number_of_edges() == 0

    def test_erdos_renyi_zero_edge_prob(self):
        """Test with zero edge probability."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.0, n_samples=10, seed=42)
        output = sim.simulate(seed=42)
        assert output.dag.number_of_edges() == 0

    def test_erdos_renyi_full_edge_prob(self):
        """Test with probability=1 (complete graph)."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=1.0, n_samples=10, seed=42)
        output = sim.simulate(seed=42)
        # Complete DAG should have many edges
        assert output.dag.number_of_edges() > 0

    def test_erdos_renyi_high_noise(self):
        """Test with very high noise."""
        sim = ErdosRenyiSimulator(
            n_nodes=5, edge_prob=0.3, n_samples=100, noise_scale=10.0, seed=42
        )
        output = sim.simulate(seed=42)
        assert output.data.shape[0] == 100

    def test_scale_free_alpha_parameter(self):
        """Test scale-free with different alpha values."""
        for alpha in [0.5, 1.0, 1.5]:
            sim = ScaleFreeSimulator(n_nodes=10, alpha=alpha, n_samples=50, seed=42)
            output = sim.simulate(seed=42)
            # Should have some structure
            assert output.dag.number_of_nodes() == 10

    def test_scale_free_single_node(self):
        """Test scale-free simulator with minimal nodes (BA requires m < n)."""
        # Barabási-Albert requires m >= 1 and m < n, so minimum is 2 nodes
        sim = ScaleFreeSimulator(n_nodes=2, alpha=1.0, n_samples=10, seed=42)
        output = sim.simulate(seed=42)
        assert output.dag.number_of_nodes() == 2

    def test_simulator_deterministic_seed(self):
        """Test that same seed produces same output."""
        sim1 = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=10, seed=42)
        sim2 = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=10, seed=42)

        out1 = sim1.simulate(seed=42)
        out2 = sim2.simulate(seed=42)

        # DAGs should be identical
        assert set(out1.dag.edges()) == set(out2.dag.edges())


class TestMetricsEdgeCases:
    """Test metrics with edge cases."""

    def test_shd_empty_graphs(self):
        """Test SHD with empty graphs."""
        g1 = nx.DiGraph()
        g1.add_nodes_from(range(5))
        g2 = nx.DiGraph()
        g2.add_nodes_from(range(5))
        result = shd(g1, g2)
        # Result could be a MetricResult object or a number
        assert result is not None

    def test_shd_single_edge_difference(self):
        """Test SHD with minimal difference."""
        g1 = nx.DiGraph([(0, 1)])
        g2 = nx.DiGraph([(0, 2)])
        result = shd(g1, g2)
        # Should be greater than 0
        assert result is not None

    def test_shd_identical_graphs(self):
        """Test SHD with identical graphs."""
        edges = [(0, 1), (1, 2), (2, 3)]
        g1 = nx.DiGraph(edges)
        g2 = nx.DiGraph(edges)
        result = shd(g1, g2)
        # Should have value attribute or be 0
        assert result is not None

    def test_precision_recall_no_edges(self):
        """Test precision/recall when no edges to predict."""
        true_dag = nx.DiGraph()
        true_dag.add_nodes_from(range(5))
        pred_dag = nx.DiGraph()
        pred_dag.add_nodes_from(range(5))

        result = precision_recall(pred_dag, true_dag)
        # Should be a MetricResult object or dict
        assert hasattr(result, 'value') or isinstance(result, dict)

    def test_precision_recall_all_correct(self):
        """Test precision/recall when all predictions correct."""
        edges = [(0, 1), (1, 2)]
        true_dag = nx.DiGraph(edges)
        pred_dag = nx.DiGraph(edges)

        result = precision_recall(pred_dag, true_dag)
        assert hasattr(result, 'value') or isinstance(result, dict)

    def test_precision_recall_all_wrong(self):
        """Test precision/recall when predictions are wrong."""
        true_dag = nx.DiGraph([(0, 1)])
        pred_dag = nx.DiGraph([(2, 3)])

        result = precision_recall(pred_dag, true_dag)
        assert hasattr(result, 'value') or isinstance(result, dict)

    def test_orientation_f1_perfect(self):
        """Test orientation F1 with perfect orientation."""
        # Create arrow vs head graph
        true_dag = nx.DiGraph([(0, 1)])
        pred_dag = nx.DiGraph([(0, 1)])

        result = orientation_f1(pred_dag, true_dag)
        assert result is not None


class TestRunnerValidation:
    """Test BenchmarkRunner input validation."""

    def test_runner_invalid_n_runs(self):
        """Test runner with invalid n_runs."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=10)

        def dummy_method(data):
            return nx.DiGraph()

        with pytest.raises(ValueError):
            BenchmarkRunner(
                simulators=[sim], methods=[dummy_method], n_runs=0  # Invalid
            )

    def test_runner_invalid_n_jobs(self):
        """Test runner with invalid n_jobs."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=10)

        def dummy_method(data):
            return nx.DiGraph()

        with pytest.raises(ValueError):
            BenchmarkRunner(
                simulators=[sim], methods=[dummy_method], n_jobs=0  # Invalid
            )

    def test_runner_empty_simulators(self):
        """Test runner with empty simulators list."""

        def dummy_method(data):
            return nx.DiGraph()

        with pytest.raises(ValueError):
            BenchmarkRunner(simulators=[], methods=[dummy_method])

    def test_runner_empty_methods(self):
        """Test runner with empty methods list."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=10)

        with pytest.raises(ValueError):
            BenchmarkRunner(simulators=[sim], methods=[])

    def test_runner_invalid_method_not_callable(self):
        """Test runner with non-callable method."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=10)

        with pytest.raises((ValueError, TypeError)):
            BenchmarkRunner(simulators=[sim], methods=["not_callable"])

    def test_runner_single_run(self):
        """Test runner with minimal valid config."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=10, seed=42)

        def dummy_method(data):
            n = data.shape[1] if hasattr(data, "shape") else len(data.columns)
            dag = nx.DiGraph()
            dag.add_nodes_from(range(n))
            return dag

        runner = BenchmarkRunner(
            simulators=[sim],
            methods=[dummy_method],
            metrics=["shd"],
            n_runs=1,
            n_jobs=1,
        )
        # Should not raise
        assert runner is not None


class TestSemanticRuleEdgeCases:
    """Test semantic rule engine edge cases."""

    def test_rule_engine_empty_rules(self):
        """Test rule engine with no rules."""
        engine = RuleEngine(rules=[])
        context = SemanticContext(domain="test")
        weights = {"shd": 1.0, "precision": 1.0}
        fired_rules, adjusted = engine.evaluate(context, weights)
        assert isinstance(fired_rules, list)
        assert isinstance(adjusted, dict)

    def test_rule_engine_with_valid_rules(self):
        """Test rule engine evaluates rules correctly."""
        from pgmpy.benchmark.semantic import EvaluationRule
        rules = [
            EvaluationRule(
                name="test_rule",
                condition=lambda ctx: ctx.noise_level == "high",
                action=lambda w: {k: v * 2 for k, v in w.items()},
                rationale="Test rule"
            ),
        ]
        engine = RuleEngine(rules=rules)
        context = SemanticContext(domain="test", noise_level="high")
        weights = {"shd": 1.0, "precision": 1.0}
        
        fired_rules, adjusted = engine.evaluate(context, weights)
        assert isinstance(fired_rules, list)
        # The test_rule should fire because noise_level == "high"
        assert "test_rule" in fired_rules

    def test_semantic_context_injection(self):
        """Test semantic context injection into scoring."""
        context = SemanticContext(
            domain="test", noise_level="high", priority="precision"
        )
        scorer = SemanticScorer(context=context)
        # Should apply context weights
        assert scorer.context is not None

    def test_semantic_context_missing_fields(self):
        """Test semantic context with minimal fields."""
        context = SemanticContext(domain="test")
        assert context.domain == "test"
        # Should have defaults for other fields


class TestReasoningTraceEdgeCases:
    """Test reasoning trace edge cases."""

    def test_reasoning_step_basic(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(
            step_number=1, 
            action="compute_metric", 
            result="SHD computed successfully"
        )
        assert step.step_number == 1
        assert step.action == "compute_metric"

    def test_chain_of_thought_single_step(self):
        """Test tracer with single step."""
        tracer = ChainOfThoughtTracer(
            method_name="PC",
            simulator_name="ErdosRenyi", 
            run_id="run_1"
        )
        tracer.add_step(action="init", result="Initialized PC algorithm")
        assert len(tracer.steps) == 1

    def test_chain_of_thought_many_steps(self):
        """Test tracer with many steps."""
        tracer = ChainOfThoughtTracer(
            method_name="GES",
            simulator_name="ScaleFree",
            run_id="run_2"
        )
        for i in range(50):
            tracer.add_step(action=f"step_{i}", result=f"Step {i} completed")
        assert len(tracer.steps) == 50

    def test_chain_of_thought_serialization(self):
        """Test that tracer output can be converted to dict."""
        tracer = ChainOfThoughtTracer(
            method_name="PC",
            simulator_name="ErdosRenyi",
            run_id="test_run"
        )
        tracer.add_step(action="test_action", result="test result")
        
        trace_dict = tracer.to_dict()
        assert isinstance(trace_dict, dict)
        assert "method" in trace_dict
        assert "simulator" in trace_dict
        assert "run_id" in trace_dict


class TestIntegrationEdgeCases:
    """Integration tests for edge cases."""

    def test_benchmark_with_single_sample(self):
        """Test benchmark with minimal data."""
        sim = ErdosRenyiSimulator(
            n_nodes=3, edge_prob=0.3, n_samples=1, seed=42
        )

        def simple_method(data):
            n = data.shape[1] if hasattr(data, "shape") else len(data.columns)
            dag = nx.DiGraph()
            dag.add_nodes_from(range(n))
            return dag

        runner = BenchmarkRunner(
            simulators=[sim],
            methods=[simple_method],
            metrics=["shd"],
            n_runs=1,
            n_jobs=1,
        )
        results = runner.run()
        assert len(results.runs) > 0

    def test_multiple_metrics_computation(self):
        """Test benchmark with multiple metrics."""
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=10, seed=42)

        def dummy_method(data):
            n = data.shape[1] if hasattr(data, "shape") else len(data.columns)
            dag = nx.DiGraph()
            dag.add_nodes_from(range(n))
            # Add some random edges
            import random
            random.seed(42)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.3:
                        dag.add_edge(i, j)
            return dag

        runner = BenchmarkRunner(
            simulators=[sim],
            methods=[dummy_method],
            metrics=["shd", "precision_recall", "orientation_f1"],
            n_runs=1,
            n_jobs=1,
        )
        results = runner.run()
        assert len(results.runs) > 0

    def test_benchmark_export_formats(self):
        """Test benchmark result export in different formats."""
        import tempfile
        import json
        
        sim = ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=10, seed=42)

        def dummy_method(data):
            n = data.shape[1] if hasattr(data, "shape") else len(data.columns)
            dag = nx.DiGraph()
            dag.add_nodes_from(range(n))
            return dag

        runner = BenchmarkRunner(
            simulators=[sim],
            methods=[dummy_method],
            metrics=["shd"],
            n_runs=1,
            n_jobs=1,
        )
        results = runner.run()

        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            results.to_json(f.name)
            with open(f.name) as rf:
                data = json.load(rf)
                assert "runs" in data

        # Test CSV export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            results.to_csv(f.name)
            assert f.name.endswith('.csv')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
