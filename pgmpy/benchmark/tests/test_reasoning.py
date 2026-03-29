"""
Tests for reasoning layer (Phase 3).

Tests for:
    - ChainOfThoughtTracer: Step-by-step reasoning logs
    - ReasoningStep: Individual reasoning steps
    - Explanation: Human-readable explanations
"""

import pytest
import json
from typing import Dict, Any, List

from pgmpy.benchmark.reasoning import (
    ChainOfThoughtTracer,
    ReasoningStep,
    Explanation,
)


class TestReasoningStep:
    """Test ReasoningStep functionality."""
    
    def test_reasoning_step_creation(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(
            step_number=1,
            action="run_algorithm",
            result="DAG estimated in 0.43s",
            metadata={"method": "PC", "dataset_size": 500},
        )
        
        assert step.step_number == 1
        assert step.action == "run_algorithm"
        assert step.result == "DAG estimated in 0.43s"
        assert step.metadata["method"] == "PC"
    
    def test_reasoning_step_to_dict(self):
        """Test step serialization."""
        step = ReasoningStep(
            step_number=2,
            action="compute_metric",
            result="SHD = 4",
            metadata={"metric": "shd"},
        )
        
        step_dict = step.to_dict()
        
        assert step_dict["step"] == 2
        assert step_dict["action"] == "compute_metric"
        assert step_dict["result"] == "SHD = 4"


class TestChainOfThoughtTracer:
    """Test ChainOfThoughtTracer functionality."""
    
    def test_tracer_creation(self):
        """Test creating a tracer."""
        tracer = ChainOfThoughtTracer(
            method_name="PC",
            simulator_name="ErdosRenyi",
            run_id="test_001",
        )
        
        assert tracer.method_name == "PC"
        assert tracer.simulator_name == "ErdosRenyi"
        assert tracer.run_id == "test_001"
        assert len(tracer.steps) == 0
    
    def test_tracer_add_step(self):
        """Test adding steps to tracer."""
        tracer = ChainOfThoughtTracer(
            method_name="PC",
            simulator_name="ErdosRenyi",
            run_id="test_001",
        )
        
        tracer.add_step(
            action="run_algorithm",
            result="DAG estimated",
            metadata={"time": 0.43},
        )
        
        assert len(tracer.steps) == 1
        assert tracer.steps[0].action == "run_algorithm"
    
    def test_tracer_multiple_steps(self):
        """Test adding multiple steps."""
        tracer = ChainOfThoughtTracer(
            method_name="GES",
            simulator_name="ScaleFree",
            run_id="test_002",
        )
        
        # Add multiple steps in sequence
        tracer.add_step("run_algorithm", "DAG estimated", {})
        tracer.add_step("compute_metric", "SHD = 3", {"metric": "shd"})
        tracer.add_step("compute_metric", "F1 = 0.82", {"metric": "f1"})
        tracer.add_step("interpretation", "Good performance", {})
        
        assert len(tracer.steps) == 4
        assert tracer.steps[0].step_number == 1
        assert tracer.steps[1].step_number == 2
        assert tracer.steps[3].step_number == 4
    
    def test_tracer_to_json_format(self):
        """Test tracer JSON serialization."""
        tracer = ChainOfThoughtTracer(
            method_name="HillClimbSearch",
            simulator_name="ErdosRenyi",
            run_id="test_003",
        )
        
        tracer.add_step("run_algorithm", "Running...", {})
        tracer.add_step("compute_metric", "Complete", {"time": 1.5})
        
        json_repr = tracer.to_json()
        
        # Parse JSON
        data = json.loads(json_repr)
        
        assert data["method"] == "HillClimbSearch"
        assert data["simulator"] == "ErdosRenyi"
        assert len(data["steps"]) == 2
        assert data["steps"][0]["step"] == 1
    
    def test_tracer_get_trace_string(self):
        """Test human-readable trace generation."""
        tracer = ChainOfThoughtTracer(
            method_name="PC",
            simulator_name="ErdosRenyi",
            run_id="test_004",
        )
        
        tracer.add_step("run_algorithm", "Estimating DAG", {})
        tracer.add_step("compute_metric", "SHD = 2", {"metric": "shd"})
        tracer.add_step("interpretation", "Excellent", {"quality": "high"})
        
        trace_str = tracer.get_trace_string()
        
        assert isinstance(trace_str, str)
        assert "PC" in trace_str
        assert "ErdosRenyi" in trace_str
        assert "Estimating DAG" in trace_str
        assert "SHD = 2" in trace_str


class TestExplanation:
    """Test Explanation functionality."""
    
    def test_explanation_creation(self):
        """Test creating an explanation."""
        explanation = Explanation(
            run_id="test_run",
            method_name="NOTEARS",
            composite_score=0.85,
            semantic_context={"domain": "financial", "noise": "high"},
            fired_rules=["precision_priority", "large_graph_runtime"],
            component_scores={"shd": 0.8, "precision": 0.9},
        )
        
        assert explanation.run_id == "test_run"
        assert explanation.composite_score == 0.85
        assert len(explanation.fired_rules) == 2
    
    def test_explanation_natural_language(self):
        """Test generating natural language explanation."""
        explanation = Explanation(
            run_id="test_run",
            method_name="DirectLiNGAM",
            composite_score=0.75,
            semantic_context={"domain": "biological", "noise": "medium"},
            fired_rules=["biological_network_orientation"],
            component_scores={"shd": 0.7, "f1": 0.8, "orientation": 0.75},
        )
        
        narrative = explanation.to_narrative()
        
        assert isinstance(narrative, str)
        assert "DirectLiNGAM" in narrative
        assert "0.75" in narrative or "75" in narrative
        assert len(narrative) > 50  # Should be reasonably detailed


class TestIntegration:
    """Integration tests for reasoning layer."""
    
    def test_tracer_with_benchmark_run(self):
        """Test tracer integrated with benchmark run."""
        from pgmpy.benchmark import BenchmarkRunner, ErdosRenyiSimulator
        
        class TestMethod:
            def __call__(self, data):
                import networkx as nx
                dag = nx.DiGraph()
                dag.add_edge("X0", "X1")
                return dag
        
        runner = BenchmarkRunner(
            simulators=[ErdosRenyiSimulator(n_nodes=3, edge_prob=0.3, n_samples=50, seed=42)],
            methods=[TestMethod()],
            metrics=["shd"],
            n_runs=1,
            n_jobs=1,
        )
        
        results = runner.run()
        
        # Verify results have chain-of-thought (future integration)
        assert len(results.runs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
