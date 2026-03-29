"""
Tests for semantic evaluation layer (Phase 2).

Tests for:
    - SemanticContext: Domain context attachment
    - RuleEngine: Rule-based evaluation
    - SemanticScorer: Context-adjusted scoring
"""

import pytest
import json
from typing import Dict, Any

from pgmpy.benchmark.semantic import (
    SemanticContext,
    RuleEngine,
    SemanticScorer,
    EvaluationRule,
)


class TestSemanticContext:
    """Test SemanticContext functionality."""
    
    def test_context_injection_basic(self):
        """Test basic context creation."""
        context = SemanticContext(
            domain="biological",
            noise_level="high",
            graph_size="large",
            graph_type="scale_free",
            priority="precision",
        )
        
        assert context.domain == "biological"
        assert context.noise_level == "high"
        assert context.graph_size == "large"
        assert context.priority == "precision"
    
    def test_context_to_dict(self):
        """Test context serialization."""
        context = SemanticContext(
            domain="financial",
            noise_level="medium",
            graph_size="medium",
            graph_type="erdos_renyi",
            priority="balanced",
        )
        
        ctx_dict = context.to_dict()
        
        assert isinstance(ctx_dict, dict)
        assert ctx_dict["domain"] == "financial"
        assert ctx_dict["noise_level"] == "medium"
    
    def test_context_from_dict(self):
        """Test context deserialization."""
        data = {
            "domain": "synthetic",
            "noise_level": "low",
            "graph_size": "small",
            "graph_type": "real_bn",
            "priority": "recall",
        }
        
        context = SemanticContext.from_dict(data)
        
        assert context.domain == "synthetic"
        assert context.noise_level == "low"
        assert context.priority == "recall"


class TestRuleEngine:
    """Test RuleEngine functionality."""
    
    def test_rule_evaluation_basic(self):
        """Test basic rule evaluation."""
        context = SemanticContext(
            domain="biological",
            noise_level="high",
            graph_size="large",
            graph_type="scale_free",
            priority="precision",
        )
        
        rule = EvaluationRule(
            name="high_noise_robustness",
            condition=lambda ctx: ctx.noise_level == "high",
            action=lambda w: {**w, "precision": w.get("precision", 1.0) * 1.4},
            rationale="In high-noise regimes, precision matters more",
        )
        
        # Check condition
        assert rule.evaluate_condition(context) is True
        
        # Check action on weights
        weights = {"shd": 1.0, "precision": 1.0}
        updated = rule.execute_action(weights)
        
        assert updated["precision"] == 1.4
        assert updated["shd"] == 1.0
    
    def test_rule_engine_multiple_rules(self):
        """Test RuleEngine with multiple rules."""
        context = SemanticContext(
            domain="biological",
            noise_level="high",
            graph_size="large",
            graph_type="scale_free",
            priority="precision",
        )
        
        rules = [
            EvaluationRule(
                name="high_noise_precision",
                condition=lambda ctx: ctx.noise_level == "high",
                action=lambda w: {**w, "precision": w.get("precision", 1.0) * 1.4},
                rationale="High noise → precision matters",
            ),
            EvaluationRule(
                name="large_graph_runtime",
                condition=lambda ctx: ctx.graph_size == "large",
                action=lambda w: {**w, "runtime": w.get("runtime", 1.0) * 1.5},
                rationale="Large graphs → runtime critical",
            ),
        ]
        
        engine = RuleEngine(rules)
        
        # Initial weights
        weights = {"shd": 1.0, "precision": 1.0, "runtime": 1.0}
        
        # Apply all rules
        fired_rules, final_weights = engine.evaluate(context, weights)
        
        assert len(fired_rules) == 2
        assert final_weights["precision"] == 1.4
        assert final_weights["runtime"] == 1.5
        assert final_weights["shd"] == 1.0  # unchanged
    
    def test_rule_engine_no_matching_rules(self):
        """Test RuleEngine when no rules match."""
        context = SemanticContext(
            domain="financial",
            noise_level="low",
            graph_size="small",
            graph_type="erdos_renyi",
            priority="balanced",
        )
        
        rules = [
            EvaluationRule(
                name="high_noise_only",
                condition=lambda ctx: ctx.noise_level == "high",
                action=lambda w: {**w, "precision": w.get("precision", 1.0) * 1.4},
                rationale="High noise rule",
            ),
        ]
        
        engine = RuleEngine(rules)
        weights = {"shd": 1.0, "precision": 1.0}
        
        fired_rules, final_weights = engine.evaluate(context, weights)
        
        assert len(fired_rules) == 0
        assert final_weights == weights  # unchanged


class TestSemanticScorer:
    """Test SemanticScorer functionality."""
    
    def test_semantic_scorer_basic(self):
        """Test basic semantic scoring."""
        context = SemanticContext(
            domain="biological",
            noise_level="high",
            graph_size="large",
            graph_type="scale_free",
            priority="precision",
        )
        
        # Setup scorer with default rules
        scorer = SemanticScorer(context)
        
        # Raw metrics
        metrics = {
            "shd": 3.0,
            "precision": 0.85,
            "recall": 0.75,
            "f1": 0.80,
            "orientation_f1": 0.65,
            "runtime": 0.5,
        }
        
        # Get semantic score
        composite_score, trace = scorer.score(metrics)
        
        assert isinstance(composite_score, float)
        assert 0.0 <= composite_score <= 1.0
        assert isinstance(trace, list)
        assert len(trace) > 0
    
    def test_semantic_scorer_trace_generation(self):
        """Test that scorer generates human-readable traces."""
        context = SemanticContext(
            domain="financial",
            noise_level="medium",
            graph_size="medium",
            graph_type="erdos_renyi",
            priority="precision",
        )
        
        scorer = SemanticScorer(context)
        
        metrics = {
            "shd": 2.0,
            "precision": 0.90,
            "recall": 0.80,
            "f1": 0.85,
        }
        
        score, trace = scorer.score(metrics)
        
        # Check trace structure
        assert isinstance(trace, list)
        for step in trace:
            assert isinstance(step, str)
            # Each step should have some meaningful content
            assert len(step) > 0
    
    def test_semantic_scorer_normalization(self):
        """Test that scores are properly normalized."""
        context = SemanticContext(
            domain="synthetic",
            noise_level="low",
            graph_size="small",
            graph_type="erdos_renyi",
            priority="balanced",
        )
        
        scorer = SemanticScorer(context)
        
        # Perfect metrics
        perfect_metrics = {
            "shd": 0.0,    # 0 SHD is perfect
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
        
        perfect_score, _ = scorer.score(perfect_metrics)
        
        # Perfect metrics should yield high score (close to 1.0)
        assert perfect_score > 0.8
        
        # Terrible metrics
        terrible_metrics = {
            "shd": 100.0,
            "precision": 0.1,
            "recall": 0.1,
            "f1": 0.1,
        }
        
        terrible_score, _ = scorer.score(terrible_metrics)
        
        # Terrible metrics should yield low score
        assert terrible_score < 0.5


class TestSemanticIntegration:
    """Integration tests for semantic layer."""
    
    def test_benchmark_with_semantic_context(self):
        """Test BenchmarkRunner with semantic context."""
        from pgmpy.benchmark import BenchmarkRunner, ErdosRenyiSimulator
        
        class DummyMethod:
            def __call__(self, data):
                import networkx as nx
                dag = nx.DiGraph()
                dag.add_edge("X0", "X1")
                return dag
        
        # Run benchmark with semantic context
        runner = BenchmarkRunner(
            simulators=[
                ErdosRenyiSimulator(n_nodes=5, edge_prob=0.3, n_samples=100, seed=42)
            ],
            methods=[DummyMethod()],
            metrics=["shd", "precision_recall"],
            n_runs=1,
            n_jobs=1,
            semantic_context={
                "domain": "biological",
                "noise_level": "high",
                "graph_size": "small",
                "graph_type": "erdos_renyi",
                "priority": "precision",
            }
        )
        
        results = runner.run()
        
        # Results should have semantic context
        assert len(results.runs) == 1
        assert results.config["semantic_context"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
