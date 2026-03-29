"""
Reasoning layer for benchmarking framework (Phase 3).

Provides:
    - ReasoningStep: Atomic reasoning unit describing an action and result
    - ChainOfThoughtTracer: Traces benchmark execution steps with context
    - Explanation: Natural language explanation combining tracer + semantic insights
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class ReasoningStep:
    """
    Single step in a chain-of-thought reasoning trace.
    
    Attributes:
        step_number: Ordinal position in the trace (1-indexed)
        action: Name of action (e.g., 'run_algorithm', 'compute_metric')
        result: Outcome description (e.g., 'DAG estimated in 0.43s')
        metadata: Optional additional context (dict of properties)
    """
    
    step_number: int
    action: str
    result: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary (for JSON serialization)."""
        return {
            "step": self.step_number,
            "action": self.action,
            "result": self.result,
            "metadata": self.metadata,
        }
    
    def to_string(self) -> str:
        """Convert step to human-readable string."""
        base = f"[Step {self.step_number}] {self.action.upper()}: {self.result}"
        if self.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            return f"{base} ({meta_str})"
        return base


@dataclass
class ChainOfThoughtTracer:
    """
    Traces all steps in a benchmark run with reasoning context.
    
    Used to:
        - Log causal discovery algorithm execution
        - Record metric computations
        - Capture interpretations and insights
        - Enable debugging and transparency
    
    Attributes:
        method_name: Name of causal discovery method (e.g., 'PC', 'GES')
        simulator_name: Name of dataset simulator (e.g., 'ErdosRenyi')
        run_id: Unique run identifier
        steps: List of ReasoningSteps in execution order
    """
    
    method_name: str
    simulator_name: str
    run_id: str
    steps: List[ReasoningStep] = field(default_factory=list)
    
    def add_step(
        self,
        action: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a reasoning step to the trace.
        
        Args:
            action: Name of action taken
            result: Outcome of the action
            metadata: Optional dict with additional context
        """
        step_number = len(self.steps) + 1
        step = ReasoningStep(
            step_number=step_number,
            action=action,
            result=result,
            metadata=metadata or {},
        )
        self.steps.append(step)
    
    def get_trace_string(self) -> str:
        """
        Generate human-readable trace.
        
        Returns:
            Formatted string showing all steps with context.
        """
        lines = [
            f"{'='*70}",
            f"Chain-of-Thought Trace: {self.method_name} on {self.simulator_name}",
            f"Run ID: {self.run_id}",
            f"{'='*70}",
        ]
        
        for step in self.steps:
            lines.append(step.to_string())
        
        lines.append(f"{'='*70}")
        lines.append(f"Total steps: {len(self.steps)}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire trace to dict."""
        return {
            "method": self.method_name,
            "simulator": self.simulator_name,
            "run_id": self.run_id,
            "steps": [step.to_dict() for step in self.steps],
            "total_steps": len(self.steps),
        }
    
    def to_json(self) -> str:
        """Serialize trace to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Explanation:
    """
    Natural language explanation of benchmark results.
    
    Combines:
        - Chain-of-thought trace (what happened)
        - Semantic context (domain/noise/size constraints)
        - Fired rules (what influenced scoring)
        - Component scores (individual metric performance)
        - Composite score (final quality metric)
    
    Attributes:
        run_id: Benchmark run identifier
        method_name: Causal discovery method name
        composite_score: Final composite score [0, 1]
        semantic_context: Context dict with domain/noise/size
        fired_rules: List of rule names that affected scoring
        component_scores: Dict of individual metric scores
        tracer: Optional ChainOfThoughtTracer for detailed trace
    """
    
    run_id: str
    method_name: str
    composite_score: float
    semantic_context: Dict[str, Any]
    fired_rules: List[str]
    component_scores: Dict[str, float]
    tracer: Optional[ChainOfThoughtTracer] = None
    
    def to_narrative(self) -> str:
        """
        Generate natural-language narrative explanation.
        
        Returns:
            Human-readable paragraph explaining the result.
        """
        # Extract key context
        domain = self.semantic_context.get("domain", "general")
        noise = self.semantic_context.get("noise_level", "unknown")
        graph_size = self.semantic_context.get("graph_size", "unknown")
        
        # Determine quality level
        if self.composite_score >= 0.85:
            quality = "excellent"
        elif self.composite_score >= 0.70:
            quality = "good"
        elif self.composite_score >= 0.50:
            quality = "moderate"
        else:
            quality = "poor"
        
        # Build narrative
        narrative_parts = [
            f"Method {self.method_name} achieved {quality} performance "
            f"(composite score: {self.composite_score:.3f}) "
            f"on {domain} domain data",
        ]
        
        if noise != "unknown":
            narrative_parts.append(f"with {noise} noise level")
        
        if graph_size != "unknown":
            narrative_parts.append(f"and {graph_size} graph size")
        
        narrative_parts[0] += "."
        
        # Add component scores
        if self.component_scores:
            scores_str = ", ".join(
                f"{name}={score:.3f}"
                for name, score in self.component_scores.items()
            )
            narrative_parts.append(f"Component scores: {scores_str}.")
        
        # Add fired rules
        if self.fired_rules:
            rules_str = ", ".join(self.fired_rules)
            narrative_parts.append(f"Scoring adjusted by rules: {rules_str}.")
        
        return " ".join(narrative_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dict."""
        return {
            "run_id": self.run_id,
            "method": self.method_name,
            "composite_score": self.composite_score,
            "semantic_context": self.semantic_context,
            "fired_rules": self.fired_rules,
            "component_scores": self.component_scores,
            "narrative": self.to_narrative(),
        }


__all__ = [
    "ReasoningStep",
    "ChainOfThoughtTracer",
    "Explanation",
]
