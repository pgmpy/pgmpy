"""
Semantic evaluation layer — context-aware scoring with rule-based reasoning.

Inspired by OWL/SWRL reasoning from knowledge engineering.
Applies domain rules to adjust evaluation metrics contextually.

Example:
    >>> context = SemanticContext(
    ...     domain="biological",
    ...     noise_level="high",
    ...     graph_size="large",
    ...     graph_type="scale_free",
    ...     priority="precision",
    ... )
    >>>
    >>> scorer = SemanticScorer(context)
    >>> metrics = {"shd": 3, "precision": 0.85, "recall": 0.75}
    >>> score, trace = scorer.score(metrics)
    >>> print(f"Semantic score: {score}")
    >>> for step in trace:
    ...     print(step)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Callable, Optional, Tuple
import numpy as np


@dataclass
class SemanticContext:
    """
    Domain context for semantic evaluation.
    
    Attributes
    ----------
    domain : str
        Application domain: 'biological', 'financial', 'synthetic'
    noise_level : str
        Noise regime: 'low', 'medium', 'high'
    graph_size : str
        Graph complexity: 'small' (<10 nodes), 'medium' (10-30), 'large' (>30)
    graph_type : str
        Graph structure: 'erdos_renyi', 'scale_free', 'real_bn'
    priority : str
        Evaluation priority: 'precision', 'recall', 'balanced'
    """
    
    domain: str = "synthetic"
    noise_level: str = "medium"
    graph_size: str = "medium"
    graph_type: str = "erdos_renyi"
    priority: str = "balanced"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert context to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "SemanticContext":
        """Create context from dictionary."""
        return cls(**data)


@dataclass
class EvaluationRule:
    """
    Declarative evaluation rule (SWRL-inspired).
    
    Rules are applied based on context conditions and modify metric weights.
    
    Parameters
    ----------
    name : str
        Unique rule name.
    condition : callable
        Predicate function: SemanticContext -> bool.
    action : callable
        Weight modification function: Dict[str, float] -> Dict[str, float].
    rationale : str
        Human-readable explanation of the rule.
    """
    
    name: str
    condition: Callable[[SemanticContext], bool]
    action: Callable[[Dict[str, float]], Dict[str, float]]
    rationale: str
    
    def evaluate_condition(self, context: SemanticContext) -> bool:
        """Check if rule applies to context."""
        try:
            return self.condition(context)
        except Exception:
            return False
    
    def execute_action(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply rule's weight modification."""
        try:
            return self.action(weights)
        except Exception:
            return weights


class RuleEngine:
    """
    Rule evaluation engine for semantic scoring.
    
    Processes a set of rules against a semantic context and returns
    the fired rules and adjusted weights.
    """
    
    def __init__(self, rules: List[EvaluationRule]):
        """
        Initialize RuleEngine.
        
        Parameters
        ----------
        rules : List[EvaluationRule]
            Rules to evaluate.
        """
        self.rules = rules
    
    def evaluate(
        self,
        context: SemanticContext,
        weights: Dict[str, float],
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Evaluate all rules against context.
        
        Parameters
        ----------
        context : SemanticContext
            Domain context.
        weights : Dict[str, float]
            Metric weights to adjust.
            
        Returns
        -------
        Tuple[List[str], Dict[str, float]]
            (fired_rule_names, adjusted_weights)
        """
        fired_rules: List[str] = []
        current_weights = weights.copy()
        
        for rule in self.rules:
            if rule.evaluate_condition(context):
                fired_rules.append(rule.name)
                current_weights = rule.execute_action(current_weights)
        
        return fired_rules, current_weights


# Default rule library (SWRL-inspired, OWL-style reasoning)
def _default_rules() -> List[EvaluationRule]:
    """Create default set of evaluation rules."""
    return [
        EvaluationRule(
            name="high_noise_robustness",
            condition=lambda ctx: ctx.noise_level == "high",
            action=lambda w: {**w, **{k: v * 1.4 if k == "precision" else v for k, v in w.items()}},
            rationale="High noise: precision more important than SHD",
        ),
        EvaluationRule(
            name="biological_network_orientation",
            condition=lambda ctx: ctx.domain == "biological",
            action=lambda w: {**w, **{k: v * 1.8 if k in ("orientation_f1", "f1") else v for k, v in w.items()}},
            rationale="Biology: causal direction critical (gene regulatory networks)",
        ),
        EvaluationRule(
            name="large_graph_scalability",
            condition=lambda ctx: ctx.graph_size == "large",
            action=lambda w: {**w, **{k: v * 1.5 if k == "runtime" else v for k, v in w.items()}},
            rationale="Large graphs: runtime is first-class concern",
        ),
        EvaluationRule(
            name="recall_priority_penalty",
            condition=lambda ctx: ctx.priority == "recall",
            action=lambda w: {**w, **{k: v * 1.3 if k == "recall" else v * 0.9 if k == "precision" else v for k, v in w.items()}},
            rationale="Recall priority: favor recall, penalize precision",
        ),
        EvaluationRule(
            name="precision_priority_adjustment",
            condition=lambda ctx: ctx.priority == "precision",
            action=lambda w: {**w, **{k: v * 1.3 if k == "precision" else v * 0.9 if k == "recall" else v for k, v in w.items()}},
            rationale="Precision priority: favor precision, penalize recall",
        ),
        EvaluationRule(
            name="low_noise_reliability",
            condition=lambda ctx: ctx.noise_level == "low",
            action=lambda w: {**w, **{k: v * 1.2 if k in ("f1", "orientation_f1") else v for k, v in w.items()}},
            rationale="Low noise: high-quality solutions should match ground truth well",
        ),
        EvaluationRule(
            name="scale_free_structure_critical",
            condition=lambda ctx: ctx.graph_type == "scale_free",
            action=lambda w: {**w, **{k: v * 1.3 if k == "f1" else v for k, v in w.items()}},
            rationale="Scale-free: preserving hub structure is critical",
        ),
    ]


class SemanticScorer:
    """
    Context-aware composite scorer.
    
    Applies semantic rules to raw metrics and produces a normalized
    composite score with human-readable reasoning trace.
    
    Parameters
    ----------
    context : SemanticContext
        Domain context for scoring.
    rules : List[EvaluationRule], optional
        Custom rules. If None, uses default rules.
    """
    
    def __init__(
        self,
        context: SemanticContext,
        rules: Optional[List[EvaluationRule]] = None,
    ):
        """Initialize the scorer."""
        self.context = context
        self.rules = rules or _default_rules()
        self.rule_engine = RuleEngine(self.rules)
    
    def score(
        self,
        metrics: Dict[str, float],
    ) -> Tuple[float, List[str]]:
        """
        Compute semantic score with reasoning trace.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Raw metric values. Expected keys: shd, precision, recall, f1, 
            orientation_f1, runtime.
            
        Returns
        -------
        Tuple[float, List[str]]
            (composite_score, reasoning_trace)
            - composite_score: normalized score in [0, 1]
            - reasoning_trace: list of human-readable reasoning steps
        """
        trace: List[str] = []
        
        # Step 1: Log context
        trace.append(f"[CONTEXT] Domain={self.context.domain}, "
                    f"Noise={self.context.noise_level}, "
                    f"Size={self.context.graph_size}, "
                    f"Priority={self.context.priority}")
        
        # Step 2: Log raw metrics
        metric_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items()))
        trace.append(f"[METRICS] {metric_str}")
        
        # Step 3: Initialize default weights (inverse importance)
        # Normalize so SHD is worst (minimize), others are best (maximize)
        weights = {
            "shd": 1.0,          # Lower is better → weight 1.0 for now
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.2,           # F1 slightly more important
            "orientation_f1": 1.0,
            "runtime": 0.5,      # Runtime less critical by default
        }
        
        trace.append(f"[WEIGHTS] Initial: {{{', '.join(f'{k}={v:.2f}' for k, v in sorted(weights.items()))}}}")
        
        # Step 4: Evaluate rules
        fired_rules, adjusted_weights = self.rule_engine.evaluate(self.context, weights)
        
        if fired_rules:
            trace.append(f"[RULES FIRED] {', '.join(fired_rules)}")
            for rule in self.rules:
                if rule.name in fired_rules:
                    trace.append(f"  → {rule.name}: {rule.rationale}")
            trace.append(f"[WEIGHTS ADJUSTED] {{{', '.join(f'{k}={v:.2f}' for k, v in sorted(adjusted_weights.items()))}}}")
        else:
            trace.append("[RULES FIRED] None")
        
        # Step 5: Compute normalized component scores
        component_scores = self._compute_component_scores(metrics, adjusted_weights)
        trace.append(f"[COMPONENTS] {', '.join(f'{k}={v:.3f}' for k, v in sorted(component_scores.items()))}")
        
        # Step 6: Aggregate to composite score
        composite_score = self._aggregate_scores(component_scores, adjusted_weights)
        
        trace.append(f"[COMPOSITE SCORE] {composite_score:.4f}")
        trace.append(f"[INTERPRETATION] "
                    f"{'Excellent' if composite_score > 0.85 else 'Good' if composite_score > 0.70 else 'Fair' if composite_score > 0.50 else 'Poor'} "
                    f"performance for {self.context.domain} with {self.context.priority} priority")
        
        return composite_score, trace
    
    def _compute_component_scores(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute normalized component scores."""
        components = {}
        
        # SHD: normalize inversely (lower is better)
        # Assume SHD ranges from 0 (perfect) to ~50 (terrible for typical graphs)
        shd_val = metrics.get("shd", 0.0)
        components["shd"] = max(0.0, 1.0 - min(1.0, shd_val / 20.0))
        
        # Precision, Recall, F1: already in [0, 1]
        for key in ("precision", "recall", "f1", "orientation_f1"):
            components[key] = max(0.0, min(1.0, metrics.get(key, 0.5)))
        
        # Runtime: normalize inversely (lower is better)
        # Assume runtime in seconds, serious concern if > 10s
        runtime = metrics.get("runtime", 0.0)
        components["runtime"] = max(0.0, 1.0 - min(1.0, runtime / 10.0))
        
        return components
    
    def _aggregate_scores(
        self,
        components: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """Aggregate component scores with weights."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric_name, component_value in components.items():
            weight = weights.get(metric_name, 1.0)
            weighted_sum += component_value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        # Normalize to [0, 1]
        composite = weighted_sum / total_weight
        return max(0.0, min(1.0, composite))


# Export public API
__all__ = [
    "SemanticContext",
    "EvaluationRule",
    "RuleEngine",
    "SemanticScorer",
]
