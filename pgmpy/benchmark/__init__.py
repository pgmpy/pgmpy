"""
pgmpy.benchmark — Cognitive Benchmarking Framework for Causal Inference

A modular, composable benchmarking system with semantic evaluation and chain-of-thought reasoning.

Core Modules:
    - simulators: Data generation (Erdos-Renyi, scale-free, real BNs)
    - metrics: Evaluation metrics (SHD, precision/recall, SID)
    - semantic: Context-aware scoring with rule-based evaluation
    - reasoning: Chain-of-thought traces and explainability
    - storage: Result persistence and memory-based recommendations
    - runner: Orchestrator for benchmark execution

Example:
    >>> from pgmpy.benchmark import BenchmarkRunner, ErdosRenyiSimulator
    >>> from pgmpy.estimators import PC, HillClimbSearch
    >>> from pgmpy.benchmark.metrics import shd, precision_recall
    >>>
    >>> runner = BenchmarkRunner(
    ...     simulators=[ErdosRenyiSimulator(n_nodes=10, edge_prob=0.3, n_samples=500)],
    ...     methods=[PC(ci_test='pearsonr'), HillClimbSearch()],
    ...     metrics=[shd, precision_recall],
    ...     n_runs=20,
    ... )
    >>> results = runner.run()

References:
    - Architecture: pgmpy/benchmark/ARCHITECTURE.md
    - GSoC Proposal: Cognitive Benchmarking Framework for Causal Inference
    - Author: Mohamed Habib Khattat
"""

__version__ = "0.1.0"

from pgmpy.benchmark.runner import BenchmarkRunner, BenchmarkRun, BenchmarkResults
from pgmpy.benchmark.simulators import (
    ErdosRenyiSimulator,
    ScaleFreeSimulator,
    RealBNSimulator,
    LinearGaussianSEM,
)
from pgmpy.benchmark.metrics import (
    MetricsRegistry,
    shd,
    precision_recall,
    orientation_f1,
    sid,
)
from pgmpy.benchmark.semantic import (
    SemanticContext,
    EvaluationRule,
    RuleEngine,
    SemanticScorer,
)
from pgmpy.benchmark.reasoning import (
    ReasoningStep,
    ChainOfThoughtTracer,
    Explanation,
)

__all__ = [
    # Runner
    "BenchmarkRunner",
    "BenchmarkRun",
    "BenchmarkResults",
    # Simulators
    "ErdosRenyiSimulator",
    "ScaleFreeSimulator",
    "RealBNSimulator",
    "LinearGaussianSEM",
    # Metrics
    "MetricsRegistry",
    "shd",
    "precision_recall",
    "orientation_f1",
    "sid",
    # Semantic (Phase 2)
    "SemanticContext",
    "EvaluationRule",
    "RuleEngine",
    "SemanticScorer",
    # Reasoning (Phase 3)
    "ReasoningStep",
    "ChainOfThoughtTracer",
    "Explanation",
]
