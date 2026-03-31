"""
Core benchmark runner orchestrator.

The BenchmarkRunner is the main entry point for running benchmarks.
It coordinates simulators, methods, metrics, and semantic evaluation.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
import json
import time
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
import logging

from pgmpy.benchmark.base import BaseSimulator, BaseMetric, MetricResult
from pgmpy.benchmark.metrics import shd, precision_recall, orientation_f1, sid

# Setup module-level logger
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """Results from a single benchmark run."""
    
    run_id: int
    """Unique run identifier."""
    
    simulator_name: str
    """Name of the data simulator used."""
    
    method_name: str
    """Name of the causal discovery method."""
    
    metrics: Dict[str, float]
    """Dict mapping metric name -> value."""
    
    metrics_detail: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Detailed metadata for each metric."""
    
    execution_time: float = 0.0
    """Wall-clock time for method execution."""
    
    seed: Optional[int] = None
    """Random seed used."""
    
    ground_truth_dag: Optional[nx.DiGraph] = None
    """Ground truth DAG (optional, for internal use)."""
    
    estimated_dag: Optional[nx.DiGraph] = None
    """Estimated DAG from method (optional, for internal use)."""


@dataclass
class BenchmarkResults:
    """Aggregate results from multiple benchmark runs."""
    
    runs: List[BenchmarkRun]
    """Individual run results."""
    
    config: Dict[str, Any]
    """Configuration used."""
    
    timestamp: str = ""
    """Timestamp of benchmark execution."""
    
    def summary(self) -> pd.DataFrame:
        """
        Return a summary as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            One row per (simulator, method) pair, columns for each metric.
        """
        # Aggregate by (simulator, method)
        agg_data = {}
        
        for run in self.runs:
            key = (run.simulator_name, run.method_name)
            if key not in agg_data:
                agg_data[key] = {"runs": []}
            agg_data[key]["runs"].append(run)
        
        # Compute means and stds
        summary_rows = []
        for (sim_name, method_name), group in agg_data.items():
            runs = group["runs"]
            
            # Collect all metrics
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.metrics.keys())
            
            row = {
                "simulator": sim_name,
                "method": method_name,
                "n_runs": len(runs),
            }
            
            for metric_name in sorted(all_metrics):
                values = [r.metrics.get(metric_name, None) for r in runs]
                values = [v for v in values if v is not None]
                
                if values:
                    row[f"{metric_name}_mean"] = float(np.mean(values))
                    if len(values) > 1:
                        row[f"{metric_name}_std"] = float(np.std(values))
            
            summary_rows.append(row)
        
        return pd.DataFrame(summary_rows)
    
    def to_json(self, filepath: str):
        """Export results to JSON."""
        # Convert runs to dicts (exclude DAG objects)
        runs_data = []
        for run in self.runs:
            run_dict = asdict(run)
            run_dict.pop("ground_truth_dag", None)
            run_dict.pop("estimated_dag", None)
            runs_data.append(run_dict)
        
        output = {
            "timestamp": self.timestamp,
            "config": self.config,
            "runs": runs_data,
        }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
    
    def to_csv(self, filepath: str):
        """Export summary to CSV."""
        self.summary().to_csv(filepath, index=False)


class BenchmarkRunner:
    """
    Main benchmark orchestrator.
    
    Coordinates simulation, method execution, metric computation,
    and semantic evaluation.
    
    Parameters
    ----------
    simulators : List[BaseSimulator]
        List of data simulators.
    methods : List
        List of causal discovery methods (callables or pgmpy estimators).
    metrics : List[Union[BaseMetric, str]]
        List of metrics to compute. Can be metric names (str) or instances.
    n_runs : int
        Number of runs per (simulator, method) pair (default: 10).
    n_jobs : int
        Number of parallel jobs (default: 1). Use -1 for all CPUs.
    output_format : str
        Output format: 'json' or 'csv' (default: 'json').
    semantic_context : Dict, optional
        Context for semantic evaluation layer.
    verbose : int
        Verbosity level (default: 0).
    """
    
    def __init__(
        self,
        simulators: List[BaseSimulator],
        methods: List[Any],
        metrics: List[Union[BaseMetric, str]] = None,
        n_runs: int = 10,
        n_jobs: int = 1,
        output_format: str = "json",
        semantic_context: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ):
        # INPUT VALIDATION (BUG FIX #2)
        if not simulators:
            raise ValueError("At least one simulator must be provided")
        if not methods:
            raise ValueError("At least one method must be provided")
        if n_runs <= 0:
            raise ValueError(f"n_runs must be positive integer (got {n_runs})")
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"n_jobs must be -1 or positive integer (got {n_jobs})")
        if output_format not in ["json", "csv"]:
            raise ValueError(f"output_format must be 'json' or 'csv' (got {output_format})")
        
        # Validate simulators have required interface
        for i, sim in enumerate(simulators):
            if not hasattr(sim, 'simulate') or not callable(sim.simulate):
                raise TypeError(f"Simulator {i} ({sim}) must have simulate() method")
            if not hasattr(sim, 'get_name') or not callable(sim.get_name):
                raise TypeError(f"Simulator {i} ({sim}) must have get_name() method")
        
        # Validate methods are callable
        for i, method in enumerate(methods):
            if not callable(method) and not hasattr(method, 'estimate') and not hasattr(method, 'fit'):
                raise TypeError(f"Method {i} ({method}) must be callable or have estimate()/fit() methods")
        
        self.simulators = simulators
        self.methods = methods
        self.n_runs = n_runs
        self.n_jobs = n_jobs
        self.output_format = output_format
        self.semantic_context = semantic_context or {}
        self.verbose = verbose
        self.logger = logger  # Use module-level logger (BUG FIX #3)
        
        # Set up metrics (default: SHD + precision_recall)
        if metrics is None:
            metrics = [shd, precision_recall, orientation_f1, sid]
        
        self.metrics = []
        for metric in metrics:
            if isinstance(metric, str):
                from pgmpy.benchmark.metrics import MetricsRegistry
                try:
                    self.metrics.append(MetricsRegistry.get(metric))
                except KeyError:
                    raise ValueError(f"Unknown metric: '{metric}'")
            elif callable(metric):
                # Assume it's a metric function; wrap it
                self.metrics.append(_MetricWrapper(metric))
            else:
                self.metrics.append(metric)
        
        if self.verbose > 0:
            self.logger.info(f"BenchmarkRunner initialized: {len(simulators)} simulators, "
                           f"{len(methods)} methods, {len(self.metrics)} metrics")

    
    def run(self) -> BenchmarkResults:
        """
        Execute the benchmark.
        
        Returns
        -------
        BenchmarkResults
            All result data and summary information.
        """
        if self.verbose > 0:
            self.logger.info("Starting benchmark run...")
        
        all_runs = []
        run_id = 0
        
        # Generate all tasks
        tasks = []
        for sim_idx, simulator in enumerate(self.simulators):
            for method_idx, method in enumerate(self.methods):
                for run_num in range(self.n_runs):
                    seed = run_num if self.n_runs > 1 else None
                    tasks.append({
                        "run_id": run_id,
                        "simulator": simulator,
                        "method": method,
                        "seed": seed,
                    })
                    run_id += 1
        
        # Execute tasks
        if self.n_jobs == 1:
            results = [self._run_single(**task) for task in tasks]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_single)(**task) for task in tasks
            )
        
        all_runs.extend(results)
        
        if self.verbose > 0:
            self.logger.info(f"Completed {len(all_runs)} runs")
        
        config = {
            "n_simulators": len(self.simulators),
            "n_methods": len(self.methods),
            "n_runs": self.n_runs,
            "n_metrics": len(self.metrics),
            "n_jobs": self.n_jobs,
        }
        
        # Add semantic context if present
        if self.semantic_context:
            config["semantic_context"] = self.semantic_context
        
        return BenchmarkResults(
            runs=all_runs,
            config=config,
            timestamp=pd.Timestamp.now().isoformat(),
        )
    
    def _run_single(
        self,
        run_id: int,
        simulator: BaseSimulator,
        method: Any,
        seed: Optional[int] = None,
    ) -> BenchmarkRun:
        """
        Execute a single benchmark run.
        
        Parameters
        ----------
        run_id : int
            Unique run identifier.
        simulator : BaseSimulator
            Data simulator.
        method : Any
            Causal discovery method.
        seed : int, optional
            Random seed.
            
        Returns
        -------
        BenchmarkRun
            Result from this single run.
        """
        # Generate synthetic data
        sim_output = simulator.simulate(seed=seed)
        
        # Run the method
        start_time = time.time()
        try:
            estimated_dag = self._run_method(method, sim_output.data)
        except Exception as e:
            if self.verbose > 0:
                self.logger.warning(f"Method {self._method_name(method)} failed: {e}")
            estimated_dag = nx.DiGraph()  # Empty fallback
        
        execution_time = time.time() - start_time
        
        # Compute metrics
        metrics_dict = {}
        metrics_detail = {}
        
        for metric in self.metrics:
            try:
                result = metric.compute(estimated_dag, sim_output.dag)
                if not isinstance(result, MetricResult):
                    raise TypeError(f"Metric {metric} must return MetricResult, got {type(result)}")
                metrics_dict[result.name] = result.value
                metrics_detail[result.name] = result.metadata
            except Exception as e:
                if self.verbose > 0:
                    self.logger.warning(f"Metric computation failed: {e}")
                # Better fallback handling
                metric_func = getattr(metric, 'func', metric)
                metric_name = getattr(metric, '_name', getattr(metric_func, '__name__', str(metric)))
                metrics_dict[metric_name] = np.nan
        
        return BenchmarkRun(
            run_id=run_id,
            simulator_name=simulator.get_name(),
            method_name=self._method_name(method),
            metrics=metrics_dict,
            metrics_detail=metrics_detail,
            execution_time=execution_time,
            seed=seed,
            ground_truth_dag=sim_output.dag,
            estimated_dag=estimated_dag,
        )
    
    def _run_method(self, method: Any, data: pd.DataFrame) -> nx.DiGraph:
        """
        Run a causal discovery method on data.
        
        Parameters
        ----------
        method : Any
            Method instance (pgmpy estimator or callable).
        data : pd.DataFrame
            Observational data.
            
        Returns
        -------
        nx.DiGraph
            Estimated causal DAG.
            
        Raises
        ------
        TypeError
            If method doesn't return a valid nx.DiGraph.
        """
        result = None
        
        # Try to detect method type and call appropriately
        if hasattr(method, "estimate") and callable(getattr(method, "estimate")):
            # pgmpy estimator interface
            result = method.estimate()
        elif hasattr(method, "fit") and callable(getattr(method, "fit")):
            # sklearn-like interface
            method.fit(data)
            if not hasattr(method, 'graph_'):
                raise AttributeError(f"Method {self._method_name(method)} must have 'graph_' attribute after fit()")
            result = method.graph_
        elif callable(method):
            # User-provided callable
            result = method(data)
        else:
            raise ValueError(f"Unknown method type: {type(method)}")
        
        # VALIDATE RESULT TYPE (BUG FIX #5)
        if not isinstance(result, nx.DiGraph):
            raise TypeError(
                f"Method {self._method_name(method)} must return nx.DiGraph, "
                f"got {type(result).__name__}"
            )
        
        return result
    
    def _method_name(self, method: Any) -> str:
        """Get descriptive name for a method."""
        if hasattr(method, "__class__"):
            return method.__class__.__name__
        else:
            return str(method)


class _MetricWrapper(BaseMetric):
    """Wrapper for metric functions to conform to BaseMetric interface."""
    
    def __init__(self, func: Callable):
        self.func = func
    
    def compute(
        self,
        estimated_dag: nx.DiGraph,
        ground_truth_dag: nx.DiGraph,
    ) -> MetricResult:
        """Compute by calling wrapped function."""
        result = self.func(estimated_dag, ground_truth_dag)
        # Assume result is a MetricResult or dict
        if isinstance(result, MetricResult):
            return result
        else:
            # Try to convert
            return MetricResult(
                name=getattr(self.func, "__name__", "unknown"),
                value=float(result) if isinstance(result, (int, float)) else result,
                metadata={},
            )
    
    def get_name(self) -> str:
        """Return function name."""
        return getattr(self.func, "__name__", "wrapped_metric")


import numpy as np  # Add numpy import for summary
