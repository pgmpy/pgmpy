from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
import json
import time
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
import logging
import numpy as np

from pgmpy.benchmark.base import BaseSimulator, BaseMetric, MetricResult
from pgmpy.benchmark.metrics import shd, precision_recall, orientation_f1, sid

logger = logging.getLogger(__name__)
logging.getLogger("pgmpy.benchmark").disabled = True

@dataclass
class BenchmarkRun:
    run_id: int
    simulator_name: str
    method_name: str
    metrics: Dict[str, float]
    metrics_detail: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_time: float = 0.0
    seed: Optional[int] = None
    ground_truth_dag: Optional[nx.DiGraph] = None
    estimated_dag: Optional[nx.DiGraph] = None

@dataclass
class BenchmarkResults:
    runs: List[BenchmarkRun]
    config: Dict[str, Any]
    timestamp: str = ""

    def summary(self) -> pd.DataFrame:
        agg_data = {}
        for run in self.runs:
            key = (run.simulator_name, run.method_name)
            if key not in agg_data:
                agg_data[key] = {"runs": []}
            agg_data[key]["runs"].append(run)
        summary_rows = []
        for (sim_name, method_name), group in agg_data.items():
            runs = group["runs"]
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.metrics.keys())
            row = {"simulator": sim_name, "method": method_name, "n_runs": len(runs)}
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
        runs_data = []
        for run in self.runs:
            run_dict = asdict(run)
            run_dict.pop("ground_truth_dag", None)
            run_dict.pop("estimated_dag", None)
            runs_data.append(run_dict)
        output = {"timestamp": self.timestamp, "config": self.config, "runs": runs_data}
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

    def to_csv(self, filepath: str):
        self.summary().to_csv(filepath, index=False)

class BenchmarkRunner:
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
        for i, sim in enumerate(simulators):
            if not hasattr(sim, 'simulate') or not callable(sim.simulate):
                raise TypeError(f"Simulator {i} ({sim}) must have simulate() method")
            if not hasattr(sim, 'get_name') or not callable(sim.get_name):
                raise TypeError(f"Simulator {i} ({sim}) must have get_name() method")
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
        self.logger = logger
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
                self.metrics.append(_MetricWrapper(metric))
            else:
                self.metrics.append(metric)
        if self.verbose > 0:
            self.logger.info(f"BenchmarkRunner initialized: {len(simulators)} simulators, {len(methods)} methods, {len(self.metrics)} metrics")

    def run(self) -> BenchmarkResults:
        if self.verbose > 0:
            self.logger.info("Starting benchmark run...")
        all_runs = []
        run_id = 0
        tasks = []
        for simulator in self.simulators:
            for method in self.methods:
                for run_num in range(self.n_runs):
                    seed = run_num if self.n_runs > 1 else None
                    tasks.append({"run_id": run_id, "simulator": simulator, "method": method, "seed": seed})
                    run_id += 1
        if self.n_jobs == 1:
            results = [self._run_single(**task) for task in tasks]
        else:
            results = Parallel(n_jobs=self.n_jobs)(delayed(self._run_single)(**task) for task in tasks)
        all_runs.extend(results)
        if self.verbose > 0:
            self.logger.info(f"Completed {len(all_runs)} runs")
        config = {"n_simulators": len(self.simulators), "n_methods": len(self.methods), "n_runs": self.n_runs, "n_metrics": len(self.metrics), "n_jobs": self.n_jobs}
        if self.semantic_context:
            config["semantic_context"] = self.semantic_context
        return BenchmarkResults(runs=all_runs, config=config, timestamp=pd.Timestamp.now().isoformat())

    def _run_single(self, run_id: int, simulator: BaseSimulator, method: Any, seed: Optional[int] = None) -> BenchmarkRun:
        sim_output = simulator.simulate(seed=seed)
        start_time = time.time()
        try:
            estimated_dag = self._run_method(method, sim_output.data)
        except Exception as e:
            if self.verbose > 0:
                self.logger.warning(f"Method {self._method_name(method)} failed: {e}")
            estimated_dag = nx.DiGraph()
        execution_time = time.time() - start_time
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
                metric_func = getattr(metric, 'func', metric)
                metric_name = getattr(metric, '_name', getattr(metric_func, '__name__', str(metric)))
                metrics_dict[metric_name] = np.nan
        return BenchmarkRun(run_id=run_id, simulator_name=simulator.get_name(), method_name=self._method_name(method), metrics=metrics_dict, metrics_detail=metrics_detail, execution_time=execution_time, seed=seed, ground_truth_dag=sim_output.dag, estimated_dag=estimated_dag)

    def _run_method(self, method: Any, data: pd.DataFrame) -> nx.DiGraph:
        result = None
        if hasattr(method, "estimate") and callable(getattr(method, "estimate")):
            result = method.estimate()
        elif hasattr(method, "fit") and callable(getattr(method, "fit")):
            method.fit(data)
            if not hasattr(method, 'graph_'):
                raise AttributeError(f"Method {self._method_name(method)} must have 'graph_' attribute after fit()")
            result = method.graph_
        elif callable(method):
            result = method(data)
        else:
            raise ValueError(f"Unknown method type: {type(method)}")
        if not isinstance(result, nx.DiGraph):
            raise TypeError(f"Method {self._method_name(method)} must return nx.DiGraph, got {type(result).__name__}")
        return result

    def _method_name(self, method: Any) -> str:
        if hasattr(method, "__class__"):
            return method.__class__.__name__
        return str(method)

class _MetricWrapper(BaseMetric):
    def __init__(self, func: Callable):
        self.func = func
    def compute(self, estimated_dag: nx.DiGraph, ground_truth_dag: nx.DiGraph) -> MetricResult:
        result = self.func(estimated_dag, ground_truth_dag)
        if isinstance(result, MetricResult):
            return result
        return MetricResult(name=getattr(self.func, "__name__", "unknown"), value=float(result) if isinstance(result, (int, float)) else result, metadata={})
    def get_name(self) -> str:
        return getattr(self.func, "__name__", "wrapped_metric")