import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import time


@dataclass
class SolveMetrics:
    solved: bool = False
    is_valid: bool = False

    expanded_nodes: int = 0
    backtracks: int = 0
    propagation_steps: int = 0

    solve_time_ms: float = 0.0
    model_time_ms: float = 0.0
    heuristic_time_ms: float = 0.0

    model_calls: int = 0
    avg_entropy: float = 0.0
    fallback_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "solved": self.solved,
            "is_valid": self.is_valid,
            "expanded_nodes": self.expanded_nodes,
            "backtracks": self.backtracks,
            "propagation_steps": self.propagation_steps,
            "solve_time_ms": round(self.solve_time_ms, 2),
            "model_time_ms": round(self.model_time_ms, 2),
            "heuristic_time_ms": round(self.heuristic_time_ms, 2),
            "model_calls": self.model_calls,
            "avg_entropy": round(self.avg_entropy, 4),
            "fallback_count": self.fallback_count
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SolveMetrics":
        return cls(**d)


@dataclass
class BenchmarkResults:
    total_puzzles: int = 0
    solved_count: int = 0

    nodes_mean: float = 0.0
    nodes_median: float = 0.0
    nodes_p95: float = 0.0
    nodes_p99: float = 0.0

    time_mean: float = 0.0
    time_median: float = 0.0
    time_p95: float = 0.0
    time_p99: float = 0.0

    backtracks_mean: float = 0.0
    backtracks_median: float = 0.0

    model_time_mean: float = 0.0
    model_calls_mean: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_puzzles": self.total_puzzles,
            "solved_count": self.solved_count,
            "solve_rate": round(self.solved_count / self.total_puzzles * 100, 2) if self.total_puzzles > 0 else 0,
            "nodes": {
                "mean": round(self.nodes_mean, 2),
                "median": round(self.nodes_median, 2),
                "p95": round(self.nodes_p95, 2),
                "p99": round(self.nodes_p99, 2)
            },
            "time_ms": {
                "mean": round(self.time_mean, 2),
                "median": round(self.time_median, 2),
                "p95": round(self.time_p95, 2),
                "p99": round(self.time_p99, 2)
            },
            "backtracks": {
                "mean": round(self.backtracks_mean, 2),
                "median": round(self.backtracks_median, 2)
            },
            "model": {
                "time_mean_ms": round(self.model_time_mean, 2),
                "calls_mean": round(self.model_calls_mean, 2)
            }
        }

    def __str__(self) -> str:
        d = self.to_dict()
        lines = [
            f"Benchmark Results:",
            f"  Total: {d['total_puzzles']}, Solved: {d['solved_count']} ({d['solve_rate']}%)",
            f"  Nodes: mean={d['nodes']['mean']}, median={d['nodes']['median']}, p95={d['nodes']['p95']}, p99={d['nodes']['p99']}",
            f"  Time (ms): mean={d['time']['mean']}, median={d['time']['median']}, p95={d['time']['p95']}, p99={d['time']['p99']}",
            f"  Backtracks: mean={d['backtracks']['mean']}, median={d['backtracks']['median']}",
            f"  Model: time_mean={d['model']['time_mean_ms']}ms, calls_mean={d['model']['calls_mean']}"
        ]
        return "\n".join(lines)


class MetricsCollector:
    def __init__(self):
        self.metrics_list: List[SolveMetrics] = []
        self._start_time: Optional[float] = None
        self._current_metrics: Optional[SolveMetrics] = None

    def start_solve(self):
        self._start_time = time.time()
        self._current_metrics = SolveMetrics()

    def end_solve(self, solved: bool, is_valid: bool):
        if self._current_metrics is None:
            return

        self._current_metrics.solved = solved
        self._current_metrics.is_valid = is_valid
        self._current_metrics.solve_time_ms = (time.time() - self._start_time) * 1000

        self.metrics_list.append(self._current_metrics)
        self._current_metrics = None
        self._start_time = None

    def increment_nodes(self):
        if self._current_metrics:
            self._current_metrics.expanded_nodes += 1

    def increment_backtracks(self):
        if self._current_metrics:
            self._current_metrics.backtracks += 1

    def increment_propagation(self):
        if self._current_metrics:
            self._current_metrics.propagation_steps += 1

    def add_model_time(self, time_ms: float):
        if self._current_metrics:
            self._current_metrics.model_time_ms += time_ms
            self._current_metrics.model_calls += 1

    def add_heuristic_time(self, time_ms: float):
        if self._current_metrics:
            self._current_metrics.heuristic_time_ms += time_ms

    def set_entropy(self, entropy: float):
        if self._current_metrics:
            self._current_metrics.avg_entropy = entropy

    def increment_fallback(self):
        if self._current_metrics:
            self._current_metrics.fallback_count += 1

    def increment_model_calls(self):
        if self._current_metrics:
            self._current_metrics.model_calls += 1

    def compute_benchmark_results(self) -> BenchmarkResults:
        if not self.metrics_list:
            return BenchmarkResults()

        solved_metrics = [m for m in self.metrics_list if m.solved]

        nodes = np.array([m.expanded_nodes for m in solved_metrics])
        times = np.array([m.solve_time_ms for m in solved_metrics])
        backtracks = np.array([m.backtracks for m in solved_metrics])
        model_times = np.array([m.model_time_ms for m in solved_metrics])
        model_calls = np.array([m.model_calls for m in solved_metrics])

        return BenchmarkResults(
            total_puzzles=len(self.metrics_list),
            solved_count=len(solved_metrics),
            nodes_mean=float(np.mean(nodes)) if len(nodes) > 0 else 0,
            nodes_median=float(np.median(nodes)) if len(nodes) > 0 else 0,
            nodes_p95=float(np.percentile(nodes, 95)) if len(nodes) > 0 else 0,
            nodes_p99=float(np.percentile(nodes, 99)) if len(nodes) > 0 else 0,
            time_mean=float(np.mean(times)) if len(times) > 0 else 0,
            time_median=float(np.median(times)) if len(times) > 0 else 0,
            time_p95=float(np.percentile(times, 95)) if len(times) > 0 else 0,
            time_p99=float(np.percentile(times, 99)) if len(times) > 0 else 0,
            backtracks_mean=float(np.mean(backtracks)) if len(backtracks) > 0 else 0,
            backtracks_median=float(np.median(backtracks)) if len(backtracks) > 0 else 0,
            model_time_mean=float(np.mean(model_times)) if len(model_times) > 0 else 0,
            model_calls_mean=float(np.mean(model_calls)) if len(model_calls) > 0 else 0
        )

    def save_results(self, filepath: str):
        results = self.compute_benchmark_results()
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

    def clear(self):
        self.metrics_list.clear()
        self._current_metrics = None
        self._start_time = None
