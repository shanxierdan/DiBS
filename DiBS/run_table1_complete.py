#!/usr/bin/env python3
"""
Table 1 Experiment Runner - 完整版 (无超时限制)
用于获取真正的解出率，并输出完整的统计信息

设计思路：
- 不设置超时限制，让所有求解器尽可能解出更多题目
- 记录每道题的求解时间和状态
- 统计解出率 (solved_pct)
- 输出完整的 mean/median/p95 统计
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent.parent))

from DiBS.solver import DiBSSolver, BaselineSolver
from DiBS.constraints import SudokuConstraints

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "prepared_data" / "royle_17clue.txt"
OUTPUT_DIR = PROJECT_ROOT / "DiBS" / "results" / "parallel" / "Table_1"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "diffusion-vs-ar" / "output" / "sudoku" / "royle17-20260323-210104"


@dataclass
class InstanceResult:
    run_id: str
    instance_id: int
    puzzle: str
    givens: int

    solver_family: str
    solver_name: str

    status: str
    solution: Optional[str]
    valid: bool

    time_ms: float
    nodes: int
    backtracks: int

    propagation_steps: int = 0
    max_depth: int = 0

    model_calls: int = 0
    model_time_ms: float = 0.0

    error: str = ""


def count_givens(puzzle: str) -> int:
    return sum(1 for c in puzzle if c not in '0.')


def load_puzzles(filepath: str, max_puzzles: Optional[int] = None,
                 start_idx: int = 0, end_idx: int = -1) -> List[str]:
    puzzles = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) >= 81 and not line.startswith('#'):
                puzzle = line[:81].replace('.', '0')
                puzzles.append(puzzle)

    if end_idx < 0:
        end_idx = len(puzzles)

    puzzles = puzzles[start_idx:end_idx]

    if max_puzzles and len(puzzles) > max_puzzles:
        puzzles = puzzles[:max_puzzles]

    return puzzles


def solve_cp_mrv(puzzle: str, instance_id: int, max_nodes: int = 10000000,
                 timeout_ms: float = float('inf'),
                 eval_timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=False, max_nodes=max_nodes, timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > eval_timeout_ms:
        status = "timeout"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="CP",
        solver_name="MRV",
        status=status,
        solution=solution,
        valid=metrics.is_valid,
        time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes,
        backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls,
        model_time_ms=metrics.model_time_ms
    )


def solve_cp_mrv_fc(puzzle: str, instance_id: int, max_nodes: int = 10000000,
                    timeout_ms: float = float('inf'),
                    eval_timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=True, max_nodes=max_nodes, timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > eval_timeout_ms:
        status = "timeout"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="CP",
        solver_name="MRV+FC",
        status=status,
        solution=solution,
        valid=metrics.is_valid,
        time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes,
        backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls,
        model_time_ms=metrics.model_time_ms
    )


def solve_cp_mrv_lcv(puzzle: str, instance_id: int, max_nodes: int = 10000000,
                     timeout_ms: float = float('inf'),
                     eval_timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=True, use_fc=False, max_nodes=max_nodes, timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > eval_timeout_ms:
        status = "timeout"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="CP",
        solver_name="MRV+LCV",
        status=status,
        solution=solution,
        valid=metrics.is_valid,
        time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes,
        backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls,
        model_time_ms=metrics.model_time_ms
    )


def solve_cp_mrv_fc_lcv(puzzle: str, instance_id: int, max_nodes: int = 10000000,
                        timeout_ms: float = float('inf'),
                        eval_timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=True, use_fc=True, max_nodes=max_nodes, timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > eval_timeout_ms:
        status = "timeout"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="CP",
        solver_name="MRV+FC+LCV",
        status=status,
        solution=solution,
        valid=metrics.is_valid,
        time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes,
        backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls,
        model_time_ms=metrics.model_time_ms
    )


def solve_cp_mrv_degree(puzzle: str, instance_id: int, max_nodes: int = 10000000,
                        timeout_ms: float = float('inf'),
                        eval_timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=False, use_degree_tiebreak=True, max_nodes=max_nodes, timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > eval_timeout_ms:
        status = "timeout"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="CP",
        solver_name="MRV+Degree",
        status=status,
        solution=solution,
        valid=metrics.is_valid,
        time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes,
        backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls,
        model_time_ms=metrics.model_time_ms
    )


def solve_cp_mrv_fc_degree(puzzle: str, instance_id: int, max_nodes: int = 10000000,
                           timeout_ms: float = float('inf'),
                           eval_timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=True, use_degree_tiebreak=True, max_nodes=max_nodes, timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > eval_timeout_ms:
        status = "timeout"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="CP",
        solver_name="MRV+FC+Degree",
        status=status,
        solution=solution,
        valid=metrics.is_valid,
        time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes,
        backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls,
        model_time_ms=metrics.model_time_ms
    )


def solve_cp_mrv_fc_lcv_degree(puzzle: str, instance_id: int, max_nodes: int = 10000000,
                               timeout_ms: float = float('inf'),
                               eval_timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=True, use_fc=True, use_degree_tiebreak=True, max_nodes=max_nodes, timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > eval_timeout_ms:
        status = "timeout"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="CP",
        solver_name="MRV+FC+LCV+Degree",
        status=status,
        solution=solution,
        valid=metrics.is_valid,
        time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes,
        backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls,
        model_time_ms=metrics.model_time_ms
    )


# Global model instance for process-level caching (avoids reloading in multiprocessing)
_dibs_solver_instance = None

def solve_dibs(puzzle: str, instance_id: int, model_path: str, max_nodes: int = 10000000,
               timeout_ms: float = float('inf'),
               eval_timeout_ms: float = 30000) -> InstanceResult:
    global _dibs_solver_instance

    # Initialize solver once per process (lazy initialization)
    if _dibs_solver_instance is None:
        _dibs_solver_instance = DiBSSolver(
            model_path=model_path,
            use_heuristic=True,
            use_lcv=False,
            use_fc=True,
            timeout_ms=timeout_ms
        )
    else:
        # Clear previous puzzle state but reuse loaded model
        _dibs_solver_instance.clear_metrics()

    start_time = time.perf_counter()
    solution, metrics = _dibs_solver_instance.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > eval_timeout_ms:
        status = "timeout"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="CP",
        solver_name="DiBS",
        status=status,
        solution=solution,
        valid=metrics.is_valid,
        time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes,
        backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls,
        model_time_ms=metrics.model_time_ms
    )


def run_solver_on_puzzle(args):
    solver_name, puzzle, instance_id, model_path, max_nodes = args

    try:
        if solver_name == "MRV":
            result = solve_cp_mrv(puzzle, instance_id, max_nodes)
        elif solver_name == "MRV+FC":
            result = solve_cp_mrv_fc(puzzle, instance_id, max_nodes)
        elif solver_name == "MRV+LCV":
            result = solve_cp_mrv_lcv(puzzle, instance_id, max_nodes)
        elif solver_name == "MRV+FC+LCV":
            result = solve_cp_mrv_fc_lcv(puzzle, instance_id, max_nodes)
        elif solver_name == "MRV+Degree":
            result = solve_cp_mrv_degree(puzzle, instance_id, max_nodes)
        elif solver_name == "MRV+FC+Degree":
            result = solve_cp_mrv_fc_degree(puzzle, instance_id, max_nodes)
        elif solver_name == "MRV+FC+LCV+Degree":
            result = solve_cp_mrv_fc_lcv_degree(puzzle, instance_id, max_nodes)
        elif solver_name == "DiBS":
            result = solve_dibs(puzzle, instance_id, model_path, max_nodes)
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        return result
    except Exception as e:
        return InstanceResult(
            run_id="",
            instance_id=instance_id,
            puzzle=puzzle,
            givens=count_givens(puzzle),
            solver_family="Unknown",
            solver_name=solver_name,
            status="error",
            solution=None,
            valid=False,
            time_ms=0,
            nodes=0,
            backtracks=0,
            error=str(e)
        )


def _init_worker_process():
    """Initialize worker process with spawn method for CUDA compatibility"""
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)


def run_experiment(
    solver_name: str,
    puzzles: List[str],
    model_path: str,
    max_nodes: int,
    num_workers: int,
    run_id: str,
    start_idx: int = 0
) -> List[InstanceResult]:

    print(f"\n{'='*60}")
    print(f"Running {solver_name} on {len(puzzles)} puzzles (no timeout)")
    print(f"Workers: {num_workers}, Max nodes: {max_nodes}")
    print(f"{'='*60}")

    tasks = [
        (solver_name, puzzle, start_idx + i, model_path, max_nodes)
        for i, puzzle in enumerate(puzzles)
    ]

    results = []
    solved_count = 0
    total_puzzles = len(puzzles)

    print(f"\nStarting at {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    start_time = time.time()

    if num_workers <= 1:
        for i, task in enumerate(tasks):
            result = run_solver_on_puzzle(task)
            result.run_id = run_id
            results.append(result)

            if result.status == "solved":
                solved_count += 1

            if (i + 1) % 10 == 0 or (i + 1) == total_puzzles:
                pct = (i + 1) / total_puzzles * 100
                bar_len = 30
                filled = int(bar_len * (i + 1) / total_puzzles)
                bar = "█" * filled + "░" * (bar_len - filled)
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (total_puzzles - i - 1) if i > 0 else 0
                print(f"  [{bar}] {i+1}/{total_puzzles} ({pct:5.1f}%) | Solved: {solved_count} | ETA: {eta:.0f}s", end='\r')
    else:
        start_time = time.time()
        # Use 'spawn' method for CUDA compatibility
        mp_ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
            futures = {executor.submit(run_solver_on_puzzle, task): task for task in tasks}

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                result.run_id = run_id
                results.append(result)

                if result.status == "solved":
                    solved_count += 1

                completed += 1
                if completed % 10 == 0 or completed == total_puzzles:
                    pct = completed / total_puzzles * 100
                    bar_len = 30
                    filled = int(bar_len * completed / total_puzzles)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    elapsed = time.time() - start_time
                    eta = elapsed / completed * (total_puzzles - completed) if completed > 0 else 0
                    print(f"  [{bar}] {completed}/{total_puzzles} ({pct:5.1f}%) | Solved: {solved_count} | ETA: {eta:.0f}s", end='\r')

    print()  # newline after progress bar
    results.sort(key=lambda x: x.instance_id)
    return results


def save_results(results: List[InstanceResult], output_dir: Path, run_id: str, solver_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{run_id}_{solver_name.replace('+', '_').replace(' ', '_')}.jsonl"
    with open(jsonl_path, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')

    solved = [r for r in results if r.status == "solved"]
    timeout = [r for r in results if r.status == "timeout"]
    failed = [r for r in results if r.status == "failed"]
    total = len(results)

    times = [r.time_ms for r in results] if results else [0]
    nodes = [r.nodes for r in solved] if solved else [0]
    backtracks = [r.backtracks for r in solved] if solved else [0]
    model_calls = [r.model_calls for r in solved] if solved else [0]
    model_times = [r.model_time_ms for r in solved] if solved else [0]

    import numpy as np
    summary = {
        "run_id": run_id,
        "solver_family": results[0].solver_family if results else "Unknown",
        "solver_name": solver_name,
        "total_puzzles": total,
        "solved_count": len(solved),
        "solved_pct": len(solved) / total * 100 if total > 0 else 0,
        "timeout_count": len(timeout),
        "timeout_pct": len(timeout) / total * 100 if total > 0 else 0,
        "failed_count": len(failed),
        "time_ms": {
            "mean": float(np.mean(times)) if times else 0,
            "median": float(np.median(times)) if times else 0,
            "p95": float(np.percentile(times, 95)) if times else 0,
            "min": float(np.min(times)) if times else 0,
            "max": float(np.max(times)) if times else 0,
        },
        "nodes": {
            "mean": float(np.mean(nodes)) if nodes else 0,
            "median": float(np.median(nodes)) if nodes else 0,
            "p95": float(np.percentile(nodes, 95)) if nodes else 0,
            "min": float(np.min(nodes)) if nodes else 0,
            "max": float(np.max(nodes)) if nodes else 0,
        },
        "backtracks": {
            "mean": float(np.mean(backtracks)) if backtracks else 0,
            "median": float(np.median(backtracks)) if backtracks else 0,
            "p95": float(np.percentile(backtracks, 95)) if backtracks else 0,
            "min": float(np.min(backtracks)) if backtracks else 0,
            "max": float(np.max(backtracks)) if backtracks else 0,
        },
        "model_calls": {
            "mean": float(np.mean(model_calls)) if model_calls else 0,
            "median": float(np.median(model_calls)) if model_calls else 0,
            "p95": float(np.percentile(model_calls, 95)) if model_calls else 0,
            "min": float(np.min(model_calls)) if model_calls else 0,
            "max": float(np.max(model_calls)) if model_calls else 0,
        },
        "model_time_ms": {
            "mean": float(np.mean(model_times)) if model_times else 0,
            "median": float(np.median(model_times)) if model_times else 0,
            "p95": float(np.percentile(model_times, 95)) if model_times else 0,
            "min": float(np.min(model_times)) if model_times else 0,
            "max": float(np.max(model_times)) if model_times else 0,
        }
    }

    summary_path = output_dir / f"{run_id}_{solver_name.replace('+', '_').replace(' ', '_')}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{solver_name} Results:")
    print(f"  Solved: {len(solved)}/{total} ({len(solved)/total*100:.2f}%)")
    print(f"  Timeout: {len(timeout)}/{total} ({len(timeout)/total*100:.2f}%)")
    print(f"  Mean time: {summary['time_ms']['mean']:.2f}ms")
    print(f"  Median time: {summary['time_ms']['median']:.2f}ms")
    print(f"  P95 time: {summary['time_ms']['p95']:.2f}ms")
    if solver_name == "DiBS":
        print(f"  Mean model calls: {summary['model_calls']['mean']:.1f}")
        print(f"  Mean model time: {summary['model_time_ms']['mean']:.2f}ms")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Table 1 Experiments (No Timeout)")
    parser.add_argument("--puzzles", type=str, default=str(DATASET_PATH),
                        help="Path to puzzle file")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to DiBS model")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--max-nodes", type=int, default=10000000,
                        help="Maximum nodes per puzzle")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--max-puzzles", type=int, default=None,
                        help="Maximum puzzles to solve (for testing)")
    parser.add_argument("--solvers", type=str, default="all",
                        help="Comma-separated list of solvers to run")

    args = parser.parse_args()

    if args.workers is None:
        args.workers = min(multiprocessing.cpu_count(), 32)

    all_solvers = ["MRV", "MRV+FC", "MRV+LCV", "MRV+FC+LCV", "DiBS"]
    if args.solvers == "all":
        solvers_to_run = all_solvers
    else:
        solvers_to_run = [s.strip() for s in args.solvers.split(",")]

    puzzles = load_puzzles(args.puzzles, args.max_puzzles)
    print(f"Loaded {len(puzzles)} puzzles")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "puzzle_file": args.puzzles,
        "num_puzzles": len(puzzles),
        "max_nodes": args.max_nodes,
        "num_workers": args.workers,
        "model_path": args.model,
        "solvers": solvers_to_run,
        "note": "No timeout - all solvers run to completion or max_nodes"
    }

    with open(output_dir / f"{run_id}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    summaries = []

    for solver_name in solvers_to_run:
        results = run_experiment(
            solver_name=solver_name,
            puzzles=puzzles,
            model_path=args.model,
            max_nodes=args.max_nodes,
            num_workers=args.workers,
            run_id=run_id,
            start_idx=0
        )

        summary = save_results(results, output_dir, run_id, solver_name)
        summaries.append(summary)

    print("\n" + "="*60)
    print("Experiment completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

    print("\n" + "="*80)
    print("FINAL TABLE 1 RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Solver':<15} {'Solved%':>10} {'Time Mean':>12} {'Time Med':>12} {'Time P95':>12}")
    print(f"{'':<15} {'':<10} {'Nodes Mean':>12} {'Nodes Med':>12} {'Nodes P95':>12}")
    print(f"{'':<15} {'':<10} {'Back Mean':>12} {'Back Med':>12} {'Back P95':>12}")
    if "DiBS" in solvers_to_run:
        print(f"{'':<15} {'':<10} {'ModelCalls':>12} {'ModelTime':>12}")
    print("-"*80)

    for s in summaries:
        name = s.get('solver_name', 'Unknown')
        solved_pct = s.get('solved_pct', 0)
        time_mean = s.get('time_ms', {}).get('mean', 0)
        time_med = s.get('time_ms', {}).get('median', 0)
        time_p95 = s.get('time_ms', {}).get('p95', 0)
        nodes_mean = s.get('nodes', {}).get('mean', 0)
        nodes_med = s.get('nodes', {}).get('median', 0)
        nodes_p95 = s.get('nodes', {}).get('p95', 0)
        back_mean = s.get('backtracks', {}).get('mean', 0)
        back_med = s.get('backtracks', {}).get('median', 0)
        back_p95 = s.get('backtracks', {}).get('p95', 0)

        print(f"{name:<15} {solved_pct:>9.2f}% {time_mean:>11.1f}ms {time_med:>11.1f}ms {time_p95:>11.1f}ms")
        print(f"{'':<15} {'':<10} {nodes_mean:>11.0f} {nodes_med:>11.0f} {nodes_p95:>11.0f}")
        print(f"{'':<15} {'':<10} {back_mean:>11.0f} {back_med:>11.0f} {back_p95:>11.0f}")

        if name == "DiBS":
            model_calls = s.get('model_calls', {}).get('mean', 0)
            model_time = s.get('model_time_ms', {}).get('mean', 0)
            print(f"{'':<15} {'':<10} {model_calls:>11.1f} {model_time:>11.1f}ms")

        print()

    print("="*80)


if __name__ == "__main__":
    main()
