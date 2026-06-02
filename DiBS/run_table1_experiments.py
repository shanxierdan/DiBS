#!/usr/bin/env python3
"""
Table 1 Experiment Runner
Runs all baseline solvers on Royle 17-clue dataset and generates JSONL logs
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

    sat_decisions: int = 0
    sat_conflicts: int = 0
    sat_propagations: int = 0

    milp_bb_nodes: int = 0
    milp_lp_iters: int = 0

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


def solve_cp_mrv(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=False, max_nodes=10000000)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else ("timeout" if elapsed_ms >= timeout_ms else "failed")

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
        propagation_steps=metrics.propagation_steps
    )


def solve_cp_mrv_fc(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=True, max_nodes=10000000)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else ("timeout" if elapsed_ms >= timeout_ms else "failed")

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
        propagation_steps=metrics.propagation_steps
    )


def solve_cp_mrv_lcv(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=True, use_fc=False, max_nodes=10000000)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else ("timeout" if elapsed_ms >= timeout_ms else "failed")

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
        propagation_steps=metrics.propagation_steps
    )


def solve_cp_mrv_fc_lcv(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=True, use_fc=True, max_nodes=10000000)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else ("timeout" if elapsed_ms >= timeout_ms else "failed")

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
        propagation_steps=metrics.propagation_steps
    )


def solve_cp_mrv_degree(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=False, use_degree_tiebreak=True, max_nodes=10000000)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else ("timeout" if elapsed_ms >= timeout_ms else "failed")

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
        propagation_steps=metrics.propagation_steps
    )


def solve_cp_mrv_fc_degree(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=True, use_degree_tiebreak=True, max_nodes=10000000)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else ("timeout" if elapsed_ms >= timeout_ms else "failed")

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
        propagation_steps=metrics.propagation_steps
    )


def solve_cp_mrv_fc_lcv_degree(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    solver = BaselineSolver(use_lcv=True, use_fc=True, use_degree_tiebreak=True, max_nodes=10000000)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else ("timeout" if elapsed_ms >= timeout_ms else "failed")

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
        propagation_steps=metrics.propagation_steps
    )


# Global model instance for process-level caching (avoids reloading in multiprocessing)
_dibs_solver_instance = None

def solve_dibs(puzzle: str, instance_id: int, model_path: str, timeout_ms: float = 30000) -> InstanceResult:
    global _dibs_solver_instance

    # Initialize solver once per process (lazy initialization)
    if _dibs_solver_instance is None:
        _dibs_solver_instance = DiBSSolver(
            model_path=model_path,
            use_heuristic=True,
            use_lcv=False
        )
    else:
        # Clear previous puzzle state but reuse loaded model
        _dibs_solver_instance.clear_metrics()

    start_time = time.perf_counter()
    solution, metrics = _dibs_solver_instance.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.solved else ("timeout" if elapsed_ms >= timeout_ms else "failed")

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


def solve_dlx(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "dlx"))
    from dlx_solver import DLXSolver

    solver = DLXSolver(timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.get('solved', False) else ("timeout" if metrics.get('timeout', False) else "failed")

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="Exact Cover",
        solver_name="DLX",
        status=status,
        solution=solution,
        valid=metrics.get('is_valid', False),
        time_ms=metrics.get('time_ms', elapsed_ms),
        nodes=metrics.get('nodes', 0),
        backtracks=metrics.get('backtracks', 0)
    )


def solve_sat(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "sat"))
    from sat_solver import SATSolver

    solver = SATSolver(solver_name='glucose4', timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.get('solved', False) else "failed"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="SAT",
        solver_name="CDCL (Glucose4)",
        status=status,
        solution=solution,
        valid=metrics.get('is_valid', False),
        time_ms=metrics.get('time_ms', elapsed_ms),
        nodes=metrics.get('decisions', 0),
        backtracks=metrics.get('conflicts', 0),
        sat_decisions=metrics.get('decisions', 0),
        sat_conflicts=metrics.get('conflicts', 0),
        sat_propagations=metrics.get('propagations', 0)
    )


def solve_milp(puzzle: str, instance_id: int, timeout_ms: float = 30000) -> InstanceResult:
    sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "milp"))
    from milp_solver import MILPSolver

    solver = MILPSolver(timeout_ms=timeout_ms)

    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    status = "solved" if metrics.get('solved', False) else "failed"

    return InstanceResult(
        run_id="",
        instance_id=instance_id,
        puzzle=puzzle,
        givens=count_givens(puzzle),
        solver_family="MILP",
        solver_name="B&B (CBC)",
        status=status,
        solution=solution,
        valid=metrics.get('is_valid', False),
        time_ms=metrics.get('time_ms', elapsed_ms),
        nodes=metrics.get('bb_nodes', 0),
        backtracks=metrics.get('bb_nodes', 0),
        milp_bb_nodes=metrics.get('bb_nodes', 0),
        milp_lp_iters=metrics.get('lp_iters', 0)
    )


def run_solver_on_puzzle(args):
    solver_name, puzzle, instance_id, model_path, timeout_ms = args

    try:
        if solver_name == "MRV":
            result = solve_cp_mrv(puzzle, instance_id, timeout_ms)
        elif solver_name == "MRV+FC":
            result = solve_cp_mrv_fc(puzzle, instance_id, timeout_ms)
        elif solver_name == "MRV+LCV":
            result = solve_cp_mrv_lcv(puzzle, instance_id, timeout_ms)
        elif solver_name == "MRV+FC+LCV":
            result = solve_cp_mrv_fc_lcv(puzzle, instance_id, timeout_ms)
        elif solver_name == "MRV+Degree":
            result = solve_cp_mrv_degree(puzzle, instance_id, timeout_ms)
        elif solver_name == "MRV+FC+Degree":
            result = solve_cp_mrv_fc_degree(puzzle, instance_id, timeout_ms)
        elif solver_name == "MRV+FC+LCV+Degree":
            result = solve_cp_mrv_fc_lcv_degree(puzzle, instance_id, timeout_ms)
        elif solver_name == "DiBS":
            result = solve_dibs(puzzle, instance_id, model_path, timeout_ms)
        elif solver_name == "DLX":
            result = solve_dlx(puzzle, instance_id, timeout_ms)
        elif solver_name == "SAT":
            result = solve_sat(puzzle, instance_id, timeout_ms)
        elif solver_name == "MILP":
            result = solve_milp(puzzle, instance_id, timeout_ms)
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


def run_experiment(
    solver_name: str,
    puzzles: List[str],
    model_path: str,
    timeout_ms: float,
    num_workers: int,
    run_id: str,
    start_idx: int = 0
) -> List[InstanceResult]:

    print(f"\n{'='*60}")
    print(f"Running {solver_name} on {len(puzzles)} puzzles")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}")

    tasks = [
        (solver_name, puzzle, start_idx + i, model_path, timeout_ms)
        for i, puzzle in enumerate(puzzles)
    ]

    results = []

    if num_workers <= 1:
        for task in tasks:
            result = run_solver_on_puzzle(task)
            result.run_id = run_id
            results.append(result)

            if len(results) % 100 == 0:
                solved = sum(1 for r in results if r.status == "solved")
                print(f"  Progress: {len(results)}/{len(puzzles)}, Solved: {solved}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(run_solver_on_puzzle, task): task for task in tasks}

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                result.run_id = run_id
                results.append(result)

                completed += 1
                if completed % 100 == 0:
                    solved = sum(1 for r in results if r.status == "solved")
                    print(f"  Progress: {completed}/{len(puzzles)}, Solved: {solved}")

    results.sort(key=lambda x: x.instance_id)
    return results


def save_results(results: List[InstanceResult], output_dir: Path, run_id: str, solver_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{run_id}_{solver_name.replace('+', '_')}.jsonl"
    with open(jsonl_path, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')

    solved = [r for r in results if r.status == "solved"]

    times = [r.time_ms for r in solved] if solved else [0]
    nodes = [r.nodes for r in solved] if solved else [0]
    backtracks = [r.backtracks for r in solved] if solved else [0]
    model_calls = [r.model_calls for r in solved] if solved else [0]
    model_times = [r.model_time_ms for r in solved] if solved else [0]

    import numpy as np
    summary = {
        "run_id": run_id,
        "solver_family": results[0].solver_family if results else "Unknown",
        "solver_name": solver_name,
        "total_puzzles": len(results),
        "solved_count": len(solved),
        "solved_pct": len(solved) / len(results) * 100 if results else 0,
        "time_ms": {
            "avg": float(np.mean(times)) if times else 0,
            "med": float(np.median(times)) if times else 0,
            "p95": float(np.percentile(times, 95)) if times else 0,
        },
        "nodes": {
            "avg": float(np.mean(nodes)) if nodes else 0,
            "med": float(np.median(nodes)) if nodes else 0,
            "p95": float(np.percentile(nodes, 95)) if nodes else 0,
        },
        "backtracks": {
            "avg": float(np.mean(backtracks)) if backtracks else 0,
            "med": float(np.median(backtracks)) if backtracks else 0,
            "p95": float(np.percentile(backtracks, 95)) if backtracks else 0,
        },
        "model_calls": {
            "avg": float(np.mean(model_calls)) if model_calls else 0,
            "med": float(np.median(model_calls)) if model_calls else 0,
            "p95": float(np.percentile(model_calls, 95)) if model_calls else 0,
        },
        "model_time_ms": {
            "avg": float(np.mean(model_times)) if model_times else 0,
            "med": float(np.median(model_times)) if model_times else 0,
            "p95": float(np.percentile(model_times, 95)) if model_times else 0,
        }
    }

    summary_path = output_dir / f"{run_id}_{solver_name.replace('+', '_')}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{solver_name} Results:")
    print(f"  Solved: {len(solved)}/{len(results)} ({len(solved)/len(results)*100:.1f}%)")
    print(f"  Avg time: {summary['time_ms']['avg']:.2f}ms")
    print(f"  Avg nodes: {summary['nodes']['avg']:.1f}")
    print(f"  Avg backtracks: {summary['backtracks']['avg']:.1f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Table 1 experiments")
    parser.add_argument("--puzzles", type=str, default=str(DATASET_PATH),
                        help="Path to puzzle file")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to DiBS model")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--timeout", type=int, default=30000,
                        help="Timeout per puzzle in ms")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--max-puzzles", type=int, default=None,
                        help="Maximum puzzles to solve (for testing)")
    parser.add_argument("--solvers", type=str, default="all",
                        help="Comma-separated list of solvers to run")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID for DiBS")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index of puzzles to process")
    parser.add_argument("--end-idx", type=int, default=-1,
                        help="End index of puzzles to process (exclusive, -1 for all)")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.workers is None:
        args.workers = min(multiprocessing.cpu_count(), 32)

    all_solvers = ["MRV", "MRV+FC", "MRV+FC+LCV", "DiBS"]
    if args.solvers == "all":
        solvers_to_run = all_solvers
    else:
        solvers_to_run = [s.strip() for s in args.solvers.split(",")]

    puzzles = load_puzzles(args.puzzles, args.max_puzzles, args.start_idx, args.end_idx)
    print(f"Loaded {len(puzzles)} puzzles (range: {args.start_idx}-{args.end_idx if args.end_idx > 0 else 'end'})")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "puzzle_file": args.puzzles,
        "num_puzzles": len(puzzles),
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "timeout_ms": args.timeout,
        "num_workers": args.workers,
        "model_path": args.model,
        "solvers": solvers_to_run
    }

    with open(output_dir / f"{run_id}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    summaries = []

    for solver_name in solvers_to_run:
        results = run_experiment(
            solver_name=solver_name,
            puzzles=puzzles,
            model_path=args.model,
            timeout_ms=args.timeout,
            num_workers=args.workers,
            run_id=run_id,
            start_idx=args.start_idx
        )

        summary = save_results(results, output_dir, run_id, solver_name)
        summaries.append(summary)

    print("\n" + "="*60)
    print("Experiment completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
