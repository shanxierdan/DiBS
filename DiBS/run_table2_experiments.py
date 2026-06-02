#!/usr/bin/env python3
"""
Table 2 Experiment Runner - 多数据集泛化对比
在不同来源、不同难度分布的数据集上对比 MRV 和 DiBS

支持多 GPU 并行：
  python3 run_table2_experiments.py --gpus "0,1,2,3" --solvers "DiBS"
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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from DiBS.solver import DiBSSolver, BaselineSolver

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "prepared_data"
OUTPUT_DIR = PROJECT_ROOT / "DiBS" / "results" / "parallel" / "Table_2"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "diffusion-vs-ar" / "output" / "sudoku" / "royle17-20260323-210104"

DATASETS = {
    "Hardest-1106": {"path": "royle_forum_hardest_1106.txt", "max_puzzles": None},
    "Top1465": {"path": "royle_magictour_top1465.txt", "max_puzzles": None},
    "Serg-10k": {"path": "royle_serg_benchmark.txt", "max_puzzles": None},
    "SATNet-10k": {"path": "satnet_puzzles.txt", "max_puzzles": None},
    "Kaggle-10k": {"path": "kaggle_puzzles.txt", "max_puzzles": 10000},
    "Hardest-11plus": {"path": "royle_forum_hardest_11plus.txt", "max_puzzles": 10000},
    "Hardest-1905": {"path": "royle_forum_hardest_1905.txt", "max_puzzles": 10000},
    "Royle-Kaggle": {"path": "royle_kaggle.txt", "max_puzzles": 10000},
    "Unbiased-10k": {"path": "royle_unbiased.txt", "max_puzzles": 10000},
}


@dataclass
class InstanceResult:
    run_id: str
    dataset: str
    instance_id: int
    puzzle: str
    givens: int
    solver_name: str
    status: str
    solution: Optional[str]
    valid: bool
    time_ms: float
    nodes: int
    backtracks: int
    propagation_steps: int = 0
    model_calls: int = 0
    model_time_ms: float = 0.0
    error: str = ""


def count_givens(puzzle: str) -> int:
    return sum(1 for c in puzzle if c not in '0.')


def load_puzzles(filepath: str, max_puzzles: Optional[int] = None) -> List[str]:
    puzzles = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) >= 81 and not line.startswith('#'):
                puzzle = line[:81].replace('.', '0')
                puzzles.append(puzzle)
    if max_puzzles and len(puzzles) > max_puzzles:
        puzzles = puzzles[:max_puzzles]
    return puzzles


def solve_mrv(puzzle: str, instance_id: int, dataset: str) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=True, max_nodes=10000000, timeout_ms=float('inf'))
    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > 30000:
        status = "timeout"
    return InstanceResult(
        run_id="", dataset=dataset, instance_id=instance_id, puzzle=puzzle,
        givens=count_givens(puzzle), solver_name="MRV", status=status,
        solution=solution, valid=metrics.is_valid, time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes, backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps
    )


# Global model instance for process-level caching (avoids reloading in multiprocessing)
_dibs_solver_instance = None

def solve_dibs(puzzle: str, instance_id: int, dataset: str, model_path: str, gpu_id: int = None) -> InstanceResult:
    global _dibs_solver_instance

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.init()

    # Initialize solver once per process (lazy initialization)
    if _dibs_solver_instance is None:
        _dibs_solver_instance = DiBSSolver(
            model_path=model_path,
            use_heuristic=True,
            use_lcv=False,
            use_fc=True,
            timeout_ms=float('inf')
        )
    else:
        # Clear previous puzzle state but reuse loaded model
        _dibs_solver_instance.clear_metrics()

    start_time = time.perf_counter()
    solution, metrics = _dibs_solver_instance.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > 30000:
        status = "timeout"
    return InstanceResult(
        run_id="", dataset=dataset, instance_id=instance_id, puzzle=puzzle,
        givens=count_givens(puzzle), solver_name="DiBS", status=status,
        solution=solution, valid=metrics.is_valid, time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes, backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls, model_time_ms=metrics.model_time_ms
    )


def run_solver_on_puzzle(args):
    solver_name, puzzle, instance_id, dataset, model_path, gpu_id = args
    try:
        if solver_name == "MRV":
            return solve_mrv(puzzle, instance_id, dataset)
        elif solver_name == "DiBS":
            return solve_dibs(puzzle, instance_id, dataset, model_path, gpu_id)
    except Exception as e:
        return InstanceResult(
            run_id="", dataset=dataset, instance_id=instance_id, puzzle=puzzle,
            givens=count_givens(puzzle), solver_name=solver_name, status="error",
            solution=None, valid=False, time_ms=0, nodes=0, backtracks=0, error=str(e)
        )


def run_experiment(dataset_name: str, puzzles: List[str], solver_name: str,
                   model_path: str, num_workers: int, run_id: str,
                   gpus: Optional[List[int]] = None) -> List[InstanceResult]:

    if solver_name == "DiBS" and gpus:
        effective_workers = len(gpus)
        print(f"\n[INFO] DiBS solver uses {effective_workers} GPUs: {gpus}")
    elif solver_name == "DiBS":
        effective_workers = 1
        print(f"\n[INFO] DiBS solver uses single process (no GPU specified)")
    else:
        effective_workers = num_workers

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} ({len(puzzles)} puzzles) | Solver: {solver_name}")
    print(f"{'='*60}")

    if gpus and solver_name == "DiBS":
        tasks = [
            (solver_name, puzzle, i, dataset_name, model_path, gpus[i % len(gpus)])
            for i, puzzle in enumerate(puzzles)
        ]
    else:
        tasks = [
            (solver_name, puzzle, i, dataset_name, model_path, None)
            for i, puzzle in enumerate(puzzles)
        ]

    results = []
    solved_count = 0
    total_puzzles = len(puzzles)

    print(f"Starting at {datetime.now().strftime('%H:%M:%S')} (workers: {effective_workers})")
    start_time = time.time()

    if effective_workers <= 1:
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
        mp_ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=mp_ctx) as executor:
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

    print()
    results.sort(key=lambda x: x.instance_id)
    return results


def compute_summary(results: List[InstanceResult], solver_name: str, dataset_name: str) -> Dict:
    solved = [r for r in results if r.status == "solved"]
    timeout = [r for r in results if r.status == "timeout"]
    total = len(results)
    times = [r.time_ms for r in results] if results else [0]
    nodes = [r.nodes for r in solved] if solved else [0]
    backtracks = [r.backtracks for r in solved] if solved else [0]
    model_calls = [r.model_calls for r in solved] if solved else [0]
    model_times = [r.model_time_ms for r in solved] if solved else [0]

    return {
        "dataset": dataset_name,
        "solver": solver_name,
        "total_puzzles": total,
        "solved_count": len(solved),
        "solved_pct": len(solved) / total * 100 if total > 0 else 0,
        "timeout_count": len(timeout),
        "timeout_pct": len(timeout) / total * 100 if total > 0 else 0,
        "time_ms": {
            "mean": float(np.mean(times)) if times else 0,
            "median": float(np.median(times)) if times else 0,
            "p95": float(np.percentile(times, 95)) if times else 0,
        },
        "nodes": {"mean": float(np.mean(nodes)) if nodes else 0, "median": float(np.median(nodes)) if nodes else 0},
        "backtracks": {"mean": float(np.mean(backtracks)) if backtracks else 0},
        "model_calls": {"mean": float(np.mean(model_calls)) if model_calls else 0},
        "model_time_ms": {"mean": float(np.mean(model_times)) if model_times else 0},
    }


def save_results(results: List[InstanceResult], output_dir: Path, run_id: str,
                  dataset_name: str, solver_name: str) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{run_id}_{dataset_name}_{solver_name.replace('+', '_')}.jsonl"
    with open(jsonl_path, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')

    summary = compute_summary(results, solver_name, dataset_name)

    summary_path = output_dir / f"{run_id}_{dataset_name}_{solver_name.replace('+', '_')}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    total = len(results)
    print(f"\n{solver_name} on {dataset_name}:")
    print(f"  Solved: {summary['solved_count']}/{total} ({summary['solved_pct']:.2f}%)")
    print(f"  Timeout: {summary['timeout_count']}/{total} ({summary['timeout_pct']:.2f}%)")
    print(f"  Mean time: {summary['time_ms']['mean']:.2f}ms, P95: {summary['time_ms']['p95']:.2f}ms")
    if solver_name == "DiBS":
        print(f"  Model calls: {summary['model_calls']['mean']:.1f}")

    return summary


def compute_comparison(summaries: List[Dict]) -> Dict:
    comparison = {}
    datasets = set(s['dataset'] for s in summaries)

    for dataset in datasets:
        mrv = next((s for s in summaries if s['dataset'] == dataset and s['solver'] == 'MRV'), None)
        dibs = next((s for s in summaries if s['dataset'] == dataset and s['solver'] == 'DiBS'), None)

        if mrv and dibs:
            mrv_time = mrv['time_ms']['mean']
            dibs_time = dibs['time_ms']['mean']
            mrv_nodes = mrv['nodes']['mean']
            dibs_nodes = dibs['nodes']['mean']

            speedup = mrv_time / dibs_time if dibs_time > 0 else 0
            node_red = (mrv_nodes - dibs_nodes) / mrv_nodes * 100 if mrv_nodes > 0 else 0

            comparison[dataset] = {
                "speedup": speedup,
                "node_reduction_pct": node_red,
            }

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Table 2 - Multi-dataset Generalization")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs for DiBS multi-GPU parallel (e.g., '0,1,2,3')")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--solvers", type=str, default="MRV,DiBS")
    parser.add_argument("--max-puzzles", type=int, default=None, help="Max puzzles per dataset (for testing)")

    args = parser.parse_args()

    if args.workers is None:
        args.workers = min(multiprocessing.cpu_count(), 32)

    gpus = None
    if args.gpus:
        gpus = [int(g.strip()) for g in args.gpus.split(",")]
        print(f"[INFO] GPUs specified for DiBS: {gpus}")

    datasets_to_run = list(DATASETS.keys()) if args.datasets == "all" else [d.strip() for d in args.datasets.split(",")]
    solvers_to_run = [s.strip() for s in args.solvers.split(",")]

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {"run_id": run_id, "timestamp": datetime.now().isoformat(), "datasets": datasets_to_run, "solvers": solvers_to_run, "gpus": gpus}
    with open(output_dir / f"{run_id}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    all_summaries = []

    for dataset_name in datasets_to_run:
        if dataset_name not in DATASETS:
            print(f"Warning: Unknown dataset {dataset_name}")
            continue

        dataset_info = DATASETS[dataset_name]
        dataset_path = DATA_DIR / dataset_info['path']

        if not dataset_path.exists():
            print(f"Warning: File not found {dataset_path}")
            continue

        max_puzzles = args.max_puzzles if args.max_puzzles else dataset_info['max_puzzles']
        puzzles = load_puzzles(str(dataset_path), max_puzzles)
        print(f"\nLoaded {len(puzzles)} puzzles from {dataset_name}")

        for solver_name in solvers_to_run:
            results = run_experiment(dataset_name, puzzles, solver_name, args.model, args.workers, run_id, gpus)
            summary = save_results(results, output_dir, run_id, dataset_name, solver_name)
            all_summaries.append(summary)

    comparison = compute_comparison(all_summaries)
    with open(output_dir / f"{run_id}_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*80)
    print("TABLE 2 RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Dataset':<15} {'Solver':<10} {'Solved%':>8} {'Time Mean':>12} {'Time P95':>12} {'Nodes':>12}")
    print("-"*80)
    for s in all_summaries:
        print(f"{s['dataset']:<15} {s['solver']:<10} {s['solved_pct']:>7.2f}% {s['time_ms']['mean']:>11.1f}ms {s['time_ms']['p95']:>11.1f}ms {s['nodes']['mean']:>12.0f}")

    print("\n" + "="*80)
    print("SPEEDUP (DiBS vs MRV)")
    print("="*80)
    print(f"\n{'Dataset':<15} {'Speedup':>10} {'Node Red%':>12}")
    print("-"*40)
    for dataset, comp in comparison.items():
        print(f"{dataset:<15} {comp['speedup']:>9.2f}x {comp['node_reduction_pct']:>11.2f}%")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
