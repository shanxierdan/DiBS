#!/usr/bin/env python3
"""
Table 3 Experiment Runner - 消融实验与调参
验证 smart-call 与 consistency 项的必要性，以及参数稳定性
"""

import os
import sys
import json
import time
import argparse
import random
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from DiBS.solver import DiBSSolver, BaselineSolver
from DiBS.config import DiBSConfig

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "prepared_data"
MODEL_DATA_DIR = PROJECT_ROOT / "model" / "diffusion-vs-ar" / "data"
OUTPUT_DIR = PROJECT_ROOT / "DiBS" / "results" / "parallel" / "Table_3"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "diffusion-vs-ar" / "output" / "sudoku" / "royle17-20260323-210104"

DATASET = {
    "path": "royle17_test.csv",
    "max_puzzles": 5000,
    "description": "Royle17 held-out test split (sampled)"
}


@dataclass
class InstanceResult:
    run_id: str
    variant: str
    instance_id: int
    puzzle: str
    givens: int
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


def load_puzzles(filepath: str, max_puzzles: Optional[int] = None, seed: int = 42) -> List[str]:
    puzzles = []
    path_obj = Path(filepath)
    if path_obj.suffix.lower() == ".csv":
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                puzzle = str(row.get("quizzes", "")).strip().replace('.', '0')
                if len(puzzle) >= 81:
                    puzzles.append(puzzle[:81])
    else:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) >= 81 and not line.startswith('#'):
                    puzzle = line[:81].replace('.', '0')
                    puzzles.append(puzzle)

    if max_puzzles and len(puzzles) > max_puzzles:
        random.seed(seed)
        puzzles = random.sample(puzzles, max_puzzles)

    return puzzles


def solve_baseline(puzzle: str, instance_id: int) -> InstanceResult:
    solver = BaselineSolver(use_lcv=False, use_fc=True, max_nodes=10000000, timeout_ms=float('inf'))
    start_time = time.perf_counter()
    solution, metrics = solver.solve(puzzle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    status = "solved" if metrics.solved else "failed"
    if elapsed_ms > 30000:
        status = "timeout"
    return InstanceResult(
        run_id="", variant="Base", instance_id=instance_id, puzzle=puzzle,
        givens=count_givens(puzzle), status=status, solution=solution,
        valid=metrics.is_valid, time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes, backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps
    )


# Global model instance for process-level caching (avoids reloading in multiprocessing)
_dibs_solver_instance = None

def solve_dibs_variant(puzzle: str, instance_id: int, model_path: str,
                       variant: str, alpha: float = 0.8,
                       smart_call_threshold: int = 2,
                       gpu_id: int = None) -> InstanceResult:
    global _dibs_solver_instance

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.init()

    # Initialize solver once per process (lazy initialization)
    if _dibs_solver_instance is None:
        config = DiBSConfig(
            alpha=alpha,
            use_heuristic=True,
        )

        always_call = (variant == "always-call")

        _dibs_solver_instance = DiBSSolver(
            model_path=model_path,
            config=config,
            use_heuristic=True,
            use_lcv=False,
            use_fc=True,
            smart_call=not always_call,
            timeout_ms=float('inf')
        )

        if not always_call:
            _dibs_solver_instance.smart_config.min_mrv_threshold = smart_call_threshold

        if variant == "logits-only":
            _dibs_solver_instance.config.alpha = 1.0
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
        run_id="", variant=variant, instance_id=instance_id, puzzle=puzzle,
        givens=count_givens(puzzle), status=status, solution=solution,
        valid=metrics.is_valid, time_ms=elapsed_ms,
        nodes=metrics.expanded_nodes, backtracks=metrics.backtracks,
        propagation_steps=metrics.propagation_steps,
        model_calls=metrics.model_calls, model_time_ms=metrics.model_time_ms
    )


def run_solver_on_puzzle(args):
    solver_type, puzzle, instance_id, model_path, variant, alpha, threshold, gpu_id = args
    try:
        if solver_type == "baseline":
            return solve_baseline(puzzle, instance_id)
        elif solver_type == "dibs":
            return solve_dibs_variant(puzzle, instance_id, model_path, variant, alpha, threshold, gpu_id)
    except Exception as e:
        return InstanceResult(
            run_id="", variant=variant, instance_id=instance_id, puzzle=puzzle,
            givens=count_givens(puzzle), status="error", solution=None,
            valid=False, time_ms=0, nodes=0, backtracks=0, error=str(e)
        )


def run_experiment(puzzles: List[str], solver_type: str, model_path: str,
                   variant: str, alpha: float, threshold: int,
                   num_workers: int, run_id: str,
                   gpus: Optional[List[int]] = None) -> List[InstanceResult]:
    print(f"\n{'='*60}")
    print(f"Variant: {variant} (alpha={alpha}, threshold={threshold})")
    print(f"{'='*60}")

    if solver_type == "dibs" and gpus:
        effective_workers = len(gpus)
        print(f"[INFO] Using {effective_workers} GPUs: {gpus}")
    else:
        effective_workers = num_workers

    if gpus and solver_type == "dibs":
        tasks = [(solver_type, puzzle, i, model_path, variant, alpha, threshold, gpus[i % len(gpus)])
                 for i, puzzle in enumerate(puzzles)]
    else:
        tasks = [(solver_type, puzzle, i, model_path, variant, alpha, threshold, None)
                 for i, puzzle in enumerate(puzzles)]

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


def compute_summary(results: List[InstanceResult], variant: str) -> Dict:
    solved = [r for r in results if r.status == "solved"]
    timeout = [r for r in results if r.status == "timeout"]
    total = len(results)
    times = [r.time_ms for r in results] if results else [0]
    nodes = [r.nodes for r in solved] if solved else [0]
    backtracks = [r.backtracks for r in solved] if solved else [0]
    model_calls = [r.model_calls for r in solved] if solved else [0]
    model_times = [r.model_time_ms for r in solved] if solved else [0]

    total_time = sum(times)
    total_model_time = sum(model_times)
    overhead = total_model_time / total_time * 100 if total_time > 0 else 0

    return {
        "variant": variant,
        "total_puzzles": total,
        "solved_count": len(solved),
        "solved_pct": len(solved) / total * 100 if total > 0 else 0,
        "timeout_count": len(timeout),
        "time_ms": {
            "mean": float(np.mean(times)) if times else 0,
            "median": float(np.median(times)) if times else 0,
            "p95": float(np.percentile(times, 95)) if times else 0,
        },
        "nodes": {"mean": float(np.mean(nodes)) if nodes else 0},
        "backtracks": {"mean": float(np.mean(backtracks)) if backtracks else 0},
        "model_calls": {"mean": float(np.mean(model_calls)) if model_calls else 0},
        "model_time_ms": {"mean": float(np.mean(model_times)) if model_times else 0, "total": total_model_time},
        "overhead_pct": overhead,
    }


def save_results(results: List[InstanceResult], output_dir: Path, run_id: str, variant: str) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{run_id}_{variant}.jsonl"
    with open(jsonl_path, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')

    summary = compute_summary(results, variant)

    summary_path = output_dir / f"{run_id}_{variant}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    total = len(results)
    print(f"\n{variant}:")
    print(f"  Solved: {summary['solved_count']}/{total} ({summary['solved_pct']:.2f}%)")
    print(f"  Time: {summary['time_ms']['mean']:.2f}ms, Nodes: {summary['nodes']['mean']:.0f}")
    if summary['model_calls']['mean'] > 0:
        print(f"  Model calls: {summary['model_calls']['mean']:.1f}, Overhead: {summary['overhead_pct']:.1f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Table 3 - Ablation and Hyperparameter")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs for DiBS multi-GPU parallel")
    parser.add_argument("--max-puzzles", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "ablation", "alpha"])
    parser.add_argument("--alpha-values", type=str, default="0.0,0.3,0.5,1.0",
                        help="Comma-separated alpha values for alpha tuning (e.g. 0.2,0.6)")

    args = parser.parse_args()

    if args.workers is None:
        args.workers = min(multiprocessing.cpu_count(), 16)

    gpus = None
    if args.gpus:
        gpus = [int(g.strip()) for g in args.gpus.split(",")]
        print(f"[INFO] GPUs specified for DiBS: {gpus}")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if DATASET["path"].endswith(".csv"):
        dataset_path = MODEL_DATA_DIR / DATASET["path"]
    else:
        dataset_path = DATA_DIR / DATASET["path"]
    puzzles = load_puzzles(str(dataset_path), args.max_puzzles, args.seed)
    print(f"\nLoaded {len(puzzles)} puzzles from {DATASET['description']} (seed={args.seed})")

    meta = {"run_id": run_id, "timestamp": datetime.now().isoformat(),
            "num_puzzles": len(puzzles), "seed": args.seed, "gpus": gpus}
    with open(output_dir / f"{run_id}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    all_summaries = []

    if args.experiment in ["all", "ablation"]:
        print("\n" + "="*80)
        print("ABLATION EXPERIMENTS")
        print("="*80)

        ablation_variants = [
            ("baseline", "Base", 0.8, 2),
            ("dibs", "logits-only", 1.0, 2),
            ("dibs", "DiBS-full", 0.8, 2),
            ("dibs", "MRV>=3", 0.8, 3),
            ("dibs", "always-call", 0.8, 100),
        ]

        for solver_type, variant, alpha, threshold in ablation_variants:
            results = run_experiment(puzzles, solver_type, args.model, variant, alpha, threshold,
                                    args.workers, run_id, gpus)
            summary = save_results(results, output_dir, run_id, variant)
            all_summaries.append(summary)

    if args.experiment in ["all", "alpha"]:
        print("\n" + "="*80)
        print("ALPHA TUNING EXPERIMENTS")
        print("="*80)

        alpha_values = [float(x.strip()) for x in args.alpha_values.split(",") if x.strip() != ""]
        if not alpha_values:
            raise ValueError("alpha-values is empty")

        for alpha in alpha_values:
            variant = f"alpha={alpha}"
            results = run_experiment(puzzles, "dibs", args.model, variant, alpha, 2,
                                    args.workers, run_id, gpus)
            summary = save_results(results, output_dir, run_id, variant)
            all_summaries.append(summary)

    baseline_summary = next((s for s in all_summaries if s['variant'] == 'Base'), None)
    if baseline_summary:
        baseline_time = baseline_summary['time_ms']['mean']
        for s in all_summaries:
            if s['variant'] != 'Base':
                s['speedup'] = baseline_time / s['time_ms']['mean'] if s['time_ms']['mean'] > 0 else 0

    with open(output_dir / f"{run_id}_all_summaries.json", 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "="*80)
    print("TABLE 3 RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Variant':<20} {'Solved%':>8} {'Time':>12} {'Nodes':>10} {'K':>8} {'Overhead':>10} {'Speedup':>8}")
    print("-"*80)
    for s in all_summaries:
        speedup = s.get('speedup', '-')
        speedup_str = f"{speedup:.2f}x" if isinstance(speedup, float) else speedup
        print(f"{s['variant']:<20} {s['solved_pct']:>7.2f}% {s['time_ms']['mean']:>11.1f}ms {s['nodes']['mean']:>10.0f} {s['model_calls']['mean']:>8.1f} {s['overhead_pct']:>9.1f}% {speedup_str:>8}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
