from __future__ import annotations

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from DiBS.config import DiBSConfig
from DiBS.solver import BaselineSolver, DiBSSolver


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPARED_DATA_DIR = PROJECT_ROOT / "dataset" / "prepared_data"
MODEL_DATA_DIR = PROJECT_ROOT / "model" / "diffusion-vs-ar" / "data"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "diffusion-vs-ar"
    / "output"
    / "sudoku"
    / "royle17-20260323-210104"
)


SOURCE_PRIORITY: List[Tuple[str, Path]] = [
    ("royle_17clue", PROJECT_ROOT / "dataset" / "prepared_data" / "royle_17clue.txt"),
    ("royle_forum_hardest_1905", PROJECT_ROOT / "dataset" / "prepared_data" / "royle_forum_hardest_1905.txt"),
    ("royle_forum_hardest_11plus", PROJECT_ROOT / "dataset" / "prepared_data" / "royle_forum_hardest_11plus.txt"),
    ("royle_magictour_top1465", PROJECT_ROOT / "dataset" / "prepared_data" / "royle_magictour_top1465.txt"),
    ("royle_unbiased", PROJECT_ROOT / "dataset" / "prepared_data" / "royle_unbiased.txt"),
]


def list_prepared_puzzle_files(prepared_dir: Optional[Path] = None) -> List[Path]:
    base = prepared_dir or PREPARED_DATA_DIR
    files = []
    for p in sorted(base.glob("*.txt")):
        name = p.name.lower()
        if "solution" in name:
            continue
        files.append(p)
    return files


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def count_givens(puzzle: str) -> int:
    return sum(1 for c in puzzle if c not in ("0", "."))


def iter_puzzles(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if len(s) < 81:
                continue
            p = s[:81].replace(".", "0")
            if len(p) == 81:
                yield p


def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _puzzle_fingerprint_u64(puzzle: str) -> int:
    digest = hashlib.blake2b(puzzle.encode("ascii", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def build_merged_prepared_dataset(
    output_jsonl: Path,
    stats_json: Path,
    source_files: Optional[Sequence[Path]] = None,
    resume_if_exists: bool = True,
) -> Dict:
    if resume_if_exists and output_jsonl.exists() and stats_json.exists():
        return json.loads(stats_json.read_text(encoding="utf-8"))

    source_files = list(source_files) if source_files is not None else list_prepared_puzzle_files(PREPARED_DATA_DIR)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen_fp = set()
    counts_by_givens: Dict[int, int] = {}
    counts_by_source: Dict[str, int] = {}
    total_raw = 0
    total_unique = 0

    with output_jsonl.open("w", encoding="utf-8") as out:
        for src in source_files:
            src_name = src.stem
            counts_by_source.setdefault(src_name, 0)
            for puzzle in iter_puzzles(src):
                total_raw += 1
                fp = _puzzle_fingerprint_u64(puzzle)
                if fp in seen_fp:
                    continue
                seen_fp.add(fp)
                g = count_givens(puzzle)
                row = {"puzzle": puzzle, "givens": g, "source": src_name}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_unique += 1
                counts_by_source[src_name] += 1
                counts_by_givens[g] = counts_by_givens.get(g, 0) + 1

    stats = {
        "total_raw": total_raw,
        "total_unique": total_unique,
        "sources": [str(p) for p in source_files],
        "counts_by_source": counts_by_source,
        "counts_by_givens": {str(k): int(v) for k, v in sorted(counts_by_givens.items())},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def choose_hard_buckets_from_counts(
    counts_by_givens: Dict[int, int],
    per_bucket: int,
    bucket_count: int = 10,
    min_givens: int = 17,
) -> List[int]:
    candidates = [g for g, c in counts_by_givens.items() if g >= min_givens and c >= per_bucket]
    candidates.sort()
    if len(candidates) < bucket_count:
        raise RuntimeError(
            f"Not enough feasible givens buckets: need {bucket_count}, got {len(candidates)} "
            f"(per_bucket={per_bucket}, min_givens={min_givens})."
        )
    return candidates[:bucket_count]


def sample_buckets_from_merged_jsonl(
    merged_jsonl: Path,
    givens_values: Sequence[int],
    per_bucket: int,
    seed: int,
) -> Tuple[List[Dict], Dict]:
    rng = np.random.default_rng(seed)
    target_set = set(int(g) for g in givens_values)
    reservoirs: Dict[int, List[Dict]] = {int(g): [] for g in givens_values}
    seen_per_bucket: Dict[int, int] = {int(g): 0 for g in givens_values}

    with merged_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            g = int(row["givens"])
            if g not in target_set:
                continue
            seen_per_bucket[g] += 1
            rec = {"puzzle": row["puzzle"], "givens": g, "source": row.get("source", "merged")}
            arr = reservoirs[g]
            if len(arr) < per_bucket:
                arr.append(rec)
            else:
                j = int(rng.integers(0, seen_per_bucket[g]))
                if j < per_bucket:
                    arr[j] = rec

    missing = {g: per_bucket - len(reservoirs[g]) for g in givens_values if len(reservoirs[g]) < per_bucket}
    if missing:
        raise RuntimeError(f"Insufficient samples in merged dataset for buckets: {missing}")

    rows: List[Dict] = []
    instance_id = 0
    for g in sorted(givens_values):
        bucket = reservoirs[g]
        rng.shuffle(bucket)
        for i, rec in enumerate(bucket):
            rows.append(
                {
                    "instance_id": instance_id,
                    "bucket_givens": g,
                    "bucket_index": i,
                    "puzzle": rec["puzzle"],
                    "source": rec["source"],
                }
            )
            instance_id += 1

    meta = {
        "givens_values": list(sorted(int(g) for g in givens_values)),
        "per_bucket": per_bucket,
        "total": len(rows),
        "seen_per_bucket": {str(k): int(v) for k, v in sorted(seen_per_bucket.items())},
    }
    return rows, meta


def load_table3_style_puzzles(dataset_csv: Path, max_puzzles: int, seed: int) -> List[str]:
    import csv

    puzzles: List[str] = []
    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = str(row.get("quizzes", "")).strip().replace(".", "0")
            if len(p) >= 81:
                puzzles.append(p[:81])
    if len(puzzles) > max_puzzles:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(puzzles), size=max_puzzles, replace=False)
        puzzles = [puzzles[int(i)] for i in idx]
    return puzzles


def sample_buckets_from_sources(
    givens_values: Sequence[int],
    per_bucket: int,
    seed: int,
    source_priority: Optional[List[Tuple[str, Path]]] = None,
) -> Tuple[List[Dict], Dict]:
    rng = np.random.default_rng(seed)
    source_priority = source_priority or SOURCE_PRIORITY
    buckets = {g: [] for g in givens_values}
    seen = set()
    source_counts = {name: 0 for name, _ in source_priority}

    target_total = len(givens_values) * per_bucket

    for source_name, source_path in source_priority:
        if not source_path.exists():
            continue
        for puzzle in iter_puzzles(source_path):
            g = count_givens(puzzle)
            if g not in buckets:
                continue
            if len(buckets[g]) >= per_bucket:
                continue
            if puzzle in seen:
                continue
            seen.add(puzzle)
            buckets[g].append({"puzzle": puzzle, "givens": g, "source": source_name})
            source_counts[source_name] += 1
            if sum(len(v) for v in buckets.values()) >= target_total:
                break
        if sum(len(v) for v in buckets.values()) >= target_total:
            break

    missing = {g: per_bucket - len(buckets[g]) for g in givens_values if len(buckets[g]) < per_bucket}
    if missing:
        raise RuntimeError(f"Insufficient samples for buckets: {missing}")

    rows: List[Dict] = []
    instance_id = 0
    for g in sorted(givens_values):
        arr = buckets[g]
        rng.shuffle(arr)
        arr = arr[:per_bucket]
        for i, rec in enumerate(arr):
            rows.append(
                {
                    "instance_id": instance_id,
                    "bucket_givens": g,
                    "bucket_index": i,
                    "puzzle": rec["puzzle"],
                    "source": rec["source"],
                }
            )
            instance_id += 1

    return rows, {
        "givens_values": list(sorted(givens_values)),
        "per_bucket": per_bucket,
        "total": len(rows),
        "source_counts": source_counts,
    }


def _mean_median_p95(values: List[float]) -> Dict[str, float]:
    if not values:
        values = [0.0]
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def summarize_records(rows: Sequence[Dict]) -> Dict:
    total = len(rows)
    solved_rows = [r for r in rows if r.get("status") == "solved"]
    time_all = [float(r.get("time_ms", 0.0)) for r in rows]
    nodes_all = [float(r.get("nodes", 0.0)) for r in rows]
    back_all = [float(r.get("backtracks", 0.0)) for r in rows]
    model_calls_all = [float(r.get("model_calls", 0.0)) for r in rows]
    model_time_all = [float(r.get("model_time_ms", 0.0)) for r in rows]

    time_solved = [float(r.get("time_ms", 0.0)) for r in solved_rows]
    nodes_solved = [float(r.get("nodes", 0.0)) for r in solved_rows]
    back_solved = [float(r.get("backtracks", 0.0)) for r in solved_rows]
    model_calls_solved = [float(r.get("model_calls", 0.0)) for r in solved_rows]
    model_time_solved = [float(r.get("model_time_ms", 0.0)) for r in solved_rows]

    return {
        "total": total,
        "solved": len(solved_rows),
        "solved_pct": (100.0 * len(solved_rows) / total) if total else 0.0,
        "timeout": sum(1 for r in rows if r.get("status") == "timeout"),
        "time_ms": {
            "all": _mean_median_p95(time_all),
            "solved_only": _mean_median_p95(time_solved),
        },
        "nodes": {
            "all": _mean_median_p95(nodes_all),
            "solved_only": _mean_median_p95(nodes_solved),
        },
        "backtracks": {
            "all": _mean_median_p95(back_all),
            "solved_only": _mean_median_p95(back_solved),
        },
        "model_calls": {
            "all": _mean_median_p95(model_calls_all),
            "solved_only": _mean_median_p95(model_calls_solved),
        },
        "model_time_ms": {
            "all": _mean_median_p95(model_time_all),
            "solved_only": _mean_median_p95(model_time_solved),
        },
    }


def _rankdata(vals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(arr, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if len(xa) < 2:
        return 0.0
    if np.std(xa) == 0 or np.std(ya) == 0:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    xr = _rankdata(x)
    yr = _rankdata(y)
    return pearson_corr(xr, yr)


_WORKER_SOLVER_CACHE: Dict[Tuple, object] = {}


def _get_solver(
    solver_name: str,
    model_path: str,
    timeout_ms: float,
    max_nodes: int,
    gpu: Optional[int],
    denoise_steps: int,
    denoise_strategy: str,
    mdm_decoding_strategy: str,
):
    key = (
        solver_name,
        model_path,
        timeout_ms,
        max_nodes,
        gpu if gpu is not None else -1,
        denoise_steps,
        denoise_strategy,
        mdm_decoding_strategy,
    )
    if key in _WORKER_SOLVER_CACHE:
        return _WORKER_SOLVER_CACHE[key]

    if solver_name == "MRV+FC+LCV":
        solver = BaselineSolver(use_lcv=True, use_fc=True, max_nodes=max_nodes, timeout_ms=timeout_ms)
    elif solver_name == "DiBS":
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        if use_cuda and gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cfg = DiBSConfig(
            device=device,
            max_nodes=max_nodes,
            denoise_strategy=denoise_strategy,
            mdm_decoding_strategy=mdm_decoding_strategy,
        )
        solver = DiBSSolver(
            model_path=model_path,
            config=cfg,
            use_heuristic=True,
            use_lcv=False,
            use_fc=True,
            timeout_ms=timeout_ms if timeout_ms > 0 else 10**12,
            denoise_steps=denoise_steps,
        )
    else:
        raise ValueError(f"Unsupported solver: {solver_name}")

    _WORKER_SOLVER_CACHE[key] = solver
    return solver


def solve_instance_task(task: Dict) -> Dict:
    solver_name = str(task["solver"])
    puzzle = str(task["puzzle"])
    instance_id = int(task["instance_id"])
    bucket_givens = int(task["bucket_givens"])
    run_id = str(task["run_id"])
    model_path = str(task["model_path"])
    timeout_ms = float(task.get("timeout_ms", 0.0))
    max_nodes = int(task.get("max_nodes", 1000000))
    gpu = task.get("gpu")
    denoise_steps = int(task.get("denoise_steps", 1))
    denoise_strategy = str(task.get("denoise_strategy", "legacy_repeat"))
    mdm_decoding_strategy = str(task.get("mdm_decoding_strategy", "deterministic-cosine"))

    solver = _get_solver(
        solver_name=solver_name,
        model_path=model_path,
        timeout_ms=(timeout_ms if timeout_ms > 0 else 10**12),
        max_nodes=max_nodes,
        gpu=gpu if gpu is not None else None,
        denoise_steps=denoise_steps,
        denoise_strategy=denoise_strategy,
        mdm_decoding_strategy=mdm_decoding_strategy,
    )

    t0 = time.perf_counter()
    try:
        solution, metrics = solver.solve(puzzle)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if bool(metrics.solved):
            status = "solved"
        elif timeout_ms > 0 and elapsed_ms >= timeout_ms:
            status = "timeout"
        else:
            status = "failed"
        return {
            "run_id": run_id,
            "instance_id": instance_id,
            "bucket_givens": bucket_givens,
            "solver": solver_name,
            "denoise_steps": denoise_steps,
            "denoise_strategy": denoise_strategy,
            "mdm_decoding_strategy": mdm_decoding_strategy,
            "status": status,
            "valid": bool(metrics.is_valid),
            "time_ms": float(elapsed_ms),
            "nodes": int(metrics.expanded_nodes),
            "backtracks": int(metrics.backtracks),
            "propagation_steps": int(metrics.propagation_steps),
            "model_calls": int(metrics.model_calls),
            "model_time_ms": float(metrics.model_time_ms),
            "solution": solution,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "run_id": run_id,
            "instance_id": instance_id,
            "bucket_givens": bucket_givens,
            "solver": solver_name,
            "denoise_steps": denoise_steps,
            "denoise_strategy": denoise_strategy,
            "mdm_decoding_strategy": mdm_decoding_strategy,
            "status": "error",
            "valid": False,
            "time_ms": float(elapsed_ms),
            "nodes": 0,
            "backtracks": 0,
            "propagation_steps": 0,
            "model_calls": 0,
            "model_time_ms": 0.0,
            "solution": None,
            "error": str(exc),
        }
