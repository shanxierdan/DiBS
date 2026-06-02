#!/usr/bin/env python3
"""Summarize Table4 results by difficulty buckets and givens-ratio bins."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DIFFICULTY_ORDER = [
    "very_hard",
    "hard",
    "medium",
    "easy",
    "very_easy",
    "unknown",
]


def load_test_difficulty_map(data_root: Path) -> Dict[Tuple[str, str, str], str]:
    mapping = {}
    for fam_dir in data_root.iterdir():
        if not fam_dir.is_dir():
            continue
        family = fam_dir.name
        for size_dir in fam_dir.iterdir():
            if not size_dir.is_dir():
                continue
            size = size_dir.name
            test_path = size_dir / "test.jsonl"
            if not test_path.exists():
                continue
            with test_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    puzzle = row["puzzle"]
                    diff = row.get("difficulty", "unknown")
                    mapping[(family, size, puzzle)] = diff
    return mapping


def load_test_givens_map(data_root: Path) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    mapping = {}
    for fam_dir in data_root.iterdir():
        if not fam_dir.is_dir():
            continue
        family = fam_dir.name
        for size_dir in fam_dir.iterdir():
            if not size_dir.is_dir():
                continue
            size = size_dir.name
            test_path = size_dir / "test.jsonl"
            if not test_path.exists():
                continue
            if family == "generalized_sudoku":
                n = int(size.split("x")[0])
                total = float(n * n)
                empty_symbol = "0"
            else:
                n = int(size)
                total = float(n * n)
                empty_symbol = "."
            with test_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    puzzle = row["puzzle"]
                    givens = row.get("givens")
                    if givens is None:
                        givens = sum(1 for ch in puzzle if ch != empty_symbol)
                    mapping[(family, size, puzzle)] = {
                        "givens": float(givens),
                        "givens_ratio": float(givens) / total if total > 0 else 0.0,
                    }
    return mapping


def parse_result_identity(name: str) -> Optional[Tuple[str, str, str, str]]:
    if name.endswith("_DiBS-full.jsonl"):
        solver = "DiBS-full"
        stem = name[: -len("_DiBS-full.jsonl")]
    elif name.endswith("_MRV_FC_LCV.jsonl"):
        solver = "MRV+FC+LCV"
        stem = name[: -len("_MRV_FC_LCV.jsonl")]
    elif name.endswith("_MRV+FC+LCV.jsonl"):
        solver = "MRV+FC+LCV"
        stem = name[: -len("_MRV+FC+LCV.jsonl")]
    elif name.endswith("_MRV_FC.jsonl"):
        solver = "MRV+FC"
        stem = name[: -len("_MRV_FC.jsonl")]
    elif name.endswith("_MRV+FC.jsonl"):
        solver = "MRV+FC"
        stem = name[: -len("_MRV+FC.jsonl")]
    else:
        return None
    if "_generalized_sudoku_" in stem:
        run_id, rest = stem.split("_generalized_sudoku_", 1)
        family, size = "generalized_sudoku", rest
    elif "_nqueens_" in stem:
        run_id, rest = stem.split("_nqueens_", 1)
        family, size = "nqueens", rest
    else:
        return None
    return run_id, family, size, solver


def ratio_bucket(ratio: float, width: float) -> str:
    width = max(1e-6, min(1.0, width))
    lo = math.floor(ratio / width) * width
    hi = min(1.0, lo + width)
    # keep rightmost closed bin
    if ratio >= 1.0 - 1e-12:
        lo = max(0.0, 1.0 - width)
        hi = 1.0
    return f"[{lo:.2f},{hi:.2f}]"


def summarize_result_jsonl(
    result_path: Path,
    diff_map: Dict[Tuple[str, str, str], str],
    givens_map: Dict[Tuple[str, str, str], Dict[str, float]],
    ratio_bin_width: float,
):
    parsed = parse_result_identity(result_path.name)
    if not parsed:
        return None
    run_id, family, size, solver = parsed
    buckets = defaultdict(
        lambda: {
            "total": 0,
            "solved": 0,
            "timeout": 0,
            "time_sum": 0.0,
            "nodes_sum": 0.0,
            "backtracks_sum": 0.0,
            "model_calls_sum": 0.0,
            "model_time_sum": 0.0,
        }
    )
    ratio_buckets = defaultdict(
        lambda: {
            "total": 0,
            "solved": 0,
            "timeout": 0,
            "time_sum": 0.0,
            "nodes_sum": 0.0,
            "backtracks_sum": 0.0,
            "model_calls_sum": 0.0,
            "model_time_sum": 0.0,
            "givens_ratio_sum": 0.0,
        }
    )
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            puzzle = row["puzzle"]
            diff = diff_map.get((family, size, puzzle), "unknown")
            givens_obj = givens_map.get((family, size, puzzle))
            if givens_obj is None:
                if family == "generalized_sudoku":
                    n = int(size.split("x")[0])
                    ratio = sum(1 for ch in puzzle if ch != "0") / float(n * n)
                else:
                    n = int(size)
                    ratio = puzzle.count("Q") / float(n * n)
            else:
                ratio = float(givens_obj["givens_ratio"])
            rb = ratio_bucket(ratio, ratio_bin_width)

            b = buckets[diff]
            b["total"] += 1
            b["time_sum"] += float(row.get("time_ms", 0.0))
            b["nodes_sum"] += float(row.get("nodes", 0.0))
            b["backtracks_sum"] += float(row.get("backtracks", 0.0))
            b["model_calls_sum"] += float(row.get("model_calls", 0.0))
            b["model_time_sum"] += float(row.get("model_time_ms", 0.0))
            st = row.get("status")
            if st == "solved":
                b["solved"] += 1
            elif st == "timeout":
                b["timeout"] += 1

            r = ratio_buckets[rb]
            r["total"] += 1
            r["time_sum"] += float(row.get("time_ms", 0.0))
            r["nodes_sum"] += float(row.get("nodes", 0.0))
            r["backtracks_sum"] += float(row.get("backtracks", 0.0))
            r["model_calls_sum"] += float(row.get("model_calls", 0.0))
            r["model_time_sum"] += float(row.get("model_time_ms", 0.0))
            r["givens_ratio_sum"] += ratio
            if st == "solved":
                r["solved"] += 1
            elif st == "timeout":
                r["timeout"] += 1

    out = []
    for diff in sorted(
        buckets.keys(),
        key=lambda x: DIFFICULTY_ORDER.index(x) if x in DIFFICULTY_ORDER else len(DIFFICULTY_ORDER),
    ):
        b = buckets.get(diff)
        if not b or b["total"] == 0:
            continue
        out.append(
            {
                "difficulty": diff,
                "total": b["total"],
                "solved": b["solved"],
                "solved_pct": b["solved"] / b["total"] * 100.0,
                "timeout": b["timeout"],
                "time_mean_ms": b["time_sum"] / b["total"],
                "nodes_mean": b["nodes_sum"] / b["total"],
                "backtracks_mean": b["backtracks_sum"] / b["total"],
                "model_calls_mean": b["model_calls_sum"] / b["total"],
                "model_time_mean_ms": b["model_time_sum"] / b["total"],
            }
        )
    ratio_out = []
    for diff in sorted(
        ratio_buckets.keys(),
        key=lambda x: float(x.split(",")[0].strip("[")),
    ):
        b = ratio_buckets[diff]
        ratio_out.append(
            {
                "givens_ratio_bucket": diff,
                "total": b["total"],
                "solved": b["solved"],
                "solved_pct": b["solved"] / b["total"] * 100.0,
                "timeout": b["timeout"],
                "time_mean_ms": b["time_sum"] / b["total"],
                "nodes_mean": b["nodes_sum"] / b["total"],
                "backtracks_mean": b["backtracks_sum"] / b["total"],
                "model_calls_mean": b["model_calls_sum"] / b["total"],
                "model_time_mean_ms": b["model_time_sum"] / b["total"],
                "givens_ratio_mean": b["givens_ratio_sum"] / b["total"],
            }
        )
    return {
        "run_id": run_id,
        "task_family": family,
        "size": size,
        "solver": solver,
        "difficulty_buckets": out,
        "givens_ratio_buckets": ratio_out,
    }


def _pct_delta(base: float, target: float) -> Optional[float]:
    if abs(base) < 1e-12:
        return None
    return (target - base) / base * 100.0


def build_dibs_gain_by_ratio(rows: List[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[str, str], Dict[str, Dict]] = defaultdict(dict)
    for row in rows:
        by_key[(row["task_family"], row["size"])][row["solver"]] = row

    out = []
    for (family, size), obj in sorted(by_key.items()):
        dibs = obj.get("DiBS-full")
        mrv = obj.get("MRV+FC+LCV") or obj.get("MRV+FC")
        if not dibs or not mrv:
            continue

        mrv_bins = {b["givens_ratio_bucket"]: b for b in mrv.get("givens_ratio_buckets", [])}
        dibs_bins = {b["givens_ratio_bucket"]: b for b in dibs.get("givens_ratio_buckets", [])}
        common_bins = sorted(
            set(mrv_bins.keys()) & set(dibs_bins.keys()),
            key=lambda x: float(x.split(",")[0].strip("[")),
        )
        for bucket in common_bins:
            m = mrv_bins[bucket]
            d = dibs_bins[bucket]
            out.append(
                {
                    "task_family": family,
                    "size": size,
                    "givens_ratio_bucket": bucket,
                    "givens_ratio_mean": d.get("givens_ratio_mean", m.get("givens_ratio_mean", 0.0)),
                    "count": min(int(m["total"]), int(d["total"])),
                    "mrv_solved_pct": float(m["solved_pct"]),
                    "dibs_solved_pct": float(d["solved_pct"]),
                    "delta_solved_pct_points": float(d["solved_pct"]) - float(m["solved_pct"]),
                    "mrv_time_mean_ms": float(m["time_mean_ms"]),
                    "dibs_time_mean_ms": float(d["time_mean_ms"]),
                    "dibs_time_delta_pct_vs_mrv": _pct_delta(float(m["time_mean_ms"]), float(d["time_mean_ms"])),
                    "mrv_nodes_mean": float(m["nodes_mean"]),
                    "dibs_nodes_mean": float(d["nodes_mean"]),
                    "dibs_nodes_delta_pct_vs_mrv": _pct_delta(float(m["nodes_mean"]), float(d["nodes_mean"])),
                    "mrv_backtracks_mean": float(m["backtracks_mean"]),
                    "dibs_backtracks_mean": float(d["backtracks_mean"]),
                    "dibs_backtracks_delta_pct_vs_mrv": _pct_delta(float(m["backtracks_mean"]), float(d["backtracks_mean"])),
                    "dibs_model_calls_mean": float(d["model_calls_mean"]),
                    "dibs_model_time_mean_ms": float(d["model_time_mean_ms"]),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "dataset" / "table4_extension"))
    parser.add_argument("--results-root", type=str, default=str(PROJECT_ROOT / "DiBS" / "results" / "parallel" / "Table_4"))
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--ratio-bin-width", type=float, default=0.1)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    run_id = args.run_id
    diff_map = load_test_difficulty_map(data_root)
    givens_map = load_test_givens_map(data_root)

    result_files = sorted(results_root.glob(f"{run_id}_*.jsonl"))
    rows = []
    for p in result_files:
        if p.name.endswith("_all_summaries.jsonl"):
            continue
        parsed = summarize_result_jsonl(
            p,
            diff_map=diff_map,
            givens_map=givens_map,
            ratio_bin_width=args.ratio_bin_width,
        )
        if parsed:
            rows.append(parsed)
    gains = build_dibs_gain_by_ratio(rows)

    out_json = results_root / f"{run_id}_difficulty_report.json"
    out_md = results_root / f"{run_id}_difficulty_report.md"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    out_gain_json = results_root / f"{run_id}_givens_ratio_gain_report.json"
    out_gain_md = results_root / f"{run_id}_givens_ratio_gain_report.md"
    out_gain_json.write_text(json.dumps(gains, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append(f"# Difficulty Report: {run_id}")
    lines.append("")
    lines.append("| Task | Solver | Difficulty | Solved | Solved% | Timeout | Time mean (ms) | Nodes mean | Backtracks mean | Model calls mean | Model time mean (ms) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        task = f"{r['task_family']}:{r['size']}"
        for b in r["difficulty_buckets"]:
            lines.append(
                f"| {task} | {r['solver']} | {b['difficulty']} | "
                f"{b['solved']}/{b['total']} | {b['solved_pct']:.2f} | {b['timeout']} | {b['time_mean_ms']:.2f} | "
                f"{b['nodes_mean']:.2f} | {b['backtracks_mean']:.2f} | {b['model_calls_mean']:.2f} | {b['model_time_mean_ms']:.2f} |"
            )
    lines.append("")
    lines.append("## By Givens Ratio")
    lines.append("")
    lines.append("| Task | Solver | Givens ratio bucket | Mean ratio | Solved | Solved% | Time mean (ms) | Nodes mean | Backtracks mean |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        task = f"{r['task_family']}:{r['size']}"
        for b in r.get("givens_ratio_buckets", []):
            lines.append(
                f"| {task} | {r['solver']} | {b['givens_ratio_bucket']} | {b['givens_ratio_mean']:.3f} | "
                f"{b['solved']}/{b['total']} | {b['solved_pct']:.2f} | {b['time_mean_ms']:.2f} | "
                f"{b['nodes_mean']:.2f} | {b['backtracks_mean']:.2f} |"
            )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    gain_lines = []
    gain_lines.append(f"# Givens-Ratio DiBS Gain Report: {run_id}")
    gain_lines.append("")
    gain_lines.append("| Task | Ratio bucket | Mean ratio | Count | Solved pp (DiBS-MRV) | Time delta % (DiBS vs MRV) | Nodes delta % | Backtracks delta % | DiBS model calls | DiBS model time (ms) |")
    gain_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in gains:
        t = f"{g['task_family']}:{g['size']}"
        td = "NA" if g["dibs_time_delta_pct_vs_mrv"] is None else f"{g['dibs_time_delta_pct_vs_mrv']:.2f}"
        nd = "NA" if g["dibs_nodes_delta_pct_vs_mrv"] is None else f"{g['dibs_nodes_delta_pct_vs_mrv']:.2f}"
        bd = "NA" if g["dibs_backtracks_delta_pct_vs_mrv"] is None else f"{g['dibs_backtracks_delta_pct_vs_mrv']:.2f}"
        gain_lines.append(
            f"| {t} | {g['givens_ratio_bucket']} | {g['givens_ratio_mean']:.3f} | {g['count']} | "
            f"{g['delta_solved_pct_points']:.2f} | {td} | {nd} | {bd} | "
            f"{g['dibs_model_calls_mean']:.2f} | {g['dibs_model_time_mean_ms']:.2f} |"
        )
    out_gain_md.write_text("\n".join(gain_lines), encoding="utf-8")

    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"saved: {out_gain_json}")
    print(f"saved: {out_gain_md}")


if __name__ == "__main__":
    main()
