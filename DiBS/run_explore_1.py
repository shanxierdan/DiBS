#!/usr/bin/env python3
"""Explore 1: DiBS gain vs Sudoku difficulty (givens buckets).

Build a merged+deduped pool from dataset/prepared_data, auto-select feasible
hard buckets by availability, then compare MRV+FC+LCV vs DiBS.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DiBS.explore_common import (
    DEFAULT_MODEL_PATH,
    build_merged_prepared_dataset,
    choose_hard_buckets_from_counts,
    load_jsonl,
    list_prepared_puzzle_files,
    now_ts,
    pearson_corr,
    sample_buckets_from_merged_jsonl,
    solve_instance_task,
    spearman_corr,
    summarize_records,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "explore_runs"


def existing_instance_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for row in load_jsonl(path):
        if "instance_id" in row:
            ids.add(int(row["instance_id"]))
    return ids


def run_solver_on_bucket(
    run_id: str,
    solver_name: str,
    bucket_givens: int,
    bucket_rows: List[Dict],
    output_jsonl: Path,
    model_path: str,
    timeout_ms: float,
    max_nodes: int,
    workers: int,
    gpu: int,
    denoise_steps: int,
    resume: bool,
) -> List[Dict]:
    done_ids = existing_instance_ids(output_jsonl) if resume else set()
    pending = [r for r in bucket_rows if int(r["instance_id"]) not in done_ids]
    existing_rows = load_jsonl(output_jsonl) if (resume and output_jsonl.exists()) else []
    results: List[Dict] = list(existing_rows)

    if not pending:
        print(f"[{solver_name}][givens={bucket_givens}] resume-skip: {len(results)}/{len(bucket_rows)}", flush=True)
        return results

    tasks = []
    for row in pending:
        tasks.append(
            {
                "run_id": run_id,
                "solver": solver_name,
                "puzzle": row["puzzle"],
                "instance_id": int(row["instance_id"]),
                "bucket_givens": bucket_givens,
                "model_path": model_path,
                "timeout_ms": timeout_ms,
                "max_nodes": max_nodes,
                "gpu": (gpu if solver_name == "DiBS" else None),
                "denoise_steps": denoise_steps,
            }
        )

    mode = "a" if (resume and output_jsonl.exists()) else "w"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[{solver_name}][givens={bucket_givens}] start total={len(bucket_rows)} pending={len(tasks)} workers={workers}",
        flush=True,
    )
    with output_jsonl.open(mode, encoding="utf-8") as f:
        if workers <= 1:
            for idx, task in enumerate(tasks, start=1):
                out = solve_instance_task(task)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                results.append(out)
                if idx % 20 == 0 or idx == len(tasks):
                    print(
                        f"[{solver_name}][givens={bucket_givens}] {len(results)}/{len(bucket_rows)} done",
                        flush=True,
                    )
        else:
            mp_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
                futures = [ex.submit(solve_instance_task, t) for t in tasks]
                finished = 0
                for fut in as_completed(futures):
                    out = fut.result()
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    results.append(out)
                    finished += 1
                    if finished % 20 == 0 or finished == len(tasks):
                        print(
                            f"[{solver_name}][givens={bucket_givens}] {len(results)}/{len(bucket_rows)} done",
                            flush=True,
                        )

    results.sort(key=lambda x: int(x["instance_id"]))
    return results


def build_bucket_table(baseline_summary: Dict[int, Dict], dibs_summary: Dict[int, Dict]) -> List[Dict]:
    rows = []
    for g in sorted(baseline_summary.keys()):
        b = baseline_summary[g]
        d = dibs_summary[g]
        b_time = float(b["time_ms"]["all"]["mean"])
        d_time = float(d["time_ms"]["all"]["mean"])
        b_nodes = float(b["nodes"]["all"]["mean"])
        d_nodes = float(d["nodes"]["all"]["mean"])
        b_back = float(b["backtracks"]["all"]["mean"])
        d_back = float(d["backtracks"]["all"]["mean"])
        b_solved = float(b["solved_pct"])
        d_solved = float(d["solved_pct"])
        rows.append(
            {
                "givens": g,
                "hardness": -float(g),
                "baseline_solved_pct": b_solved,
                "dibs_solved_pct": d_solved,
                "solved_gain_pp": d_solved - b_solved,
                "baseline_time_mean": b_time,
                "dibs_time_mean": d_time,
                "time_gain_pct": ((b_time - d_time) / b_time * 100.0) if b_time > 0 else 0.0,
                "baseline_nodes_mean": b_nodes,
                "dibs_nodes_mean": d_nodes,
                "nodes_gain_pct": ((b_nodes - d_nodes) / b_nodes * 100.0) if b_nodes > 0 else 0.0,
                "baseline_backtracks_mean": b_back,
                "dibs_backtracks_mean": d_back,
                "backtracks_gain_pct": ((b_back - d_back) / b_back * 100.0) if b_back > 0 else 0.0,
            }
        )
    return rows


def write_markdown_report(path: Path, payload: Dict) -> None:
    rows = payload["bucket_comparison"]
    corr = payload["correlations"]
    lines: List[str] = []
    lines.append(f"# Explore 1 Report ({payload['run_id']})")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Buckets: `{payload['givens_values']}`")
    lines.append(f"- Per bucket: `{payload['per_bucket']}`")
    lines.append(f"- Solvers: `MRV+FC+LCV` vs `DiBS`")
    lines.append("")
    lines.append("## Bucket Comparison")
    lines.append("")
    lines.append("| givens | solved gain (pp) | time gain (%) | nodes gain (%) | backtracks gain (%) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['givens']} | {r['solved_gain_pp']:.2f} | {r['time_gain_pct']:.2f} | "
            f"{r['nodes_gain_pct']:.2f} | {r['backtracks_gain_pct']:.2f} |"
        )
    lines.append("")
    lines.append("## Correlation (hardness = -givens)")
    lines.append("")
    lines.append(f"- solved_gain_pp: pearson={corr['solved_gain_pp']['pearson']:.4f}, spearman={corr['solved_gain_pp']['spearman']:.4f}")
    lines.append(f"- time_gain_pct: pearson={corr['time_gain_pct']['pearson']:.4f}, spearman={corr['time_gain_pct']['spearman']:.4f}")
    lines.append(f"- nodes_gain_pct: pearson={corr['nodes_gain_pct']['pearson']:.4f}, spearman={corr['nodes_gain_pct']['spearman']:.4f}")
    lines.append(
        f"- backtracks_gain_pct: pearson={corr['backtracks_gain_pct']['pearson']:.4f}, "
        f"spearman={corr['backtracks_gain_pct']['spearman']:.4f}"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore 1: difficulty vs DiBS gain.")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--prepared-dir", type=str, default=str(PROJECT_ROOT / "dataset" / "prepared_data"))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--givens-values", type=str, default="", help="comma list, e.g. 17,18,19; empty=auto")
    parser.add_argument("--bucket-count", type=int, default=10, help="used only when givens-values is empty")
    parser.add_argument("--min-givens", type=int, default=17, help="used only when givens-values is empty")
    parser.add_argument("--per-bucket", type=int, default=500)
    parser.add_argument("--timeout-ms", type=float, default=0.0, help="0 means no wall-time timeout")
    parser.add_argument("--max-nodes", type=int, default=1000000)
    parser.add_argument("--workers-baseline", type=int, default=32)
    parser.add_argument("--workers-dibs", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id.strip() or now_ts()
    run_dir = Path(args.output_root) / run_id / "explore_1"
    data_dir = run_dir / "data"
    result_dir = run_dir / "results"
    report_dir = run_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    givens_values: List[int] = []
    sampled_path = data_dir / "sampled_puzzles.jsonl"
    sample_meta_path = data_dir / "sample_meta.json"
    merged_jsonl = data_dir / "prepared_merged.jsonl"
    merged_stats_json = data_dir / "prepared_merged_stats.json"

    if args.resume and sampled_path.exists():
        sampled_rows = load_jsonl(sampled_path)
        sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8")) if sample_meta_path.exists() else {}
        givens_values = sorted({int(r["bucket_givens"]) for r in sampled_rows})
    else:
        merged_stats = build_merged_prepared_dataset(
            output_jsonl=merged_jsonl,
            stats_json=merged_stats_json,
            source_files=list_prepared_puzzle_files(Path(args.prepared_dir)),
            resume_if_exists=True,
        )
        counts_by_givens = {int(k): int(v) for k, v in merged_stats.get("counts_by_givens", {}).items()}
        if args.givens_values.strip():
            givens_values = sorted({int(x.strip()) for x in args.givens_values.split(",") if x.strip()})
            missing = {g: args.per_bucket - counts_by_givens.get(g, 0) for g in givens_values if counts_by_givens.get(g, 0) < args.per_bucket}
            if missing:
                raise RuntimeError(f"Requested givens-values insufficient in merged pool: {missing}")
        else:
            givens_values = choose_hard_buckets_from_counts(
                counts_by_givens=counts_by_givens,
                per_bucket=args.per_bucket,
                bucket_count=args.bucket_count,
                min_givens=args.min_givens,
            )
        sampled_rows, sample_meta = sample_buckets_from_merged_jsonl(
            merged_jsonl=merged_jsonl,
            givens_values=givens_values,
            per_bucket=args.per_bucket,
            seed=args.seed,
        )
        sample_meta["merged_stats"] = merged_stats
        sample_meta["selected_givens_values"] = givens_values
        write_jsonl(sampled_path, sampled_rows)
        sample_meta_path.write_text(json.dumps(sample_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    by_bucket: Dict[int, List[Dict]] = {g: [] for g in givens_values}
    for row in sampled_rows:
        by_bucket[int(row["bucket_givens"])].append(row)

    run_meta = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "explore_1",
        "model": args.model,
        "seed": args.seed,
        "givens_values": givens_values,
        "per_bucket": args.per_bucket,
        "timeout_ms": args.timeout_ms,
        "max_nodes": args.max_nodes,
        "workers_baseline": args.workers_baseline,
        "workers_dibs": args.workers_dibs,
        "gpu": args.gpu,
        "resume": bool(args.resume),
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    baseline_summary: Dict[int, Dict] = {}
    dibs_summary: Dict[int, Dict] = {}

    for g in givens_values:
        bucket_rows = sorted(by_bucket[g], key=lambda x: int(x["instance_id"]))

        baseline_path = result_dir / "per_instance" / "MRV_FC_LCV" / f"givens_{g}.jsonl"
        baseline_rows = run_solver_on_bucket(
            run_id=run_id,
            solver_name="MRV+FC+LCV",
            bucket_givens=g,
            bucket_rows=bucket_rows,
            output_jsonl=baseline_path,
            model_path=args.model,
            timeout_ms=args.timeout_ms,
            max_nodes=args.max_nodes,
            workers=max(1, int(args.workers_baseline)),
            gpu=args.gpu,
            denoise_steps=1,
            resume=args.resume,
        )
        baseline_summary[g] = summarize_records(baseline_rows)
        (result_dir / "summaries" / "MRV_FC_LCV").mkdir(parents=True, exist_ok=True)
        (result_dir / "summaries" / "MRV_FC_LCV" / f"givens_{g}_summary.json").write_text(
            json.dumps(baseline_summary[g], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        dibs_path = result_dir / "per_instance" / "DiBS" / f"givens_{g}.jsonl"
        dibs_rows = run_solver_on_bucket(
            run_id=run_id,
            solver_name="DiBS",
            bucket_givens=g,
            bucket_rows=bucket_rows,
            output_jsonl=dibs_path,
            model_path=args.model,
            timeout_ms=args.timeout_ms,
            max_nodes=args.max_nodes,
            workers=max(1, int(args.workers_dibs)),
            gpu=args.gpu,
            denoise_steps=1,
            resume=args.resume,
        )
        dibs_summary[g] = summarize_records(dibs_rows)
        (result_dir / "summaries" / "DiBS").mkdir(parents=True, exist_ok=True)
        (result_dir / "summaries" / "DiBS" / f"givens_{g}_summary.json").write_text(
            json.dumps(dibs_summary[g], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    bucket_table = build_bucket_table(baseline_summary, dibs_summary)
    x = [float(r["hardness"]) for r in bucket_table]
    corr = {}
    for metric in ("solved_gain_pp", "time_gain_pct", "nodes_gain_pct", "backtracks_gain_pct"):
        y = [float(r[metric]) for r in bucket_table]
        corr[metric] = {
            "pearson": pearson_corr(x, y),
            "spearman": spearman_corr(x, y),
        }

    global_payload = {
        "run_id": run_id,
        "mode": "explore_1",
        "givens_values": givens_values,
        "per_bucket": args.per_bucket,
        "sample_meta": sample_meta,
        "baseline_summary_by_bucket": baseline_summary,
        "dibs_summary_by_bucket": dibs_summary,
        "bucket_comparison": bucket_table,
        "correlations": corr,
    }
    (result_dir / "explore_1_global_summary.json").write_text(
        json.dumps(global_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_report(report_dir / "explore_1_report.md", global_payload)

    print(f"Explore 1 done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
