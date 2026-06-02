#!/usr/bin/env python3
"""Explore 2: multi-step denoising usage study for DiBS only."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DiBS.explore_common import (
    DEFAULT_MODEL_PATH,
    MODEL_DATA_DIR,
    count_givens,
    load_table3_style_puzzles,
    load_jsonl,
    now_ts,
    solve_instance_task,
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


def run_solver_all(
    run_id: str,
    rows: List[Dict],
    output_jsonl: Path,
    model_path: str,
    timeout_ms: float,
    max_nodes: int,
    workers: int,
    gpus: List[int],
    gpu: int,
    denoise_steps: int,
    denoise_strategy: str,
    mdm_decoding_strategy: str,
    resume: bool,
) -> List[Dict]:
    done_ids = existing_instance_ids(output_jsonl) if resume else set()
    pending = [r for r in rows if int(r["instance_id"]) not in done_ids]
    existing_rows = load_jsonl(output_jsonl) if (resume and output_jsonl.exists()) else []
    out_rows: List[Dict] = list(existing_rows)

    if not pending:
        print(f"[DiBS][steps={denoise_steps}] resume-skip {len(out_rows)}/{len(rows)}", flush=True)
        return out_rows

    tasks = []
    for idx, row in enumerate(pending):
        gpu_id = gpus[idx % len(gpus)] if gpus else gpu
        tasks.append(
            {
                "run_id": run_id,
                "solver": "DiBS",
                "puzzle": row["puzzle"],
                "instance_id": int(row["instance_id"]),
                "bucket_givens": int(row["bucket_givens"]),
                "model_path": model_path,
                "timeout_ms": timeout_ms,
                "max_nodes": max_nodes,
                "gpu": gpu_id,
                "denoise_steps": denoise_steps,
                "denoise_strategy": denoise_strategy,
                "mdm_decoding_strategy": mdm_decoding_strategy,
            }
        )

    effective_workers = workers
    if gpus:
        effective_workers = len(gpus)
    effective_workers = max(1, int(effective_workers))

    mode = "a" if (resume and output_jsonl.exists()) else "w"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[DiBS][steps={denoise_steps}] start pending={len(tasks)} workers={effective_workers} gpus={gpus if gpus else [gpu]}",
        flush=True,
    )
    with output_jsonl.open(mode, encoding="utf-8") as f:
        if effective_workers <= 1:
            for i, task in enumerate(tasks, start=1):
                row = solve_instance_task(task)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                out_rows.append(row)
                if i % 20 == 0 or i == len(tasks):
                    print(f"[DiBS][steps={denoise_steps}] {len(out_rows)}/{len(rows)} done", flush=True)
        else:
            mp_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=effective_workers, mp_context=mp_ctx) as ex:
                futures = [ex.submit(solve_instance_task, t) for t in tasks]
                finished = 0
                for fut in as_completed(futures):
                    row = fut.result()
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    out_rows.append(row)
                    finished += 1
                    if finished % 20 == 0 or finished == len(tasks):
                        print(f"[DiBS][steps={denoise_steps}] {len(out_rows)}/{len(rows)} done", flush=True)
    out_rows.sort(key=lambda x: int(x["instance_id"]))
    return out_rows


def build_step_comparison(step_summaries: Dict[int, Dict]) -> Dict:
    base = step_summaries[min(step_summaries.keys())]
    base_solved = float(base["solved_pct"])
    base_time = float(base["time_ms"]["all"]["mean"])
    base_nodes = float(base["nodes"]["all"]["mean"])
    base_backtracks = float(base["backtracks"]["all"]["mean"])
    base_model_calls = float(base["model_calls"]["all"]["mean"])
    base_model_time = float(base["model_time_ms"]["all"]["mean"])
    step_rows = []
    for step in sorted(step_summaries.keys()):
        cur = step_summaries[step]
        cur_solved = float(cur["solved_pct"])
        cur_time = float(cur["time_ms"]["all"]["mean"])
        cur_nodes = float(cur["nodes"]["all"]["mean"])
        cur_backtracks = float(cur["backtracks"]["all"]["mean"])
        cur_model_calls = float(cur["model_calls"]["all"]["mean"])
        cur_model_time = float(cur["model_time_ms"]["all"]["mean"])
        step_rows.append(
            {
                "steps": step,
                "dibs_solved_pct": cur_solved,
                "dibs_time_mean": cur_time,
                "dibs_nodes_mean": cur_nodes,
                "dibs_backtracks_mean": cur_backtracks,
                "dibs_model_calls_mean": cur_model_calls,
                "dibs_model_time_mean": cur_model_time,
                "delta_solved_pp_vs_step1": cur_solved - base_solved,
                "delta_time_pct_vs_step1": ((cur_time - base_time) / base_time * 100.0) if base_time > 0 else 0.0,
                "delta_nodes_pct_vs_step1": ((cur_nodes - base_nodes) / base_nodes * 100.0) if base_nodes > 0 else 0.0,
                "delta_backtracks_pct_vs_step1": ((cur_backtracks - base_backtracks) / base_backtracks * 100.0)
                if base_backtracks > 0
                else 0.0,
                "delta_model_calls_pct_vs_step1": ((cur_model_calls - base_model_calls) / base_model_calls * 100.0)
                if base_model_calls > 0
                else 0.0,
                "delta_model_time_pct_vs_step1": ((cur_model_time - base_model_time) / base_model_time * 100.0)
                if base_model_time > 0
                else 0.0,
            }
        )
    return {"rows": step_rows}


def pareto_candidates(step_rows: List[Dict]) -> List[Dict]:
    out = []
    for i, row_i in enumerate(step_rows):
        dominated = False
        for j, row_j in enumerate(step_rows):
            if i == j:
                continue
            better_or_eq_time = row_j["dibs_time_mean"] <= row_i["dibs_time_mean"]
            better_or_eq_nodes = row_j["dibs_nodes_mean"] <= row_i["dibs_nodes_mean"]
            strict = (
                row_j["dibs_time_mean"] < row_i["dibs_time_mean"]
                or row_j["dibs_nodes_mean"] < row_i["dibs_nodes_mean"]
            )
            if better_or_eq_time and better_or_eq_nodes and strict:
                dominated = True
                break
        if not dominated:
            out.append(row_i)
    out.sort(key=lambda r: (r["dibs_time_mean"], r["dibs_nodes_mean"]))
    return out


def write_report(path: Path, payload: Dict) -> None:
    lines: List[str] = []
    lines.append(f"# Explore 2 Report ({payload['run_id']})")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Dataset CSV: `{payload['dataset_csv']}`")
    lines.append(f"- Sample size: `{payload['max_puzzles']}`")
    lines.append(f"- Steps: `{payload['steps']}`")
    lines.append(f"- Denoise strategy: `{payload.get('denoise_strategy', 'legacy_repeat')}`")
    lines.append(f"- MDM decoding: `{payload.get('mdm_decoding_strategy', 'deterministic-cosine')}`")
    lines.append("")
    lines.append("## Step Comparison (vs step=1)")
    lines.append("")
    lines.append("| steps | time mean (ms) | nodes mean | backtracks mean | dTime (%) | dNodes (%) | dBacktracks (%) | dModelTime (%) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["step_comparison"]["rows"]:
        lines.append(
            f"| {row['steps']} | {row['dibs_time_mean']:.2f} | {row['dibs_nodes_mean']:.2f} | {row['dibs_backtracks_mean']:.2f} | "
            f"{row['delta_time_pct_vs_step1']:.2f} | {row['delta_nodes_pct_vs_step1']:.2f} | "
            f"{row['delta_backtracks_pct_vs_step1']:.2f} | {row['delta_model_time_pct_vs_step1']:.2f} |"
        )
    lines.append("")
    lines.append("## Pareto Candidates (time low, nodes low)")
    lines.append("")
    for row in payload["pareto"]:
        lines.append(
            f"- steps={row['steps']}: time_mean={row['dibs_time_mean']:.2f}ms, "
            f"nodes_mean={row['dibs_nodes_mean']:.2f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore 2: denoise step trade-off.")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=str(MODEL_DATA_DIR / "royle17_test.csv"),
        help="Table3-style csv with quizzes column",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-puzzles", type=int, default=5000)
    parser.add_argument("--steps", type=str, default="1,2,4,8")
    parser.add_argument(
        "--denoise-strategy",
        type=str,
        default="legacy_repeat",
        choices=["legacy_repeat", "mdm_iterative"],
        help="model call strategy inside DiBS",
    )
    parser.add_argument(
        "--mdm-decoding-strategy",
        type=str,
        default="deterministic-cosine",
        choices=["deterministic-cosine", "deterministic-linear"],
        help="used when denoise-strategy=mdm_iterative",
    )
    parser.add_argument("--timeout-ms", type=float, default=0.0)
    parser.add_argument("--max-nodes", type=int, default=1000000)
    parser.add_argument("--workers-dibs", type=int, default=1)
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id.strip() or now_ts()
    run_dir = Path(args.output_root) / run_id / "explore_2"
    data_dir = run_dir / "data"
    result_dir = run_dir / "results"
    report_dir = run_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    steps = sorted({int(x.strip()) for x in args.steps.split(",") if x.strip()})
    if 1 not in steps:
        steps = [1] + steps
    gpu_list: List[int] = []
    if args.gpus.strip():
        gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    sampled_path = data_dir / "sampled_puzzles.jsonl"
    sample_meta_path = data_dir / "sample_meta.json"

    if args.resume and sampled_path.exists():
        sampled_rows = load_jsonl(sampled_path)
        sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8")) if sample_meta_path.exists() else {}
    else:
        puzzles = load_table3_style_puzzles(
            dataset_csv=Path(args.dataset_csv),
            max_puzzles=args.max_puzzles,
            seed=args.seed,
        )
        sampled_rows = []
        for i, p in enumerate(puzzles):
            sampled_rows.append(
                {
                    "instance_id": i,
                    "bucket_givens": int(count_givens(p)),
                    "bucket_index": i,
                    "puzzle": p,
                    "source": "table3_csv",
                }
            )
        givens_hist: Dict[str, int] = {}
        for row in sampled_rows:
            g = str(int(row["bucket_givens"]))
            givens_hist[g] = givens_hist.get(g, 0) + 1
        sample_meta = {
            "source": "table3_style_csv",
            "dataset_csv": str(Path(args.dataset_csv).resolve()),
            "max_puzzles": args.max_puzzles,
            "total": len(sampled_rows),
            "givens_histogram": dict(sorted(givens_hist.items(), key=lambda kv: int(kv[0]))),
        }
        write_jsonl(sampled_path, sampled_rows)
        sample_meta_path.write_text(json.dumps(sample_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    run_meta = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "explore_2",
        "model": args.model,
        "dataset_csv": args.dataset_csv,
        "seed": args.seed,
        "max_puzzles": args.max_puzzles,
        "steps": steps,
        "denoise_strategy": args.denoise_strategy,
        "mdm_decoding_strategy": args.mdm_decoding_strategy,
        "timeout_ms": args.timeout_ms,
        "max_nodes": args.max_nodes,
        "workers_dibs": args.workers_dibs,
        "gpus": gpu_list,
        "gpu": args.gpu,
        "resume": bool(args.resume),
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    all_rows = sorted(sampled_rows, key=lambda x: int(x["instance_id"]))
    step_summaries: Dict[int, Dict] = {}
    for step in steps:
        out_path = result_dir / "per_instance" / f"DiBS_step_{step}" / "all.jsonl"
        rows = run_solver_all(
            run_id=run_id,
            rows=all_rows,
            output_jsonl=out_path,
            model_path=args.model,
            timeout_ms=args.timeout_ms,
            max_nodes=args.max_nodes,
            workers=max(1, int(args.workers_dibs)),
            gpus=gpu_list,
            gpu=args.gpu,
            denoise_steps=step,
            denoise_strategy=args.denoise_strategy,
            mdm_decoding_strategy=args.mdm_decoding_strategy,
            resume=args.resume,
        )
        step_summaries[step] = summarize_records(rows)
        (result_dir / "summaries").mkdir(parents=True, exist_ok=True)
        (result_dir / "summaries" / f"DiBS_step_{step}_summary.json").write_text(
            json.dumps(step_summaries[step], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    step_comparison = build_step_comparison(step_summaries)
    pareto = pareto_candidates(step_comparison["rows"])

    payload = {
        "run_id": run_id,
        "mode": "explore_2",
        "dataset_csv": str(Path(args.dataset_csv).resolve()),
        "max_puzzles": args.max_puzzles,
        "steps": steps,
        "denoise_strategy": args.denoise_strategy,
        "mdm_decoding_strategy": args.mdm_decoding_strategy,
        "sample_meta": sample_meta,
        "dibs_summary_by_step": step_summaries,
        "step_comparison": step_comparison,
        "pareto": pareto,
    }
    (result_dir / "explore_2_global_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(report_dir / "explore_2_report.md", payload)
    print(f"Explore 2 done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
