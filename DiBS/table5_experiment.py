#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DiBS.sat_solver_dibs import DiBSSATSolver
from DiBS.sat_model_wrapper import SATMDMWrapper
from DiBS.table5_sat_data import assignment_satisfies, load_split

try:
    from pysat.solvers import Solver as PySATSolver
    HAS_PYSAT = True
except Exception:
    HAS_PYSAT = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "DiBS" / "results" / "parallel" / "Table_5"
DEFAULT_SAT_MODEL_PATHS = {
    "3sat5": PROJECT_ROOT / "model/diffusion-vs-ar/output/3sat_table5/20260503-table5-3sat5-T50-rerun/3sat5",
    "3sat7": PROJECT_ROOT / "model/diffusion-vs-ar/output/3sat_table5/20260503-table5-3sat7-ep900-lr5e4/3sat7",
    "3sat9": PROJECT_ROOT / "model/diffusion-vs-ar/output/3sat_table5/20260502-table5-strong/3sat9",
}


@dataclass
class Row:
    run_id: str
    task: str
    solver: str
    instance_id: int
    status: str
    solved: bool
    valid: bool
    time_ms: float
    nodes: int
    backtracks: int
    model_calls: int
    model_time_ms: float
    error: str = ""


def _literal_prior_scorer(train_rows):
    # P(var=True) prior from known training solutions, fallback 0.5
    stats: Dict[int, List[int]] = {}
    for row in train_rows:
        if not row.known_solution:
            continue
        for v in range(1, row.num_vars + 1):
            stats.setdefault(v, [0, 0])
            if row.known_solution[v] > 0:
                stats[v][0] += 1
            stats[v][1] += 1

    def scorer(var: int, _assignment, _clauses):
        pos, tot = stats.get(var, [1, 2])
        p = pos / max(1, tot)
        return float(p), float(1.0 - p)

    return scorer


def _load_sat_model_paths(args: argparse.Namespace) -> Dict[str, str]:
    paths = {k: str(v) for k, v in DEFAULT_SAT_MODEL_PATHS.items()}
    if args.sat_model_registry:
        registry_path = Path(args.sat_model_registry)
        obj = json.loads(registry_path.read_text(encoding="utf-8"))
        for task, value in obj.items():
            if isinstance(value, dict):
                value = value.get("checkpoint_dir") or value.get("path")
            if value:
                paths[task] = str(value)
    for override in args.sat_model:
        if "=" not in override:
            raise ValueError(f"--sat-model expects TASK=PATH, got {override}")
        task, path = override.split("=", 1)
        paths[task.strip()] = path.strip()
    return paths


def _solve_pysat(clauses: Sequence[Sequence[int]], timeout_ms: int) -> Tuple[Optional[List[int]], Dict]:
    if not HAS_PYSAT:
        return None, {"status": "error", "error": "python-sat not installed"}
    t0 = time.perf_counter()
    with PySATSolver(name="glucose4", bootstrap_with=[list(c) for c in clauses]) as s:
        solved = s.solve()
        model = s.get_model() if solved else None
        st = s.accum_stats()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if timeout_ms > 0 and elapsed_ms > timeout_ms:
        return None, {"status": "timeout", "time_ms": elapsed_ms, "nodes": 0, "backtracks": 0}
    if not solved or model is None:
        return None, {
            "status": "failed",
            "time_ms": elapsed_ms,
            "nodes": int(st.get("decisions", 0)),
            "backtracks": int(st.get("conflicts", 0)),
        }
    max_var = max(abs(l) for c in clauses for l in c)
    assignment = [0] * (max_var + 1)
    for lit in model:
        if abs(lit) <= max_var:
            assignment[abs(lit)] = 1 if lit > 0 else -1
    return assignment, {
        "status": "solved",
        "time_ms": elapsed_ms,
        "nodes": int(st.get("decisions", 0)),
        "backtracks": int(st.get("conflicts", 0)),
    }


def _make_cp_solver(args: argparse.Namespace, solver_name: str, scorer):
    if solver_name == "dibs_cp":
        return DiBSSATSolver(
            max_nodes=args.max_nodes,
            timeout_ms=args.timeout_ms,
            mrv_threshold=args.mrv_threshold,
            smart_call=bool(args.smart_call),
            literal_scorer=scorer,
            var_heuristic="mrv",
            value_heuristic="jw",
        )
    if solver_name == "dibs_mdm":
        return DiBSSATSolver(
            max_nodes=args.max_nodes,
            timeout_ms=args.timeout_ms,
            mrv_threshold=args.mrv_threshold,
            smart_call=bool(args.smart_call),
            literal_scorer=scorer,
            var_heuristic="mrv",
            value_heuristic="jw",
        )
    if solver_name == "dibs_literal_mdm":
        return DiBSSATSolver(
            max_nodes=args.max_nodes,
            timeout_ms=args.timeout_ms,
            mrv_threshold=args.mrv_threshold,
            smart_call=bool(args.smart_call),
            literal_ranker=scorer,
            var_heuristic="mrv",
            value_heuristic="literal_model",
            jw_margin_threshold=args.jw_margin_threshold,
        )
    if solver_name == "cp_mrv_jw":
        return DiBSSATSolver(
            max_nodes=args.max_nodes,
            timeout_ms=args.timeout_ms,
            literal_scorer=None,
            var_heuristic="mrv",
            value_heuristic="jw",
        )
    if solver_name == "cp_mrv_pos":
        return DiBSSATSolver(
            max_nodes=args.max_nodes,
            timeout_ms=args.timeout_ms,
            literal_scorer=None,
            var_heuristic="mrv",
            value_heuristic="pos",
        )
    if solver_name == "cp_first_pos":
        return DiBSSATSolver(
            max_nodes=args.max_nodes,
            timeout_ms=args.timeout_ms,
            literal_scorer=None,
            var_heuristic="first",
            value_heuristic="pos",
        )
    return None


def _summarize(rows: List[Row]) -> Dict:
    t = [r.time_ms for r in rows] or [0.0]
    n = [r.nodes for r in rows if r.solved] or [0]
    b = [r.backtracks for r in rows if r.solved] or [0]
    mc = [r.model_calls for r in rows if r.solved] or [0]
    mt = [r.model_time_ms for r in rows if r.solved] or [0.0]
    solved = sum(1 for r in rows if r.solved)
    return {
        "count": len(rows),
        "solved": solved,
        "solved_rate": solved / max(1, len(rows)),
        "time_mean_ms": float(np.mean(t)),
        "time_p95_ms": float(np.percentile(t, 95)),
        "nodes_mean": float(np.mean(n)),
        "backtracks_mean": float(np.mean(b)),
        "model_calls_mean": float(np.mean(mc)),
        "model_time_mean_ms": float(np.mean(mt)),
    }


def _load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(int(json.loads(line)["instance_id"]))
    return done


def _append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _latex_table(all_summaries: Dict[str, Dict]) -> str:
    solver_labels = {
        "pysat": "PySAT (Glucose4)",
        "dibs_cp": "DiBS-CP",
        "dibs_mdm": "DiBS-MDM",
        "dibs_literal_mdm": "DiBS-MDM-Literal",
        "cp_mrv_jw": "CP+MRV+JW",
        "cp_mrv_pos": "CP+MRV+True",
        "cp_first_pos": "CP+First+True",
    }
    solver_order = ["pysat", "cp_first_pos", "cp_mrv_pos", "cp_mrv_jw", "dibs_cp", "dibs_mdm", "dibs_literal_mdm"]
    task_order = ["3sat5", "3sat7", "3sat9"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l l r r r r r}",
        "\\toprule",
        "Task & Solver & Time p95 & Nodes Mean & Backtracks Mean & Model Calls & Model Time \\\\",
        "\\midrule",
    ]
    task_keys = sorted({key.split("::")[0] for key in all_summaries})
    task_keys.sort(key=lambda t: task_order.index(t) if t in task_order else len(task_order))
    for task_idx, task in enumerate(task_keys):
        solver_keys = [key.split("::")[1] for key in all_summaries if key.startswith(f"{task}::")]
        solver_keys.sort(key=lambda s: solver_order.index(s) if s in solver_order else len(solver_order))
        for solver_idx, solver in enumerate(solver_keys):
            s = all_summaries[f"{task}::{solver}"]
            task_label = task if solver_idx == 0 else ""
            lines.append(
                f"{task_label} & {solver_labels.get(solver, solver)} & "
                f"{s['time_p95_ms']:.2f} & {s['nodes_mean']:.2f} & "
                f"{s['backtracks_mean']:.2f} & {s['model_calls_mean']:.2f} & "
                f"{s['model_time_mean_ms']:.2f} \\\\"
            )
        if task_idx != len(task_keys) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\caption{Table5: DiBS on 3SAT.}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    run_dir = OUT_ROOT / args.run_id
    per_inst_dir = run_dir / "per_instance"
    summaries_dir = run_dir / "summaries"
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    all_summaries: Dict[str, Dict] = {}
    sat_model_paths = _load_sat_model_paths(args)
    sat_models: Dict[str, SATMDMWrapper] = {}

    for task in tasks:
        train_rows = load_split(task, "train")
        test_rows = load_split(task, "test")
        if args.max_instances > 0:
            test_rows = test_rows[: args.max_instances]
        scorer = _literal_prior_scorer(train_rows)
        mdm_model = None
        if "dibs_mdm" in solvers or "dibs_literal_mdm" in solvers:
            model_path = sat_model_paths.get(task)
            if not model_path:
                raise ValueError(f"No SAT MDM checkpoint configured for {task}")
            if task not in sat_models:
                sat_models[task] = SATMDMWrapper(model_path, task_name=task, device=args.device)
            mdm_model = sat_models[task]

        for solver_name in solvers:
            out_jsonl = per_inst_dir / task / f"{solver_name}.jsonl"
            done_ids = _load_done_ids(out_jsonl) if args.resume else set()
            collected: List[Row] = []

            if args.resume and out_jsonl.exists():
                with out_jsonl.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            collected.append(Row(**json.loads(line)))

            for row in test_rows:
                if row.instance_id in done_ids:
                    continue
                t0 = time.perf_counter()
                try:
                    if solver_name == "pysat":
                        assignment, m = _solve_pysat(row.clauses, args.timeout_ms)
                        status = m["status"]
                        solved = assignment is not None and assignment_satisfies(row.clauses, assignment)
                        r = Row(
                            run_id=args.run_id,
                            task=task,
                            solver=solver_name,
                            instance_id=row.instance_id,
                            status=status,
                            solved=solved,
                            valid=solved,
                            time_ms=float(m.get("time_ms", (time.perf_counter() - t0) * 1000.0)),
                            nodes=int(m.get("nodes", 0)),
                            backtracks=int(m.get("backtracks", 0)),
                            model_calls=0,
                            model_time_ms=0.0,
                        )
                    elif solver_name in {"dibs_cp", "dibs_mdm", "dibs_literal_mdm", "cp_mrv_jw", "cp_mrv_pos", "cp_first_pos"}:
                        row_scorer = scorer
                        if solver_name == "dibs_mdm":
                            if mdm_model is None:
                                raise RuntimeError(f"SAT MDM model not loaded for {task}")

                            def row_scorer(var, assignment, _clauses, model=mdm_model, raw=row.token_input):
                                return model.score_literal_pair(raw, assignment, var)

                        if solver_name == "dibs_literal_mdm":
                            if mdm_model is None:
                                raise RuntimeError(f"SAT MDM model not loaded for {task}")

                            def row_scorer(literals, assignment, _clauses, model=mdm_model, raw=row.token_input):
                                return model.score_literals(raw, assignment, literals)

                        solver = _make_cp_solver(args, solver_name, row_scorer)
                        if solver is None:
                            raise ValueError(f"Unknown CP solver: {solver_name}")
                        assignment, m = solver.solve(row.num_vars, row.clauses)
                        solved = assignment is not None and assignment_satisfies(row.clauses, assignment)
                        r = Row(
                            run_id=args.run_id,
                            task=task,
                            solver=solver_name,
                            instance_id=row.instance_id,
                            status=m.status,
                            solved=solved,
                            valid=solved,
                            time_ms=m.time_ms,
                            nodes=m.nodes,
                            backtracks=m.backtracks,
                            model_calls=m.model_calls,
                            model_time_ms=m.model_time_ms,
                        )
                    else:
                        raise ValueError(f"Unknown solver: {solver_name}")
                except Exception as e:
                    r = Row(
                        run_id=args.run_id,
                        task=task,
                        solver=solver_name,
                        instance_id=row.instance_id,
                        status="error",
                        solved=False,
                        valid=False,
                        time_ms=(time.perf_counter() - t0) * 1000.0,
                        nodes=0,
                        backtracks=0,
                        model_calls=0,
                        model_time_ms=0.0,
                        error=str(e),
                    )
                _append_jsonl(out_jsonl, asdict(r))
                collected.append(r)

            summary = _summarize(collected)
            key = f"{task}::{solver_name}"
            all_summaries[key] = summary
            _write_json(summaries_dir / f"{task}_{solver_name}_summary.json", summary)
            print(
                f"[{task}/{solver_name}] solved={summary['solved']}/{summary['count']} "
                f"time_mean={summary['time_mean_ms']:.2f}ms"
            )

    all_path = run_dir / f"{args.run_id}_all_summaries.json"
    tex_path = run_dir / f"{args.run_id}_table5.tex"
    report_path = run_dir / f"{args.run_id}_report.md"
    _write_json(all_path, all_summaries)
    tex_path.write_text(_latex_table(all_summaries), encoding="utf-8")
    report_lines = [
        f"# Table5 Report ({args.run_id})",
        "",
        f"- tasks: {args.tasks}",
        f"- solvers: {args.solvers}",
        f"- timeout_ms: {args.timeout_ms}",
        f"- max_nodes: {args.max_nodes}",
        "",
        "## Summaries",
    ]
    for key in sorted(all_summaries):
        report_lines.append(f"- {key}: {json.dumps(all_summaries[key], ensure_ascii=False)}")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"All summaries: {all_path}")
    print(f"LaTeX table : {tex_path}")
    print(f"Report      : {report_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Table5: DiBS for 3SAT")
    p.add_argument("--run-id", required=True)
    p.add_argument("--tasks", default="3sat5,3sat7,3sat9")
    p.add_argument("--solvers", default="pysat,dibs_mdm,cp_mrv_jw,cp_mrv_pos,cp_first_pos")
    p.add_argument("--timeout-ms", type=int, default=0)
    p.add_argument("--max-nodes", type=int, default=1_000_000)
    p.add_argument("--mrv-threshold", type=int, default=2)
    p.add_argument("--smart-call", type=int, default=1)
    p.add_argument("--jw-margin-threshold", type=float, default=0.25, help="Call MDM only when JW literal-ranking margin is at most this value.")
    p.add_argument("--max-instances", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)  # reserved for future parallel extension
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sat-model-registry", default="")
    p.add_argument("--sat-model", action="append", default=[], help="Override one checkpoint as TASK=PATH.")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
