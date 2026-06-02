from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DATA_DIR = PROJECT_ROOT / "model" / "diffusion-vs-ar" / "data"


@dataclass
class SATInstance:
    instance_id: int
    task_name: str
    num_vars: int
    clauses: List[List[int]]
    token_input: str
    token_output: str
    known_solution: Optional[List[int]]


def _task_num_vars(task_name: str) -> int:
    if task_name not in {"3sat5", "3sat7", "3sat9"}:
        raise ValueError(f"Unsupported task: {task_name}")
    return int(task_name.replace("3sat", ""))


def _infer_token_pairs(tokens: Sequence[str], num_vars: int) -> Tuple[List[str], List[str]]:
    uniq = sorted(set(tokens), key=lambda s: (len(s), s))

    digits = [t for t in uniq if t.isdigit()]
    letters = [t for t in uniq if t.isalpha()]

    if letters:
        # 3sat7/3sat9 style: positives are digits, negatives are letters.
        # Some instances may not contain all literals in input clauses, so we
        # complete missing tokens from canonical task vocab.
        canon_digits = [str(i) for i in range(1, num_vars + 1)]
        canon_letters = [chr(ord("b") + i) for i in range(num_vars)]
        digit_set = set(digits)
        letter_set = set(letters)
        for t in canon_digits:
            digit_set.add(t)
        for t in canon_letters:
            letter_set.add(t)
        digits = sorted(digit_set, key=lambda x: int(x) if x.isdigit() else 10**9)
        letters = sorted(letter_set)
        digits = [d for d in digits if d in canon_digits][:num_vars]
        letters = [c for c in letters if c in canon_letters][:num_vars]
        if len(digits) != num_vars or len(letters) != num_vars:
            raise ValueError(
                f"Cannot infer mixed token polarity (num_vars={num_vars}, digits={digits}, letters={letters})."
            )
        return canon_digits, canon_letters

    # 3sat5 style: token space is numeric only and usually spans 2*num_vars values.
    # If a few tokens are missing in one instance, complete from canonical range.
    if len(uniq) < 2 * num_vars:
        canon_digits = [str(i) for i in range(2 * num_vars)]
        merged = set(uniq)
        for t in canon_digits:
            merged.add(t)
        digits = sorted(merged, key=int)
    else:
        digits = sorted(uniq, key=int)
    if len(digits) < 2 * num_vars:
        raise ValueError(f"Expected at least {2 * num_vars} numeric tokens, got {len(digits)}: {digits}")
    return digits[:num_vars], digits[num_vars:]


def parse_3sat_instance(raw_input: str, num_vars: int, raw_output: Optional[str] = None) -> Tuple[List[List[int]], Optional[List[int]]]:
    literal_tokens = [tok.strip() for part in raw_input.strip().split("/") for tok in part.split(",") if tok.strip()]
    pos_tokens, neg_tokens = _infer_token_pairs(literal_tokens, num_vars)

    token_to_lit: Dict[str, int] = {}
    for i in range(num_vars):
        token_to_lit[pos_tokens[i]] = i + 1
        token_to_lit[neg_tokens[i]] = -(i + 1)

    clauses: List[List[int]] = []
    for cidx, clause_text in enumerate(raw_input.strip().split("/")):
        toks = [t.strip() for t in clause_text.split(",") if t.strip()]
        if len(toks) != 3:
            raise ValueError(f"Clause {cidx} is not 3-SAT: {toks}")
        lits = []
        for tok in toks:
            if tok not in token_to_lit:
                raise ValueError(f"Unknown literal token `{tok}`")
            lits.append(token_to_lit[tok])
        clauses.append(lits)

    known_solution: Optional[List[int]] = None
    if raw_output:
        out_tokens = [t.strip() for t in raw_output.split(",") if t.strip()]
        if len(out_tokens) != num_vars:
            raise ValueError(f"Output assignment size mismatch: {len(out_tokens)} vs {num_vars}")
        assignment = [0] * (num_vars + 1)
        for tok in out_tokens:
            lit = token_to_lit.get(tok)
            if lit is None:
                raise ValueError(f"Unknown output token `{tok}`")
            v = abs(lit)
            val = 1 if lit > 0 else -1
            if assignment[v] != 0 and assignment[v] != val:
                raise ValueError(f"Conflicting assignment token for var {v}")
            assignment[v] = val
        for v in range(1, num_vars + 1):
            if assignment[v] == 0:
                assignment[v] = 1
        known_solution = assignment

    return clauses, known_solution


def load_split(task_name: str, split: str) -> List[SATInstance]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train/test")
    num_vars = _task_num_vars(task_name)
    path = MODEL_DATA_DIR / f"{task_name}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    rows: List[SATInstance] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_input = str(obj.get("input", "")).strip()
            raw_output = str(obj.get("output", "")).strip()
            clauses, known = parse_3sat_instance(raw_input, num_vars, raw_output)
            rows.append(
                SATInstance(
                    instance_id=i,
                    task_name=task_name,
                    num_vars=num_vars,
                    clauses=clauses,
                    token_input=raw_input,
                    token_output=raw_output,
                    known_solution=known,
                )
            )
    return rows


def assignment_satisfies(clauses: Sequence[Sequence[int]], assignment: Sequence[int]) -> bool:
    for clause in clauses:
        ok = False
        for lit in clause:
            v = abs(lit)
            if v >= len(assignment):
                return False
            val = assignment[v]
            if val == 0:
                continue
            if (lit > 0 and val > 0) or (lit < 0 and val < 0):
                ok = True
                break
        if not ok:
            return False
    return True
