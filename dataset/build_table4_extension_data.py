#!/usr/bin/env python3
"""Build unified Table4 extension datasets.

Pipeline:
1) load accepted external sources
2) ingest available external samples
3) fill shortage by generation
4) write train/test jsonl + txt + meta + checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ACCEPTED = PROJECT_ROOT / "dataset" / "accepted_sources.json"
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "table4_extension"


GENERALIZED_SUDOKU_SIZES = ("4x4", "16x16", "25x25")
NQUEENS_SIZES = ("8", "9", "10")

DIFFICULTY_LEVELS = ("very_hard", "hard", "medium", "easy", "very_easy")

SUDOKU_CONFIG = {
    # Difficulty gradient by givens count; lower givens => harder.
    # Five levels are created by equal-width integer bands over this range.
    "4x4": {
        "n": 4,
        "box": 2,
        "givens_range": (5, 11),
    },
    "16x16": {
        "n": 16,
        "box": 4,
        "givens_range": (72, 156),
    },
    "25x25": {
        "n": 25,
        "box": 5,
        "givens_range": (180, 380),
    },
}

NQUEENS_GIVENS_RANGE = {
    "8": (2, 7),
    "9": (2, 8),
    "10": (2, 9),
    "12": (3, 11),
    "14": (4, 13),
    "16": (4, 15),
}


@dataclass
class Record:
    puzzle: str
    solution: str
    task_family: str
    size: str
    source: str
    givens: int
    difficulty: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_givens_sudoku(s: str) -> int:
    return sum(1 for c in s if c != "0")


def count_givens_nqueens(s: str) -> int:
    return s.count("Q")


def _build_equal_width_bands(min_givens: int, max_givens: int) -> Dict[str, Tuple[int, int]]:
    min_givens = int(min_givens)
    max_givens = int(max_givens)
    if max_givens < min_givens:
        max_givens = min_givens
    span = (max_givens - min_givens + 1)
    bands: Dict[str, Tuple[int, int]] = {}
    for i, level in enumerate(DIFFICULTY_LEVELS):
        lo = min_givens + (span * i) // len(DIFFICULTY_LEVELS)
        if i == len(DIFFICULTY_LEVELS) - 1:
            hi = max_givens
        else:
            hi = min_givens + (span * (i + 1)) // len(DIFFICULTY_LEVELS) - 1
        hi = max(lo, min(hi, max_givens))
        bands[level] = (lo, hi)
    return bands


def _normalize_difficulty_label(label: str) -> str:
    d = (label or "").strip().lower()
    if d in DIFFICULTY_LEVELS:
        return d
    if d in ("hardest", "very-hard", "vhard"):
        return "very_hard"
    if d in ("hard",):
        return "hard"
    if d in ("medium", "mid"):
        return "medium"
    if d in ("easy",):
        return "easy"
    if d in ("very_easy", "very-easy", "veasy"):
        return "very_easy"
    return "medium"


def _sudoku_difficulty_bands(size: str) -> Dict[str, Tuple[int, int]]:
    conf = SUDOKU_CONFIG[size]
    gmin, gmax = conf["givens_range"]
    return _build_equal_width_bands(gmin, gmax)


def _nqueens_difficulty_bands(size: str) -> Dict[str, Tuple[int, int]]:
    n = int(size)
    if size in NQUEENS_GIVENS_RANGE:
        gmin, gmax = NQUEENS_GIVENS_RANGE[size]
    else:
        gmin = max(2, int(round(0.2 * n)))
        gmax = max(gmin, min(n - 1, int(round(0.9 * n))))
    return _build_equal_width_bands(gmin, gmax)


def _difficulty_from_bands(givens: int, bands: Dict[str, Tuple[int, int]]) -> str:
    for level in DIFFICULTY_LEVELS:
        lo, hi = bands[level]
        if lo <= givens <= hi:
            return level
    lo0, _ = bands[DIFFICULTY_LEVELS[0]]
    if givens < lo0:
        return DIFFICULTY_LEVELS[0]
    return DIFFICULTY_LEVELS[-1]


def classify_sudoku_difficulty(n: int, givens: int) -> str:
    size = f"{n}x{n}"
    bands = _sudoku_difficulty_bands(size)
    return _difficulty_from_bands(givens, bands)


def classify_nqueens_difficulty(n: int, givens: int) -> str:
    bands = _nqueens_difficulty_bands(str(n))
    return _difficulty_from_bands(givens, bands)


def _sudoku_char_to_val(ch: str, n: int) -> int:
    if ch in ("0", "."):
        return 0
    if n <= 16:
        alphabet = "0123456789ABCDEFG"
        idx = alphabet.find(ch.upper())
        if 0 <= idx <= n:
            return idx
    if ch.isdigit():
        val = int(ch)
        if 0 <= val <= n:
            return val
    val = ord(ch.upper()) - ord("A") + 1
    if 1 <= val <= n:
        return val
    return -1


def _parse_sudoku_grid(s: str, n: int) -> Optional[List[List[int]]]:
    need = n * n
    if len(s) < need:
        return None
    vals = []
    for ch in s[:need]:
        v = _sudoku_char_to_val(ch, n)
        if v < 0:
            return None
        vals.append(v)
    return [vals[r * n : (r + 1) * n] for r in range(n)]


def _validate_sudoku_pair(puzzle: str, solution: str, n: int, box: int) -> bool:
    p_grid = _parse_sudoku_grid(puzzle, n)
    s_grid = _parse_sudoku_grid(solution, n)
    if p_grid is None or s_grid is None:
        return False

    expected = set(range(1, n + 1))
    for r in range(n):
        for c in range(n):
            pv = p_grid[r][c]
            sv = s_grid[r][c]
            if sv <= 0 or sv > n:
                return False
            if pv != 0 and pv != sv:
                return False

    for i in range(n):
        row = set(s_grid[i])
        col = set(s_grid[r][i] for r in range(n))
        if row != expected or col != expected:
            return False
    for br in range(0, n, box):
        for bc in range(0, n, box):
            block = set()
            for r in range(br, br + box):
                for c in range(bc, bc + box):
                    block.add(s_grid[r][c])
            if block != expected:
                return False
    return True


def _parse_nqueens_rows(board: str, n: int, require_complete: bool) -> Optional[List[int]]:
    need = n * n
    if len(board) < need:
        return None
    rows = [-1] * n
    for r in range(n):
        line = board[r * n : (r + 1) * n]
        q_cols = [c for c, ch in enumerate(line) if ch == "Q"]
        if require_complete:
            if len(q_cols) != 1:
                return None
            rows[r] = q_cols[0]
        else:
            if len(q_cols) > 1:
                return None
            rows[r] = q_cols[0] if q_cols else -1
    return rows


def _validate_nqueens_solution_rows(rows: Sequence[int], n: int) -> bool:
    if any(c < 0 or c >= n for c in rows):
        return False
    cols = set()
    d1 = set()
    d2 = set()
    for r, c in enumerate(rows):
        if c in cols:
            return False
        k1 = r - c
        k2 = r + c
        if k1 in d1 or k2 in d2:
            return False
        cols.add(c)
        d1.add(k1)
        d2.add(k2)
    return True


def _validate_nqueens_pair(puzzle: str, solution: str, n: int) -> bool:
    p_rows = _parse_nqueens_rows(puzzle, n, require_complete=False)
    s_rows = _parse_nqueens_rows(solution, n, require_complete=True)
    if p_rows is None or s_rows is None:
        return False
    if not _validate_nqueens_solution_rows(s_rows, n):
        return False
    for r in range(n):
        if p_rows[r] >= 0 and p_rows[r] != s_rows[r]:
            return False
    return True


def _record_quality_check(
    rec: Record,
    ensure_unique: bool,
    uniqueness_nodes: int,
    uniqueness_timeout_sec: float,
    check_unique: bool = True,
) -> Tuple[bool, str]:
    if rec.task_family == "generalized_sudoku":
        n = int(rec.size.split("x")[0])
        conf = SUDOKU_CONFIG.get(rec.size)
        if not conf:
            return False, "unsupported_sudoku_size"
        box = conf["box"]
        if not _validate_sudoku_pair(rec.puzzle, rec.solution, n=n, box=box):
            return False, "invalid_sudoku_pair"
        if ensure_unique and check_unique:
            p_grid = _parse_sudoku_grid(rec.puzzle, n)
            if p_grid is None:
                return False, "invalid_sudoku_puzzle_parse"
            sol_cnt = _count_sudoku_solutions(
                [row[:] for row in p_grid],
                n=n,
                box=box,
                limit=2,
                max_nodes=uniqueness_nodes,
                timeout_sec=uniqueness_timeout_sec,
            )
            if sol_cnt != 1:
                return False, "non_unique_sudoku"
        return True, "ok"

    if rec.task_family == "nqueens":
        n = int(rec.size)
        if not _validate_nqueens_pair(rec.puzzle, rec.solution, n=n):
            return False, "invalid_nqueens_pair"
        if ensure_unique and check_unique:
            p_rows = _parse_nqueens_rows(rec.puzzle, n, require_complete=False)
            if p_rows is None:
                return False, "invalid_nqueens_puzzle_parse"
            fixed = {r: c for r, c in enumerate(p_rows) if c >= 0}
            if _count_nqueens_solutions(fixed, n=n, limit=2) != 1:
                return False, "non_unique_nqueens"
        return True, "ok"

    return False, "unsupported_family"


def _symbol_for(n: int, value: int) -> str:
    if value == 0:
        return "0"
    if n <= 16:
        alphabet = "0123456789ABCDEFG"
        return alphabet[value]
    # 25x25 uses A-Y for 1..25
    return chr(ord("A") + value - 1)


def _make_base_sudoku(n: int, box: int) -> List[List[int]]:
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            row.append((r * box + r // box + c) % n + 1)
        grid.append(row)
    return grid


def _permute_sudoku(grid: List[List[int]], n: int, box: int, rng: random.Random) -> List[List[int]]:
    rows = list(range(n))
    cols = list(range(n))
    digits = list(range(1, n + 1))

    # shuffle row bands and rows within each band
    bands = [rows[i : i + box] for i in range(0, n, box)]
    rng.shuffle(bands)
    rows = []
    for band in bands:
        rng.shuffle(band)
        rows.extend(band)

    # shuffle col stacks and cols within each stack
    stacks = [cols[i : i + box] for i in range(0, n, box)]
    rng.shuffle(stacks)
    cols = []
    for st in stacks:
        rng.shuffle(st)
        cols.extend(st)

    rng.shuffle(digits)
    remap = {i + 1: digits[i] for i in range(n)}

    out = []
    for r in rows:
        row = []
        for c in cols:
            row.append(remap[grid[r][c]])
        out.append(row)
    return out


def _sudoku_grid_to_str(grid: List[List[int]], n: int) -> str:
    return "".join(_symbol_for(n, v) for row in grid for v in row)


def _sudoku_candidates(grid: List[List[int]], n: int, box: int, r: int, c: int) -> List[int]:
    if grid[r][c] != 0:
        return []
    used = set(grid[r])
    used.update(grid[i][c] for i in range(n))
    br = (r // box) * box
    bc = (c // box) * box
    for i in range(br, br + box):
        for j in range(bc, bc + box):
            used.add(grid[i][j])
    return [v for v in range(1, n + 1) if v not in used]


def _count_sudoku_solutions(
    grid: List[List[int]],
    n: int,
    box: int,
    limit: int = 2,
    max_nodes: int = 200000,
    timeout_sec: float = 1.0,
) -> int:
    """Count solutions up to `limit`; returns >=limit when ambiguous or budget exceeded."""
    start = time.perf_counter()
    nodes = 0
    count = 0

    def dfs() -> None:
        nonlocal nodes, count
        if count >= limit:
            return
        nodes += 1
        if nodes > max_nodes or (time.perf_counter() - start) > timeout_sec:
            count = limit
            return
        best_cell = None
        best_cand = None
        for r in range(n):
            for c in range(n):
                if grid[r][c] != 0:
                    continue
                cand = _sudoku_candidates(grid, n, box, r, c)
                if not cand:
                    return
                if best_cand is None or len(cand) < len(best_cand):
                    best_cell = (r, c)
                    best_cand = cand
                    if len(best_cand) == 1:
                        break
            if best_cand is not None and len(best_cand) == 1:
                break
        if best_cell is None:
            count += 1
            return
        r, c = best_cell
        for v in best_cand:
            grid[r][c] = v
            dfs()
            grid[r][c] = 0
            if count >= limit:
                return

    dfs()
    return count


def _count_nqueens_solutions(givens: Dict[int, int], n: int, limit: int = 2) -> int:
    cols = [False] * n
    diag1 = [False] * (2 * n)
    diag2 = [False] * (2 * n)
    pos = [-1] * n
    for r, c in givens.items():
        d1 = r - c + n
        d2 = r + c
        if cols[c] or diag1[d1] or diag2[d2]:
            return 0
        cols[c] = diag1[d1] = diag2[d2] = True
        pos[r] = c

    cnt = 0

    def dfs(row: int) -> None:
        nonlocal cnt
        if cnt >= limit:
            return
        if row == n:
            cnt += 1
            return
        if pos[row] >= 0:
            dfs(row + 1)
            return
        for c in range(n):
            d1 = row - c + n
            d2 = row + c
            if cols[c] or diag1[d1] or diag2[d2]:
                continue
            cols[c] = diag1[d1] = diag2[d2] = True
            pos[row] = c
            dfs(row + 1)
            pos[row] = -1
            cols[c] = diag1[d1] = diag2[d2] = False
            if cnt >= limit:
                return

    dfs(0)
    return cnt


def _build_sudoku_record(
    size: str,
    rng: random.Random,
    difficulty: str,
    ensure_unique: bool = True,
    max_attempts: int = 120,
    uniqueness_nodes: int = 300000,
    uniqueness_timeout_sec: float = 1.0,
) -> Record:
    conf = SUDOKU_CONFIG[size]
    n = conf["n"]
    box = conf["box"]
    bands = _sudoku_difficulty_bands(size)
    difficulty = _normalize_difficulty_label(difficulty)
    if difficulty not in bands:
        difficulty = "medium"
    gmin, gmax = bands[difficulty]
    gmin = max(1, min(gmin, n * n - 1))
    gmax = max(gmin, min(gmax, n * n - 1))

    for _ in range(max_attempts):
        givens = rng.randint(gmin, gmax)
        solution_grid = _permute_sudoku(_make_base_sudoku(n, box), n, box, rng)
        puzzle_grid = [row[:] for row in solution_grid]
        cells = [(r, c) for r in range(n) for c in range(n)]
        rng.shuffle(cells)
        to_remove = max(0, n * n - givens)
        for i in range(to_remove):
            r, c = cells[i]
            puzzle_grid[r][c] = 0
        if ensure_unique:
            test_grid = [row[:] for row in puzzle_grid]
            sol_cnt = _count_sudoku_solutions(
                test_grid,
                n=n,
                box=box,
                limit=2,
                max_nodes=uniqueness_nodes,
                timeout_sec=uniqueness_timeout_sec,
            )
            if sol_cnt != 1:
                continue
        puzzle = _sudoku_grid_to_str(puzzle_grid, n)
        solution = _sudoku_grid_to_str(solution_grid, n)
        if not _validate_sudoku_pair(puzzle, solution, n=n, box=box):
            continue
        giv = count_givens_sudoku(puzzle)
        return Record(
            puzzle=puzzle,
            solution=solution,
            task_family="generalized_sudoku",
            size=size,
            source="generated",
            givens=giv,
            difficulty=classify_sudoku_difficulty(n, giv),
        )
    raise RuntimeError(f"Failed to generate unique generalized_sudoku/{size} ({difficulty}) in {max_attempts} attempts.")


def _nq_is_safe(positions: List[int], row: int, col: int) -> bool:
    for r in range(row):
        c = positions[r]
        if c == col:
            return False
        if abs(r - row) == abs(c - col):
            return False
    return True


def _random_nq_solution(n: int, rng: random.Random) -> Optional[List[int]]:
    positions = [-1] * n

    def dfs(row: int) -> bool:
        if row == n:
            return True
        cols = list(range(n))
        rng.shuffle(cols)
        for col in cols:
            if _nq_is_safe(positions, row, col):
                positions[row] = col
                if dfs(row + 1):
                    return True
                positions[row] = -1
        return False

    if dfs(0):
        return positions
    return None


def _nq_positions_to_solution(positions: List[int], n: int) -> str:
    board = []
    for r in range(n):
        for c in range(n):
            board.append("Q" if positions[r] == c else ".")
    return "".join(board)


def _nq_positions_to_puzzle(positions: List[int], n: int, givens: int, rng: random.Random) -> str:
    rows = list(range(n))
    rng.shuffle(rows)
    keep_rows = set(rows[:givens])
    board = []
    for r in range(n):
        for c in range(n):
            if r in keep_rows and positions[r] == c:
                board.append("Q")
            else:
                board.append(".")
    return "".join(board)


def _build_nqueens_record(
    size: str,
    rng: random.Random,
    difficulty: str,
    ensure_unique: bool = True,
    max_attempts: int = 200,
) -> Record:
    n = int(size)
    bands = _nqueens_difficulty_bands(size)
    difficulty = _normalize_difficulty_label(difficulty)
    if difficulty not in bands:
        difficulty = "medium"
    gmin, gmax = bands[difficulty]
    gmin = max(1, min(gmin, n))
    gmax = max(gmin, min(gmax, n))

    for _ in range(max_attempts):
        solution_positions = _random_nq_solution(n, rng)
        if not solution_positions:
            continue
        givens = rng.randint(gmin, gmax)
        solution = _nq_positions_to_solution(solution_positions, n)
        puzzle = _nq_positions_to_puzzle(solution_positions, n, givens, rng)
        if not _validate_nqueens_pair(puzzle, solution, n=n):
            continue
        if ensure_unique:
            fixed = {}
            for r in range(n):
                row = puzzle[r * n : (r + 1) * n]
                c = row.find("Q")
                if c >= 0:
                    fixed[r] = c
            if _count_nqueens_solutions(fixed, n=n, limit=2) != 1:
                continue
        giv = count_givens_nqueens(puzzle)
        return Record(
            puzzle=puzzle,
            solution=solution,
            task_family="nqueens",
            size=size,
            source="generated",
            givens=giv,
            difficulty=classify_nqueens_difficulty(n, giv),
        )
    raise RuntimeError(f"Failed to generate unique {n}-Queens ({difficulty}) in {max_attempts} attempts.")


def _generate_candidate_worker(
    task_family: str,
    size: str,
    difficulty: str,
    seed: int,
    ensure_unique: bool,
    per_record_max_attempts: int,
    uniqueness_nodes: int,
    uniqueness_timeout_sec: float,
) -> Tuple[bool, Optional[Dict], str]:
    rng = random.Random(seed)
    try:
        if task_family == "generalized_sudoku":
            rec = _build_sudoku_record(
                size=size,
                rng=rng,
                difficulty=difficulty,
                ensure_unique=ensure_unique,
                max_attempts=per_record_max_attempts,
                uniqueness_nodes=uniqueness_nodes,
                uniqueness_timeout_sec=uniqueness_timeout_sec,
            )
        else:
            rec = _build_nqueens_record(
                size=size,
                rng=rng,
                difficulty=difficulty,
                ensure_unique=ensure_unique,
                max_attempts=per_record_max_attempts,
            )
    except Exception as exc:  # noqa: BLE001
        return False, None, f"build_failed:{type(exc).__name__}"

    ok, reason = _record_quality_check(
        rec,
        ensure_unique=ensure_unique,
        uniqueness_nodes=uniqueness_nodes,
        uniqueness_timeout_sec=uniqueness_timeout_sec,
        check_unique=False,
    )
    if not ok:
        return False, None, reason
    return True, asdict(rec), "ok"


def _read_plain_pairs(puzzle_path: Path, solution_path: Path, task_family: str, size: str, source: str) -> List[Record]:
    if not puzzle_path.exists() or not solution_path.exists():
        return []
    puzzles = [line.strip() for line in puzzle_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sols = [line.strip() for line in solution_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = min(len(puzzles), len(sols))
    out = []
    for i in range(n):
        p = puzzles[i]
        s = sols[i]
        givens = count_givens_sudoku(p) if task_family == "generalized_sudoku" else count_givens_nqueens(p)
        if task_family == "generalized_sudoku":
            n = int(size.split("x")[0])
            difficulty = classify_sudoku_difficulty(n, givens)
        else:
            difficulty = classify_nqueens_difficulty(int(size), givens)
        out.append(
            Record(
                puzzle=p,
                solution=s,
                task_family=task_family,
                size=size,
                source=source,
                givens=givens,
                difficulty=difficulty,
            )
        )
    return out


def _load_external_records(accepted_sources: Sequence[Dict]) -> Dict[Tuple[str, str], List[Record]]:
    grouped: Dict[Tuple[str, str], List[Record]] = {}
    for src in accepted_sources:
        fam = src["task_family"]
        size = src["size"]
        key = (fam, size)
        grouped.setdefault(key, [])
        source_id = src.get("source_id", "external")

        url = src.get("url", "")
        if url.startswith("file://"):
            relative = url[len("file://") :]
            # repository-local file reference
            p = PROJECT_ROOT / relative.lstrip("/")
            # infer optional pair path
            if "puzzles" in p.name:
                s = p.with_name(p.name.replace("puzzles", "solutions"))
                grouped[key].extend(_read_plain_pairs(p, s, fam, size, source=source_id))
    return grouped


def _dedupe(records: Iterable[Record]) -> List[Record]:
    seen = set()
    out = []
    for r in records:
        key = (r.task_family, r.size, r.puzzle)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _with_recomputed_difficulty(rec: Record) -> Record:
    if rec.task_family == "generalized_sudoku":
        n = int(rec.size.split("x")[0])
        difficulty = classify_sudoku_difficulty(n, rec.givens)
    elif rec.task_family == "nqueens":
        difficulty = classify_nqueens_difficulty(int(rec.size), rec.givens)
    else:
        difficulty = _normalize_difficulty_label(rec.difficulty)
    return Record(
        puzzle=rec.puzzle,
        solution=rec.solution,
        task_family=rec.task_family,
        size=rec.size,
        source=rec.source,
        givens=rec.givens,
        difficulty=difficulty,
    )


def _difficulty_target_counts(total: int) -> Dict[str, int]:
    base = total // len(DIFFICULTY_LEVELS)
    rem = total % len(DIFFICULTY_LEVELS)
    out = {level: base for level in DIFFICULTY_LEVELS}
    for i in range(rem):
        out[DIFFICULTY_LEVELS[i]] += 1
    return out


def _difficulty_counts(records: Sequence[Record]) -> Dict[str, int]:
    out = {level: 0 for level in DIFFICULTY_LEVELS}
    for r in records:
        d = _normalize_difficulty_label(r.difficulty)
        if d not in out:
            out[d] = 0
        out[d] += 1
    return out


def _summarize_givens(records: Sequence[Record]) -> Dict[str, float]:
    if not records:
        return {"min": 0.0, "p25": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}
    vals = sorted(r.givens for r in records)
    n = len(vals)

    def q(p: float) -> float:
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return float(vals[idx])

    return {
        "min": float(vals[0]),
        "p25": q(0.25),
        "mean": float(sum(vals) / n),
        "p75": q(0.75),
        "max": float(vals[-1]),
    }


def _write_split(out_dir: Path, split_name: str, records: Sequence[Record]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{split_name}.jsonl"
    txt_puzzle_path = out_dir / f"{split_name}.txt"
    txt_solution_path = out_dir / f"{split_name}_solutions.txt"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    txt_puzzle_path.write_text("\n".join(r.puzzle for r in records), encoding="utf-8")
    txt_solution_path.write_text("\n".join(r.solution for r in records), encoding="utf-8")

    return {
        "jsonl": str(jsonl_path),
        "txt": str(txt_puzzle_path),
        "solutions": str(txt_solution_path),
        "jsonl_sha256": sha256_file(jsonl_path),
        "txt_sha256": sha256_file(txt_puzzle_path),
        "solutions_sha256": sha256_file(txt_solution_path),
    }


def build_for_key(
    task_family: str,
    size: str,
    external_records: Dict[Tuple[str, str], List[Record]],
    train_count: int,
    test_count: int,
    rng: random.Random,
    max_generation_attempts: int,
    allow_duplicates: bool,
    ensure_unique_train: bool,
    ensure_unique_test: bool,
    per_record_max_attempts: int,
    uniqueness_nodes: int,
    uniqueness_timeout_sec: float,
    generation_workers: int,
    progress_every: int,
    difficulty_stratified: bool,
    generation_difficulty: str,
    progress_dir: Optional[Path] = None,
) -> Tuple[List[Record], List[Record], Dict]:
    required_total = train_count + test_count
    key = (task_family, size)
    t0 = time.time()

    from_external_raw = [_with_recomputed_difficulty(r) for r in _dedupe(external_records.get(key, []))]
    def _validate_external(records: Sequence[Record], ensure_unique_split: bool) -> Tuple[List[Record], Dict[str, int]]:
        valid: List[Record] = []
        rejected: Dict[str, int] = {}
        for rec in records:
            ok, reason = _record_quality_check(
                rec,
                ensure_unique=ensure_unique_split,
                uniqueness_nodes=uniqueness_nodes,
                uniqueness_timeout_sec=uniqueness_timeout_sec,
                check_unique=ensure_unique_split,
            )
            if ok:
                valid.append(rec)
            else:
                rejected[reason] = rejected.get(reason, 0) + 1
        return valid, rejected

    external_train_pool, external_rejected_train = _validate_external(from_external_raw, ensure_unique_train)
    external_test_pool, external_rejected_test = _validate_external(from_external_raw, ensure_unique_test)
    rng.shuffle(external_train_pool)
    rng.shuffle(external_test_pool)

    generation_difficulty = _normalize_difficulty_label(generation_difficulty)
    target_train_by_diff = _difficulty_target_counts(train_count) if difficulty_stratified else {d: 0 for d in DIFFICULTY_LEVELS}
    target_test_by_diff = _difficulty_target_counts(test_count) if difficulty_stratified else {d: 0 for d in DIFFICULTY_LEVELS}
    if not difficulty_stratified:
        target_train_by_diff[generation_difficulty] = train_count
        target_test_by_diff[generation_difficulty] = test_count

    selected_train: List[Record] = []
    selected_train_by_diff = {d: 0 for d in DIFFICULTY_LEVELS}
    seen = set()
    if difficulty_stratified:
        for rec in external_train_pool:
            d = _normalize_difficulty_label(rec.difficulty)
            if d not in selected_train_by_diff:
                continue
            if selected_train_by_diff[d] >= target_train_by_diff[d]:
                continue
            k = (rec.task_family, rec.size, rec.puzzle)
            if k in seen:
                continue
            seen.add(k)
            selected_train.append(rec)
            selected_train_by_diff[d] += 1
            if all(selected_train_by_diff[x] >= target_train_by_diff[x] for x in DIFFICULTY_LEVELS):
                break
    else:
        for rec in external_train_pool:
            if len(selected_train) >= train_count:
                break
            k = (rec.task_family, rec.size, rec.puzzle)
            if k in seen:
                continue
            seen.add(k)
            selected_train.append(rec)
            d = _normalize_difficulty_label(rec.difficulty)
            if d in selected_train_by_diff:
                selected_train_by_diff[d] += 1

    selected_test: List[Record] = []
    selected_test_by_diff = {d: 0 for d in DIFFICULTY_LEVELS}
    if difficulty_stratified:
        for rec in external_test_pool:
            d = _normalize_difficulty_label(rec.difficulty)
            if d not in selected_test_by_diff:
                continue
            if selected_test_by_diff[d] >= target_test_by_diff[d]:
                continue
            k = (rec.task_family, rec.size, rec.puzzle)
            if k in seen:
                continue
            seen.add(k)
            selected_test.append(rec)
            selected_test_by_diff[d] += 1
            if all(selected_test_by_diff[x] >= target_test_by_diff[x] for x in DIFFICULTY_LEVELS):
                break
    else:
        for rec in external_test_pool:
            if len(selected_test) >= test_count:
                break
            k = (rec.task_family, rec.size, rec.puzzle)
            if k in seen:
                continue
            seen.add(k)
            selected_test.append(rec)
            d = _normalize_difficulty_label(rec.difficulty)
            if d in selected_test_by_diff:
                selected_test_by_diff[d] += 1

    print(
        f"[{task_family}/{size}] target train={train_count} test={test_count}, "
        f"external_raw={len(from_external_raw)}, external_train_valid={len(external_train_pool)}, "
        f"external_test_valid={len(external_test_pool)}, workers={max(1, generation_workers)}, "
        f"unique(train/test)=({ensure_unique_train}/{ensure_unique_test})",
        flush=True,
    )

    progress_path = None
    if progress_dir is not None:
        progress_dir.mkdir(parents=True, exist_ok=True)

    def _write_progress(split_name: str, payload: Dict) -> None:
        if progress_dir is None:
            return
        path = progress_dir / f"progress_{task_family}_{size}_{split_name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _generate_for_split(
        split_name: str,
        need_by_diff: Dict[str, int],
        ensure_unique_split: bool,
    ) -> Tuple[List[Record], int, Dict[str, int]]:
        generated: List[Record] = []
        rejected: Dict[str, int] = {}
        attempts = 0
        remaining = {d: max(0, int(need_by_diff.get(d, 0))) for d in DIFFICULTY_LEVELS}
        need = sum(remaining.values())
        if need <= 0:
            return generated, attempts, rejected

        worker_count = max(1, generation_workers)
        submit_idx = 0
        split_t0 = time.time()
        batch_base = max(32, worker_count * 8)
        last_log_time = 0.0
        last_log_done = -1
        last_log_attempts = -1

        checkpoint_path: Optional[Path] = None
        if progress_dir is not None:
            checkpoint_path = progress_dir / f"generated_{task_family}_{size}_{split_name}.jsonl"

        def _append_checkpoint(rec: Record) -> None:
            if checkpoint_path is None:
                return
            with checkpoint_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        def _pick_requested_difficulty() -> Optional[str]:
            nonlocal submit_idx
            active = [d for d in DIFFICULTY_LEVELS if remaining[d] > 0]
            if not active:
                return None
            d = active[submit_idx % len(active)]
            submit_idx += 1
            return d

        def _handle_record(rec: Record, save_checkpoint: bool) -> None:
            d = _normalize_difficulty_label(rec.difficulty)
            if d not in remaining:
                rejected["unexpected_difficulty"] = rejected.get("unexpected_difficulty", 0) + 1
                return
            if remaining[d] <= 0:
                rejected["over_quota_difficulty"] = rejected.get("over_quota_difficulty", 0) + 1
                return
            rec_key = (rec.task_family, rec.size, rec.puzzle)
            if rec_key in seen and not allow_duplicates:
                rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                return
            if rec_key not in seen:
                seen.add(rec_key)
            generated.append(rec)
            remaining[d] -= 1
            if save_checkpoint:
                _append_checkpoint(rec)

        def _handle_result(ok: bool, payload: Optional[Dict], reason: str) -> None:
            if not ok:
                rejected[reason] = rejected.get(reason, 0) + 1
                return
            rec = _with_recomputed_difficulty(Record(**payload))
            _handle_record(rec, save_checkpoint=True)

        def _restore_from_checkpoint() -> int:
            if checkpoint_path is None or not checkpoint_path.exists():
                return 0
            restored = 0
            with checkpoint_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        rec = _with_recomputed_difficulty(Record(**payload))
                        before = len(generated)
                        _handle_record(rec, save_checkpoint=False)
                        if len(generated) > before:
                            restored += 1
                    except Exception:
                        rejected["checkpoint_parse_error"] = rejected.get("checkpoint_parse_error", 0) + 1
            if restored > 0:
                print(
                    f"[{task_family}/{size}][{split_name}] restored_from_checkpoint={restored}",
                    flush=True,
                )
            return restored

        def _maybe_log(force: bool = False) -> None:
            nonlocal last_log_time, last_log_done, last_log_attempts
            if progress_every <= 0 and not force:
                return
            done = need - sum(remaining.values())
            now = time.time()
            should_log = force
            if not should_log:
                if done > 0 and done % progress_every == 0 and done != last_log_done:
                    should_log = True
                elif attempts > 0 and attempts % max(1000, progress_every * 20) == 0 and attempts != last_log_attempts:
                    should_log = True
                elif (now - last_log_time) >= 60:
                    should_log = True
            if not should_log:
                return
            elapsed = time.time() - split_t0
            rate = done / max(1e-9, elapsed)
            remain = max(0, need - done)
            eta = remain / max(1e-9, rate) if rate > 0 else float("inf")
            eta_text = f"{eta:.1f}s" if eta != float("inf") else "inf"
            success_rate = (done / attempts) if attempts > 0 else 0.0
            rejected_total = sum(rejected.values())
            print(
                f"[{task_family}/{size}][{split_name}] done={done}/{need}, "
                f"attempts={attempts}, rejected={rejected_total}, succ={success_rate:.4f}, elapsed={elapsed:.1f}s, eta={eta_text}",
                flush=True,
            )
            _write_progress(
                split_name,
                {
                    "task_family": task_family,
                    "size": size,
                    "split": split_name,
                    "done": done,
                    "need": need,
                    "attempts": attempts,
                    "elapsed_sec": elapsed,
                    "eta_sec": eta if eta != float("inf") else None,
                    "success_rate": success_rate,
                    "rejected_total": rejected_total,
                    "rejected_breakdown": dict(sorted(rejected.items(), key=lambda x: x[1], reverse=True)[:10]),
                    "remaining_by_difficulty": remaining,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            last_log_time = now
            last_log_done = done
            last_log_attempts = attempts

        _restore_from_checkpoint()
        _maybe_log(force=True)

        if worker_count == 1:
            while sum(remaining.values()) > 0 and attempts < max_generation_attempts:
                difficulty = _pick_requested_difficulty()
                if difficulty is None:
                    break
                seed_i = rng.randint(0, 2**31 - 1)
                ok, payload, reason = _generate_candidate_worker(
                    task_family=task_family,
                    size=size,
                    difficulty=difficulty,
                    seed=seed_i,
                    ensure_unique=ensure_unique_split,
                    per_record_max_attempts=per_record_max_attempts,
                    uniqueness_nodes=uniqueness_nodes,
                    uniqueness_timeout_sec=uniqueness_timeout_sec,
                )
                attempts += 1
                _handle_result(ok, payload, reason)
                if progress_every > 0 and attempts % (progress_every * 5) == 0:
                    _maybe_log(force=True)
                else:
                    _maybe_log(force=False)
        else:
            mp_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as ex:
                while sum(remaining.values()) > 0 and attempts < max_generation_attempts:
                    remaining_attempts = max_generation_attempts - attempts
                    outstanding = sum(remaining.values())
                    batch = min(remaining_attempts, max(batch_base, outstanding * 4))
                    futures = []
                    for _ in range(batch):
                        difficulty = _pick_requested_difficulty()
                        if difficulty is None:
                            break
                        seed_i = rng.randint(0, 2**31 - 1)
                        futures.append(
                            ex.submit(
                                _generate_candidate_worker,
                                task_family,
                                size,
                                difficulty,
                                seed_i,
                                ensure_unique_split,
                                per_record_max_attempts,
                                uniqueness_nodes,
                                uniqueness_timeout_sec,
                            )
                        )
                    if not futures:
                        break
                    attempts += len(futures)
                    for fut in as_completed(futures):
                        ok, payload, reason = fut.result()
                        _handle_result(ok, payload, reason)
                        if sum(remaining.values()) <= 0:
                            break
                    if progress_every > 0 and attempts % (progress_every * 5) == 0:
                        _maybe_log(force=True)
                    else:
                        _maybe_log(force=False)

        if sum(remaining.values()) > 0:
            if allow_duplicates:
                while sum(remaining.values()) > 0:
                    difficulty = _pick_requested_difficulty()
                    if difficulty is None:
                        break
                    seed_i = rng.randint(0, 2**31 - 1)
                    ok, payload, reason = _generate_candidate_worker(
                        task_family=task_family,
                        size=size,
                        difficulty=difficulty,
                        seed=seed_i,
                        ensure_unique=ensure_unique_split,
                        per_record_max_attempts=per_record_max_attempts,
                        uniqueness_nodes=uniqueness_nodes,
                        uniqueness_timeout_sec=uniqueness_timeout_sec,
                    )
                    attempts += 1
                    if not ok:
                        rejected[reason] = rejected.get(reason, 0) + 1
                        continue
                    rec = _with_recomputed_difficulty(Record(**payload))
                    d = _normalize_difficulty_label(rec.difficulty)
                    if remaining.get(d, 0) <= 0:
                        rejected["over_quota_difficulty"] = rejected.get("over_quota_difficulty", 0) + 1
                        continue
                    rec_key = (rec.task_family, rec.size, rec.puzzle)
                    if rec_key not in seen:
                        seen.add(rec_key)
                    generated.append(rec)
                    remaining[d] -= 1
                    _append_checkpoint(rec)
                    _maybe_log(force=False)
            else:
                raise RuntimeError(
                    f"Insufficient samples for {task_family}/{size}/{split_name}: "
                    f"need={need}, got={need - sum(remaining.values())}, remaining={remaining}, "
                    f"attempts={attempts}, unique={ensure_unique_split}, workers={worker_count}. "
                    f"Increase --max-generation-attempts or reduce split size."
                )

        _maybe_log(force=True)
        return generated, attempts, rejected

    if difficulty_stratified:
        need_train_by_diff = {
            d: max(0, target_train_by_diff[d] - selected_train_by_diff[d]) for d in DIFFICULTY_LEVELS
        }
        need_test_by_diff = {
            d: max(0, target_test_by_diff[d] - selected_test_by_diff[d]) for d in DIFFICULTY_LEVELS
        }
    else:
        need_train_by_diff = {d: 0 for d in DIFFICULTY_LEVELS}
        need_test_by_diff = {d: 0 for d in DIFFICULTY_LEVELS}
        need_train_by_diff[generation_difficulty] = max(0, train_count - len(selected_train))
        need_test_by_diff[generation_difficulty] = max(0, test_count - len(selected_test))
    gen_train, attempts_train, generated_rejected_train = _generate_for_split(
        split_name="train", need_by_diff=need_train_by_diff, ensure_unique_split=ensure_unique_train
    )
    gen_test, attempts_test, generated_rejected_test = _generate_for_split(
        split_name="test", need_by_diff=need_test_by_diff, ensure_unique_split=ensure_unique_test
    )

    train = [*selected_train, *gen_train]
    test = [*selected_test, *gen_test]
    if len(train) != train_count or len(test) != test_count:
        raise RuntimeError(
            f"Split size mismatch for {task_family}/{size}: train={len(train)}/{train_count}, test={len(test)}/{test_count}"
        )

    source_breakdown = {}
    for r in [*train, *test]:
        source_breakdown[r.source] = source_breakdown.get(r.source, 0) + 1

    return train, test, {
        "task_family": task_family,
        "size": size,
        "difficulty_stratified": bool(difficulty_stratified),
        "generation_difficulty": generation_difficulty,
        "required_total": required_total,
        "external_available_raw": len(from_external_raw),
        "external_validated_train": len(external_train_pool),
        "external_validated_test": len(external_test_pool),
        "external_rejected_train": external_rejected_train,
        "external_rejected_test": external_rejected_test,
        "external_used_train": len(selected_train),
        "external_used_test": len(selected_test),
        "external_used": len(selected_train) + len(selected_test),
        "generated_used_train": len(gen_train),
        "generated_used_test": len(gen_test),
        "generated_used": len(gen_train) + len(gen_test),
        "generated_rejected_train": generated_rejected_train,
        "generated_rejected_test": generated_rejected_test,
        "attempts_train": attempts_train,
        "attempts_test": attempts_test,
        "attempts_total": attempts_train + attempts_test,
        "ensure_unique_train": bool(ensure_unique_train),
        "ensure_unique_test": bool(ensure_unique_test),
        "allow_duplicates": bool(allow_duplicates),
        "generation_workers": int(max(1, generation_workers)),
        "source_breakdown": source_breakdown,
        "difficulty_train": _difficulty_counts(train),
        "difficulty_test": _difficulty_counts(test),
        "givens_train_stats": _summarize_givens(train),
        "givens_test_stats": _summarize_givens(test),
        "avg_givens_train": sum(r.givens for r in train) / len(train),
        "avg_givens_test": sum(r.givens for r in test) / len(test),
        "elapsed_sec": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified Table4 extension datasets.")
    parser.add_argument("--accepted-sources", type=str, default=str(DEFAULT_ACCEPTED))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--train-count", type=int, default=1500)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-generation-attempts", type=int, default=1000000)
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate puzzles when unique generation saturates.")
    parser.add_argument("--disable-unique-check", action="store_true", help="Disable uniqueness verification (not recommended for formal runs).")
    parser.add_argument("--disable-train-unique-check", action="store_true", help="Disable uniqueness verification for train split.")
    parser.add_argument("--disable-test-unique-check", action="store_true", help="Disable uniqueness verification for test split.")
    parser.add_argument("--per-record-max-attempts", type=int, default=200, help="Max retries for building one generated record.")
    parser.add_argument("--uniqueness-nodes", type=int, default=300000, help="Max DFS nodes per uniqueness check (sudoku).")
    parser.add_argument("--uniqueness-timeout-sec", type=float, default=1.0, help="Wall-clock timeout per uniqueness check (sudoku).")
    parser.add_argument("--generation-workers", type=int, default=1, help="Worker processes for generation attempts.")
    parser.add_argument("--progress-every", type=int, default=500, help="Print generation progress every N generated records.")
    parser.add_argument("--disable-difficulty-stratification", action="store_true", help="Do not enforce equal difficulty buckets for train/test.")
    parser.add_argument("--generation-difficulty", type=str, default="easy", help="Requested difficulty label when stratification is disabled.")
    parser.add_argument("--sudoku-sizes", type=str, default=",".join(GENERALIZED_SUDOKU_SIZES), help="comma separated, e.g. 4x4,16x16,25x25")
    parser.add_argument("--nqueens-sizes", type=str, default=",".join(NQUEENS_SIZES), help="comma separated, e.g. 8,9,10,12")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    start = time.time()

    accepted_path = Path(args.accepted_sources)
    if accepted_path.exists():
        accepted_sources = json.loads(accepted_path.read_text(encoding="utf-8"))
    else:
        accepted_sources = []
    external_records = _load_external_records(accepted_sources)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sudoku_sizes = [s.strip() for s in args.sudoku_sizes.split(",") if s.strip()]
    nqueens_sizes = [s.strip() for s in args.nqueens_sizes.split(",") if s.strip()]
    tasks = [("generalized_sudoku", s) for s in sudoku_sizes] + [("nqueens", s) for s in nqueens_sizes]
    ensure_unique_train = (not args.disable_unique_check) and (not args.disable_train_unique_check)
    ensure_unique_test = (not args.disable_unique_check) and (not args.disable_test_unique_check)

    difficulty_stratified = not args.disable_difficulty_stratification
    generation_difficulty = _normalize_difficulty_label(args.generation_difficulty)

    meta = {
        "difficulty_schema": "givens_equal_width_v2" if difficulty_stratified else "single_level_v1",
        "difficulty_levels": list(DIFFICULTY_LEVELS),
        "difficulty_stratified": difficulty_stratified,
        "generation_difficulty": generation_difficulty,
        "run_id": time.strftime("%Y%m%d-%H%M%S"),
        "seed": args.seed,
        "train_count": args.train_count,
        "test_count": args.test_count,
        "ensure_unique_global": not args.disable_unique_check,
        "ensure_unique_train": ensure_unique_train,
        "ensure_unique_test": ensure_unique_test,
        "per_record_max_attempts": args.per_record_max_attempts,
        "uniqueness_nodes": args.uniqueness_nodes,
        "uniqueness_timeout_sec": args.uniqueness_timeout_sec,
        "generation_workers": int(max(1, args.generation_workers)),
        "progress_every": args.progress_every,
        "accepted_sources_file": str(accepted_path),
        "sudoku_sizes": sudoku_sizes,
        "nqueens_sizes": nqueens_sizes,
        "tasks": [],
    }

    for task_family, size in tasks:
        progress_dir = output_root / "progress"
        train, test, stats = build_for_key(
            task_family=task_family,
            size=size,
            external_records=external_records,
            train_count=args.train_count,
            test_count=args.test_count,
            rng=rng,
            max_generation_attempts=args.max_generation_attempts,
            allow_duplicates=args.allow_duplicates,
            ensure_unique_train=ensure_unique_train,
            ensure_unique_test=ensure_unique_test,
            per_record_max_attempts=args.per_record_max_attempts,
            uniqueness_nodes=args.uniqueness_nodes,
            uniqueness_timeout_sec=args.uniqueness_timeout_sec,
            generation_workers=args.generation_workers,
            progress_every=args.progress_every,
            difficulty_stratified=difficulty_stratified,
            generation_difficulty=generation_difficulty,
            progress_dir=progress_dir,
        )

        size_dir = output_root / task_family / size
        train_art = _write_split(size_dir, "train", train)
        test_art = _write_split(size_dir, "test", test)
        stats["train_artifacts"] = train_art
        stats["test_artifacts"] = test_art
        meta["tasks"].append(stats)
        print(
            f"[{task_family}/{size}] train={len(train)} test={len(test)} "
            f"external={stats['external_used']} generated={stats['generated_used']}"
        )

    meta["elapsed_sec"] = time.time() - start
    meta_path = output_root / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Meta saved: {meta_path}")


if __name__ == "__main__":
    main()
