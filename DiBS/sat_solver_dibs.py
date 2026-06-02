from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class SATMetrics:
    solved: bool
    valid: bool
    status: str
    time_ms: float
    nodes: int
    backtracks: int
    model_calls: int
    model_time_ms: float


LiteralScorer = Callable[[int, Sequence[int], Sequence[Sequence[int]]], Tuple[float, float]]
LiteralRanker = Callable[[Sequence[int], Sequence[int], Sequence[Sequence[int]]], Dict[int, float]]


def _lit_true(lit: int, assignment: Sequence[int]) -> Optional[bool]:
    v = abs(lit)
    val = assignment[v]
    if val == 0:
        return None
    return (lit > 0 and val > 0) or (lit < 0 and val < 0)


class DiBSSATSolver:
    def __init__(
        self,
        max_nodes: int = 1_000_000,
        timeout_ms: int = 0,
        mrv_threshold: int = 2,
        smart_call: bool = True,
        literal_scorer: Optional[LiteralScorer] = None,
        literal_ranker: Optional[LiteralRanker] = None,
        var_heuristic: str = "mrv",
        value_heuristic: str = "jw",
        jw_margin_threshold: float = 0.25,
    ) -> None:
        self.max_nodes = int(max_nodes)
        self.timeout_ms = int(timeout_ms)
        self.mrv_threshold = int(mrv_threshold)
        self.smart_call = bool(smart_call)
        self.literal_scorer = literal_scorer
        self.literal_ranker = literal_ranker
        self.var_heuristic = str(var_heuristic)
        self.value_heuristic = str(value_heuristic)
        self.jw_margin_threshold = float(jw_margin_threshold)
        self._reset()

    def _reset(self) -> None:
        self.nodes = 0
        self.backtracks = 0
        self.model_calls = 0
        self.model_time_ms = 0.0
        self.start = 0.0
        self.deadline = 0.0

    def solve(self, num_vars: int, clauses: Sequence[Sequence[int]]) -> Tuple[Optional[List[int]], SATMetrics]:
        self._reset()
        self.start = time.perf_counter()
        self.deadline = self.start + (self.timeout_ms / 1000.0) if self.timeout_ms > 0 else 0.0
        assignment = [0] * (num_vars + 1)
        solved, status = self._dfs(clauses, assignment)
        elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        sol = assignment[:] if solved else None
        metrics = SATMetrics(
            solved=solved,
            valid=solved,
            status=status,
            time_ms=elapsed_ms,
            nodes=self.nodes,
            backtracks=self.backtracks,
            model_calls=self.model_calls,
            model_time_ms=self.model_time_ms,
        )
        return sol, metrics

    def _dfs(self, clauses: Sequence[Sequence[int]], assignment: List[int]) -> Tuple[bool, str]:
        if self.timeout_ms > 0 and time.perf_counter() > self.deadline:
            return False, "timeout"
        if self.nodes >= self.max_nodes:
            return False, "max_nodes"
        self.nodes += 1

        ok, changed = self._unit_propagate(clauses, assignment)
        if not ok:
            self.backtracks += 1
            return False, "conflict"
        while changed:
            ok, changed = self._unit_propagate(clauses, assignment)
            if not ok:
                self.backtracks += 1
                return False, "conflict"

        if all(v != 0 for v in assignment[1:]):
            return True, "solved"

        if self.value_heuristic == "literal_model":
            branch_literals = self._literal_branch_order(clauses, assignment)
            if not branch_literals:
                return True, "solved"
            snapshot = assignment[:]
            for lit in branch_literals:
                assignment[abs(lit)] = 1 if lit > 0 else -1
                solved, status = self._dfs(clauses, assignment)
                if solved:
                    return True, status
                assignment[:] = snapshot
            self.backtracks += 1
            return False, "backtrack"

        if self.var_heuristic == "first":
            var, min_open = self._select_var_first(clauses, assignment)
        else:
            var, min_open = self._select_var_mrv(clauses, assignment)
        if var == 0:
            return True, "solved"
        order = self._value_order(var, assignment, clauses, min_open)

        snapshot = assignment[:]
        for val in order:
            assignment[var] = val
            solved, status = self._dfs(clauses, assignment)
            if solved:
                return True, status
            assignment[:] = snapshot
        self.backtracks += 1
        return False, "backtrack"

    def _unit_propagate(self, clauses: Sequence[Sequence[int]], assignment: List[int]) -> Tuple[bool, bool]:
        changed = False
        for clause in clauses:
            any_true = False
            unassigned = []
            for lit in clause:
                s = _lit_true(lit, assignment)
                if s is True:
                    any_true = True
                    break
                if s is None:
                    unassigned.append(lit)
            if any_true:
                continue
            if not unassigned:
                return False, changed
            if len(unassigned) == 1:
                lit = unassigned[0]
                v = abs(lit)
                val = 1 if lit > 0 else -1
                if assignment[v] == 0:
                    assignment[v] = val
                    changed = True
                elif assignment[v] != val:
                    return False, changed
        return True, changed

    def _select_var_mrv(self, clauses: Sequence[Sequence[int]], assignment: Sequence[int]) -> Tuple[int, int]:
        best_var = 0
        best_open = 10**9
        for clause in clauses:
            open_lits = []
            clause_satisfied = False
            for lit in clause:
                s = _lit_true(lit, assignment)
                if s is True:
                    clause_satisfied = True
                    break
                if s is None:
                    open_lits.append(lit)
            if clause_satisfied or not open_lits:
                continue
            if len(open_lits) < best_open:
                best_open = len(open_lits)
                for lit in open_lits:
                    v = abs(lit)
                    if assignment[v] == 0:
                        best_var = v
                        break
                if best_open == 1:
                    break
        if best_var == 0:
            for v in range(1, len(assignment)):
                if assignment[v] == 0:
                    return v, best_open if best_open < 10**9 else 999
        return best_var, best_open if best_open < 10**9 else 999

    def _value_order(
        self,
        var: int,
        assignment: Sequence[int],
        clauses: Sequence[Sequence[int]],
        min_open_clause: int,
    ) -> List[int]:
        use_model = self.literal_scorer is not None and (not self.smart_call or min_open_clause <= self.mrv_threshold)
        if use_model:
            t0 = time.perf_counter()
            score_true, score_false = self.literal_scorer(var, assignment, clauses)
            self.model_time_ms += (time.perf_counter() - t0) * 1000.0
            self.model_calls += 1
            return [1, -1] if score_true >= score_false else [-1, 1]
        if self.value_heuristic == "pos":
            return [1, -1]
        if self.value_heuristic == "neg":
            return [-1, 1]
        score_true, score_false = self._jw_scores(var, clauses, assignment)
        return [1, -1] if score_true >= score_false else [-1, 1]

    @staticmethod
    def _select_var_first(_clauses: Sequence[Sequence[int]], assignment: Sequence[int]) -> Tuple[int, int]:
        for v in range(1, len(assignment)):
            if assignment[v] == 0:
                return v, 999
        return 0, 999

    def _literal_branch_order(self, clauses: Sequence[Sequence[int]], assignment: Sequence[int]) -> List[int]:
        candidates, min_open = self._select_mrv_clause_literals(clauses, assignment)
        if not candidates:
            return []

        jw_scores = {lit: self._jw_literal_score(lit, clauses, assignment) for lit in candidates}
        use_model = self.literal_ranker is not None and (not self.smart_call or min_open <= self.mrv_threshold)
        if use_model and self.smart_call:
            sorted_scores = sorted(jw_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                margin = sorted_scores[0] - sorted_scores[1]
                use_model = margin <= self.jw_margin_threshold
            else:
                use_model = False

        if use_model and self.literal_ranker is not None:
            t0 = time.perf_counter()
            model_scores = self.literal_ranker(candidates, assignment, clauses)
            self.model_time_ms += (time.perf_counter() - t0) * 1000.0
            self.model_calls += 1
            return sorted(candidates, key=lambda lit: (model_scores.get(lit, float("-inf")), jw_scores.get(lit, 0.0)), reverse=True)

        return sorted(candidates, key=lambda lit: jw_scores.get(lit, 0.0), reverse=True)

    @staticmethod
    def _select_mrv_clause_literals(clauses: Sequence[Sequence[int]], assignment: Sequence[int]) -> Tuple[List[int], int]:
        best: List[int] = []
        best_open = 10**9
        for clause in clauses:
            open_lits: List[int] = []
            satisfied = False
            for lit in clause:
                s = _lit_true(lit, assignment)
                if s is True:
                    satisfied = True
                    break
                if s is None:
                    open_lits.append(lit)
            if satisfied or not open_lits:
                continue
            if len(open_lits) < best_open:
                best_open = len(open_lits)
                best = open_lits
                if best_open == 1:
                    break
        return best, best_open if best_open < 10**9 else 999

    @staticmethod
    def _jw_literal_score(target_lit: int, clauses: Sequence[Sequence[int]], assignment: Sequence[int]) -> float:
        score = 0.0
        for clause in clauses:
            sat = False
            k = 0
            contains = False
            for lit in clause:
                s = _lit_true(lit, assignment)
                if s is True:
                    sat = True
                    break
                if s is None:
                    k += 1
                    if lit == target_lit:
                        contains = True
            if not sat and k > 0 and contains:
                score += 2.0 ** (-k)
        return score

    @staticmethod
    def _jw_scores(var: int, clauses: Sequence[Sequence[int]], assignment: Sequence[int]) -> Tuple[float, float]:
        st, sf = 0.0, 0.0
        for clause in clauses:
            sat = False
            k = 0
            contains_pos = False
            contains_neg = False
            for lit in clause:
                s = _lit_true(lit, assignment)
                if s is True:
                    sat = True
                    break
                if s is None:
                    k += 1
                    if lit == var:
                        contains_pos = True
                    elif lit == -var:
                        contains_neg = True
            if sat or k == 0:
                continue
            w = 2.0 ** (-k)
            if contains_pos:
                st += w
            if contains_neg:
                sf += w
        return st, sf
