"""
SAT (CDCL) Sudoku Solver
Uses PySAT to encode Sudoku as CNF and solve with CDCL SAT solver
"""

import time
from typing import List, Dict, Optional, Tuple
import json
import sys
import os

try:
    from pysat.solvers import Solver
    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False


class SATSolver:
    def __init__(self, solver_name: str = 'glucose4', timeout_ms: float = 30000):
        self.solver_name = solver_name
        self.timeout_ms = timeout_ms
        self.decisions = 0
        self.conflicts = 0
        self.propagations = 0
        self.start_time = 0

    def solve(self, puzzle: str) -> Tuple[Optional[str], Dict]:
        if not HAS_PYSAT:
            return None, {'solved': False, 'error': 'PySAT not installed. Run: pip install python-sat'}

        self.decisions = 0
        self.conflicts = 0
        self.propagations = 0
        self.start_time = time.perf_counter()

        grid = self._parse_puzzle(puzzle)
        clauses = self._encode_sudoku(grid)
        solution = self._solve_cnf(clauses)

        elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if solution:
            solution_grid = self._decode_solution(solution)
            is_valid = self._validate_solution(solution_grid, puzzle)
            solution_str = ''.join(str(d) for d in solution_grid)
            return solution_str, {
                'solved': is_valid,
                'is_valid': is_valid,
                'solution': solution_str,
                'nodes': self.decisions,
                'backtracks': self.conflicts,
                'time_ms': round(elapsed_ms, 2),
                'decisions': self.decisions,
                'conflicts': self.conflicts,
                'propagations': self.propagations
            }

        return None, {
            'solved': False,
            'is_valid': False,
            'solution': None,
            'nodes': self.decisions,
            'backtracks': self.conflicts,
            'time_ms': round(elapsed_ms, 2),
            'decisions': self.decisions,
            'conflicts': self.conflicts,
            'propagations': self.propagations
        }

    def _parse_puzzle(self, puzzle: str) -> List[List[int]]:
        puzzle = puzzle.strip().replace('.', '0')
        grid = [[0] * 9 for _ in range(9)]
        for i, c in enumerate(puzzle[:81]):
            if c.isdigit():
                grid[i // 9][i % 9] = int(c)
        return grid

    def _var(self, r: int, c: int, d: int) -> int:
        return r * 81 + c * 9 + d

    def _encode_sudoku(self, grid: List[List[int]]) -> List[List[int]]:
        clauses = []

        for r in range(9):
            for c in range(9):
                clause = [self._var(r, c, d) for d in range(1, 10)]
                clauses.append(clause)

        for r in range(9):
            for c in range(9):
                for d1 in range(1, 10):
                    for d2 in range(d1 + 1, 10):
                        clauses.append([-self._var(r, c, d1), -self._var(r, c, d2)])

        for r in range(9):
            for d in range(1, 10):
                clause = [self._var(r, c, d) for c in range(9)]
                clauses.append(clause)
                for c1 in range(9):
                    for c2 in range(c1 + 1, 9):
                        clauses.append([-self._var(r, c1, d), -self._var(r, c2, d)])

        for c in range(9):
            for d in range(1, 10):
                clause = [self._var(r, c, d) for r in range(9)]
                clauses.append(clause)
                for r1 in range(9):
                    for r2 in range(r1 + 1, 9):
                        clauses.append([-self._var(r1, c, d), -self._var(r2, c, d)])

        for br in range(3):
            for bc in range(3):
                for d in range(1, 10):
                    clause = []
                    for r in range(br * 3, br * 3 + 3):
                        for c in range(bc * 3, bc * 3 + 3):
                            clause.append(self._var(r, c, d))
                    clauses.append(clause)

                    cells = [(r, c) for r in range(br * 3, br * 3 + 3) for c in range(bc * 3, bc * 3 + 3)]
                    for i in range(len(cells)):
                        for j in range(i + 1, len(cells)):
                            r1, c1 = cells[i]
                            r2, c2 = cells[j]
                            clauses.append([-self._var(r1, c1, d), -self._var(r2, c2, d)])

        for r in range(9):
            for c in range(9):
                if grid[r][c] != 0:
                    clauses.append([self._var(r, c, grid[r][c])])

        return clauses

    def _solve_cnf(self, clauses: List[List[int]]) -> Optional[List[int]]:
        try:
            with Solver(name=self.solver_name, bootstrap_with=clauses) as solver:
                result = solver.solve()
                if result:
                    stats = solver.accum_stats()
                    self.decisions = stats.get('decisions', 0)
                    self.conflicts = stats.get('conflicts', 0)
                    self.propagations = stats.get('propagations', 0)
                    return solver.get_model()
                return None
        except Exception as e:
            print(f"SAT solver error: {e}")
            return None

    def _decode_solution(self, model: List[int]) -> List[int]:
        solution = [0] * 81
        for var in model:
            if var > 0:
                var -= 1
                d = var % 9 + 1
                var //= 9
                c = var % 9
                r = var // 9
                if r < 9 and c < 9:
                    solution[r * 9 + c] = d
        return solution

    def _validate_solution(self, solution: List[int], puzzle: str) -> bool:
        if not solution or 0 in solution:
            return False

        puzzle = puzzle.strip().replace('.', '0')
        for i, c in enumerate(puzzle[:81]):
            if c.isdigit() and int(c) != 0:
                if solution[i] != int(c):
                    return False

        for i in range(9):
            if set(solution[i*9:(i+1)*9]) != set(range(1, 10)):
                return False
            if set(solution[i::9]) != set(range(1, 10)):
                return False

        for br in range(3):
            for bc in range(3):
                box = set()
                for r in range(br*3, br*3+3):
                    for c in range(bc*3, bc*3+3):
                        box.add(solution[r*9 + c])
                if box != set(range(1, 10)):
                    return False
        return True


if __name__ == '__main__':
    puzzle = sys.argv[1] if len(sys.argv) > 1 else "000000000000000001000002030000003020001040000005000060030000004070080009620007000"
    solver = SATSolver()
    solution, metrics = solver.solve(puzzle)
    print(f"Solution: {solution}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
