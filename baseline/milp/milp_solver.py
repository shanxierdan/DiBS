"""
MILP (Mixed Integer Linear Programming) Sudoku Solver
Uses PuLP with CBC solver to encode Sudoku as 0-1 ILP
"""

import time
from typing import List, Dict, Optional, Tuple
import sys

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class MILPSolver:
    def __init__(self, timeout_ms: float = 30000):
        self.timeout_ms = timeout_ms
        self.bb_nodes = 0
        self.lp_iters = 0
        self.start_time = 0

    def solve(self, puzzle: str) -> Tuple[Optional[str], Dict]:
        if not HAS_PULP:
            return None, {'solved': False, 'error': 'PuLP not installed. Run: pip install pulp'}

        self.bb_nodes = 0
        self.lp_iters = 0
        self.start_time = time.perf_counter()

        grid = self._parse_puzzle(puzzle)
        solution = self._solve_milp(grid)
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if solution:
            is_valid = self._validate_solution(solution, puzzle)
            solution_str = ''.join(str(d) for d in solution)
            return solution_str, {
                'solved': is_valid,
                'is_valid': is_valid,
                'solution': solution_str,
                'nodes': self.bb_nodes,
                'backtracks': self.bb_nodes,
                'time_ms': round(elapsed_ms, 2),
                'bb_nodes': self.bb_nodes,
                'lp_iters': self.lp_iters
            }

        return None, {
            'solved': False,
            'is_valid': False,
            'solution': None,
            'nodes': self.bb_nodes,
            'backtracks': self.bb_nodes,
            'time_ms': round(elapsed_ms, 2),
            'bb_nodes': self.bb_nodes,
            'lp_iters': self.lp_iters
        }

    def _parse_puzzle(self, puzzle: str) -> List[List[int]]:
        puzzle = puzzle.strip().replace('.', '0')
        grid = [[0] * 9 for _ in range(9)]
        for i, c in enumerate(puzzle[:81]):
            if c.isdigit():
                grid[i // 9][i % 9] = int(c)
        return grid

    def _solve_milp(self, grid: List[List[int]]) -> Optional[List[int]]:
        try:
            prob = pulp.LpProblem("Sudoku", pulp.LpMaximize)

            x = {}
            for r in range(9):
                for c in range(9):
                    for d in range(1, 10):
                        x[r, c, d] = pulp.LpVariable(f"x_{r}_{c}_{d}", cat='Binary')

            prob += 0

            for r in range(9):
                for c in range(9):
                    prob += pulp.lpSum(x[r, c, d] for d in range(1, 10)) == 1

            for r in range(9):
                for d in range(1, 10):
                    prob += pulp.lpSum(x[r, c, d] for c in range(9)) == 1

            for c in range(9):
                for d in range(1, 10):
                    prob += pulp.lpSum(x[r, c, d] for r in range(9)) == 1

            for br in range(3):
                for bc in range(3):
                    for d in range(1, 10):
                        prob += pulp.lpSum(
                            x[r, c, d]
                            for r in range(br * 3, br * 3 + 3)
                            for c in range(bc * 3, bc * 3 + 3)
                        ) == 1

            for r in range(9):
                for c in range(9):
                    if grid[r][c] != 0:
                        prob += x[r, c, grid[r][c]] == 1

            solver = pulp.PULP_CBC_CMD(timeLimit=self.timeout_ms / 1000, msg=0)
            status = prob.solve(solver)

            if status == pulp.LpStatusOptimal:
                solution = [0] * 81
                for r in range(9):
                    for c in range(9):
                        for d in range(1, 10):
                            if pulp.value(x[r, c, d]) > 0.5:
                                solution[r * 9 + c] = d
                                break
                return solution
            return None
        except Exception as e:
            print(f"MILP solver error: {e}")
            return None

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
    import json
    puzzle = sys.argv[1] if len(sys.argv) > 1 else "000000000000000001000002030000003020001040000005000060030000004070080009620007000"
    solver = MILPSolver()
    solution, metrics = solver.solve(puzzle)
    print(f"Solution: {solution}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
