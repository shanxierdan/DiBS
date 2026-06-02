"""
DLX (Algorithm X + Dancing Links) Sudoku Solver
Based on Knuth's Algorithm X with Dancing Links data structure
"""

import time
from typing import List, Dict, Optional, Tuple


class DLXNode:
    __slots__ = ['left', 'right', 'up', 'down', 'column', 'row_data']

    def __init__(self):
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column = None
        self.row_data = None


class ColumnNode(DLXNode):
    __slots__ = ['size', 'name']

    def __init__(self, name: str = ""):
        super().__init__()
        self.size = 0
        self.name = name


class DLXSolver:
    def __init__(self, timeout_ms: float = 30000):
        self.timeout_ms = timeout_ms
        self.header = None
        self.columns = []
        self.solution = []
        self.nodes = 0
        self.backtracks = 0
        self.start_time = 0
        self.timeout = False

    def solve(self, puzzle: str) -> Tuple[Optional[str], Dict]:
        self.nodes = 0
        self.backtracks = 0
        self.timeout = False
        self.solution = []
        self.start_time = time.perf_counter()

        grid = self._parse_puzzle(puzzle)

        try:
            self._build_dlx(grid)
        except Exception as e:
            return None, self._get_metrics(None, False, str(e))

        if self.header is None:
            return None, self._get_metrics(None, False)

        success = self._search()

        if success:
            solution_grid = self._extract_solution(grid)
            is_valid = self._validate_solution(solution_grid, puzzle)
            solution_str = ''.join(str(d) for d in solution_grid)
            return solution_str, self._get_metrics(solution_str, is_valid)

        return None, self._get_metrics(None, False)

    def _parse_puzzle(self, puzzle: str) -> List[List[int]]:
        puzzle = puzzle.strip().replace('.', '0')
        grid = [[0] * 9 for _ in range(9)]
        for i, c in enumerate(puzzle[:81]):
            if c.isdigit():
                grid[i // 9][i % 9] = int(c)
        return grid

    def _build_dlx(self, grid: List[List[int]]):
        self.header = ColumnNode("header")
        self.header.left = self.header
        self.header.right = self.header
        self.columns = []

        for i in range(324):
            col = ColumnNode(f"col_{i}")
            col.left = self.header.left
            col.right = self.header
            self.header.left.right = col
            self.header.left = col
            self.columns.append(col)

        for r in range(9):
            for c in range(9):
                for d in range(1, 10):
                    if grid[r][c] != 0 and grid[r][c] != d:
                        continue

                    col_indices = [
                        r * 9 + c,
                        81 + r * 9 + (d - 1),
                        162 + c * 9 + (d - 1),
                        243 + (r // 3 * 3 + c // 3) * 9 + (d - 1)
                    ]

                    self._add_row((r, c, d), col_indices)

        for r in range(9):
            for c in range(9):
                if grid[r][c] != 0:
                    d = grid[r][c]
                    col_indices = [
                        r * 9 + c,
                        81 + r * 9 + (d - 1),
                        162 + c * 9 + (d - 1),
                        243 + (r // 3 * 3 + c // 3) * 9 + (d - 1)
                    ]
                    for col_idx in col_indices:
                        self._cover(self.columns[col_idx])

    def _add_row(self, row_data: Tuple[int, int, int], col_indices: List[int]):
        first = None
        prev = None

        for col_idx in col_indices:
            col = self.columns[col_idx]
            node = DLXNode()
            node.column = col
            node.row_data = row_data
            node.up = col.up
            node.down = col
            col.up.down = node
            col.up = node
            col.size += 1

            if first is None:
                first = node
                node.left = node
                node.right = node
            else:
                node.left = prev
                node.right = first
                prev.right = node
                first.left = node

            prev = node

    def _cover(self, col):
        col.right.left = col.left
        col.left.right = col.right

        row = col.down
        while row is not col:
            node = row.right
            while node is not row:
                node.down.up = node.up
                node.up.down = node.down
                node.column.size -= 1
                node = node.right
            row = row.down

    def _uncover(self, col):
        row = col.up
        while row is not col:
            node = row.left
            while node is not row:
                node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row = row.up

        col.right.left = col
        col.left.right = col

    def _choose_column(self) -> Optional[ColumnNode]:
        min_size = float('inf')
        chosen = None

        col = self.header.right
        while col is not self.header:
            if col.size < min_size:
                min_size = col.size
                chosen = col
            col = col.right

        return chosen

    def _search(self) -> bool:
        if time.perf_counter() - self.start_time > self.timeout_ms / 1000:
            self.timeout = True
            return False

        if self.header.right is self.header:
            return True

        col = self._choose_column()
        if col is None or col.size == 0:
            return False

        self._cover(col)

        row = col.down
        while row is not col:
            self.nodes += 1
            self.solution.append(row)

            node = row.right
            while node is not row:
                self._cover(node.column)
                node = node.right

            if self._search():
                return True

            self.solution.pop()
            self.backtracks += 1

            node = row.left
            while node is not row:
                self._uncover(node.column)
                node = node.left

            row = row.down

        self._uncover(col)
        return False

    def _extract_solution(self, original_grid: List[List[int]]) -> List[int]:
        result = [original_grid[r][c] for r in range(9) for c in range(9)]

        for node in self.solution:
            r, c, d = node.row_data
            result[r * 9 + c] = d

        return result

    def _validate_solution(self, solution: List[int], puzzle: str) -> bool:
        if not solution:
            return False

        puzzle = puzzle.strip().replace('.', '0')
        for i, c in enumerate(puzzle[:81]):
            if c.isdigit() and int(c) != 0:
                if solution[i] != int(c):
                    return False

        for i in range(9):
            row = set(solution[i*9:(i+1)*9])
            if row != set(range(1, 10)):
                return False

            col = set(solution[i::9])
            if col != set(range(1, 10)):
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

    def _get_metrics(self, solution: Optional[str], is_valid: bool, error: str = "") -> Dict:
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        return {
            'solved': solution is not None and is_valid,
            'is_valid': is_valid,
            'solution': solution,
            'nodes': self.nodes,
            'backtracks': self.backtracks,
            'time_ms': round(elapsed_ms, 2),
            'timeout': self.timeout,
            'error': error
        }


if __name__ == '__main__':
    import sys
    import json

    puzzle = sys.argv[1] if len(sys.argv) > 1 else "000000000000000001000002030000003020001040000005000060030000004070080009620007000"
    solver = DLXSolver(timeout_ms=5000)
    solution, metrics = solver.solve(puzzle)
    print(f"Solution: {solution}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
