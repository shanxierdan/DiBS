import numpy as np
from typing import Tuple, List, Dict, Optional
from copy import deepcopy


class SudokuConstraints:
    DIGIT_BITS = {
        1: 1 << 0, 2: 1 << 1, 3: 1 << 2, 4: 1 << 3, 5: 1 << 4,
        6: 1 << 5, 7: 1 << 6, 8: 1 << 7, 9: 1 << 8
    }
    ALL_DIGITS = 0x1FF
    DIGIT_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self):
        self.row_used: np.ndarray = np.zeros(9, dtype=np.int32)
        self.col_used: np.ndarray = np.zeros(9, dtype=np.int32)
        self.box_used: np.ndarray = np.zeros(9, dtype=np.int32)
        self.candidates: np.ndarray = np.full(81, self.ALL_DIGITS, dtype=np.int32)
        self.grid: np.ndarray = np.zeros(81, dtype=np.int32)
        self.filled_count: int = 0

    def _cell_to_box(self, cell: int) -> int:
        row, col = cell // 9, cell % 9
        return (row // 3) * 3 + (col // 3)

    def _cell_to_row_col_box(self, cell: int) -> Tuple[int, int, int]:
        row, col = cell // 9, cell % 9
        box = (row // 3) * 3 + (col // 3)
        return row, col, box

    def initialize(self, grid: np.ndarray) -> bool:
        if isinstance(grid, str):
            grid = np.array([int(c) if c.isdigit() else 0 for c in grid])

        grid = np.asarray(grid, dtype=np.int32)
        if grid.shape == (9, 9):
            grid = grid.flatten()

        if grid.shape != (81,):
            raise ValueError(f"Grid must have 81 elements, got {grid.shape}")

        self.row_used = np.zeros(9, dtype=np.int32)
        self.col_used = np.zeros(9, dtype=np.int32)
        self.box_used = np.zeros(9, dtype=np.int32)
        self.candidates = np.full(81, self.ALL_DIGITS, dtype=np.int32)
        self.grid = grid.copy()
        self.filled_count = 0

        for cell in range(81):
            digit = grid[cell]
            if digit != 0:
                if not self._assign_initial(cell, digit):
                    return False

        return True

    def _assign_initial(self, cell: int, digit: int) -> bool:
        row, col, box = self._cell_to_row_col_box(cell)
        bit = self.DIGIT_BITS[digit]

        if self.row_used[row] & bit:
            return False
        if self.col_used[col] & bit:
            return False
        if self.box_used[box] & bit:
            return False

        self.row_used[row] |= bit
        self.col_used[col] |= bit
        self.box_used[box] |= bit
        self.candidates[cell] = 0
        self.filled_count += 1

        self._eliminate_from_peers(cell, digit)

        return True

    def _eliminate_from_peers(self, cell: int, digit: int):
        row, col, box = self._cell_to_row_col_box(cell)
        bit = self.DIGIT_BITS[digit]

        for c in range(row * 9, row * 9 + 9):
            self.candidates[c] &= ~bit

        for r in range(9):
            self.candidates[r * 9 + col] &= ~bit

        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                self.candidates[r * 9 + c] &= ~bit

    def propagate(self) -> Tuple[bool, bool]:
        changed = True
        total_changed = False
        iteration_count = 0

        while changed:
            changed = False
            iteration_count += 1

            for cell in range(81):
                if self.grid[cell] != 0:
                    continue

                cand = self.candidates[cell]
                if cand == 0:
                    return total_changed, False

                if cand & (cand - 1) == 0:
                    digit = self._bit_to_digit(cand)
                    if digit is None:
                        return total_changed, False

                    if not self.assign(cell, digit):
                        return total_changed, False

                    changed = True
                    total_changed = True

        return total_changed, True

    def _bit_to_digit(self, bit: int) -> Optional[int]:
        for digit, b in self.DIGIT_BITS.items():
            if bit == b:
                return digit
        return None

    def assign(self, cell: int, digit: int) -> bool:
        if self.grid[cell] != 0:
            return self.grid[cell] == digit

        row, col, box = self._cell_to_row_col_box(cell)
        bit = self.DIGIT_BITS[digit]

        if not (self.candidates[cell] & bit):
            return False

        self.row_used[row] |= bit
        self.col_used[col] |= bit
        self.box_used[box] |= bit
        self.candidates[cell] = 0
        self.grid[cell] = digit
        self.filled_count += 1

        self._eliminate_from_peers(cell, digit)

        return True

    def get_mrv_cells(self) -> List[int]:
        min_candidates = 10
        mrv_cells = []

        for cell in range(81):
            if self.grid[cell] != 0:
                continue

            cand = self.candidates[cell]
            count = bin(cand).count('1')

            if count < min_candidates:
                min_candidates = count
                mrv_cells = [cell]
            elif count == min_candidates:
                mrv_cells.append(cell)

        return mrv_cells

    def get_candidates(self, cell: int) -> List[int]:
        cand = self.candidates[cell]
        return [d for d in self.DIGIT_LIST if cand & self.DIGIT_BITS[d]]

    def get_candidates_bitmask(self, cell: int) -> int:
        return self.candidates[cell]

    def count_candidates(self, cell: int) -> int:
        return bin(self.candidates[cell]).count('1')

    def is_complete(self) -> bool:
        return self.filled_count == 81

    def is_valid(self) -> bool:
        for cell in range(81):
            if self.grid[cell] == 0 and self.candidates[cell] == 0:
                return False
        return True

    def save_state(self) -> Dict:
        return {
            'row_used': self.row_used.copy(),
            'col_used': self.col_used.copy(),
            'box_used': self.box_used.copy(),
            'candidates': self.candidates.copy(),
            'grid': self.grid.copy(),
            'filled_count': self.filled_count
        }

    def restore_state(self, state: Dict):
        self.row_used = state['row_used'].copy()
        self.col_used = state['col_used'].copy()
        self.box_used = state['box_used'].copy()
        self.candidates = state['candidates'].copy()
        self.grid = state['grid'].copy()
        self.filled_count = state['filled_count']

    def get_neighbors(self, cell: int) -> List[int]:
        row, col, box = self._cell_to_row_col_box(cell)
        neighbors = set()

        for c in range(row * 9, row * 9 + 9):
            if c != cell and self.grid[c] == 0:
                neighbors.add(c)

        for r in range(9):
            idx = r * 9 + col
            if idx != cell and self.grid[idx] == 0:
                neighbors.add(idx)

        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                idx = r * 9 + c
                if idx != cell and self.grid[idx] == 0:
                    neighbors.add(idx)

        return list(neighbors)

    def compute_lcv(self, cell: int, digit: int) -> int:
        neighbors = self.get_neighbors(cell)
        bit = self.DIGIT_BITS[digit]

        lcv_score = 0
        for neighbor in neighbors:
            if self.candidates[neighbor] & bit:
                lcv_score += 1

        return lcv_score

    def get_solution_string(self) -> str:
        return "".join(str(int(d) if d > 0 else 0) for d in self.grid)

    def validate_solution(self) -> bool:
        if not self.is_complete():
            return False

        for i in range(9):
            row_digits = set(self.grid[i*9:(i+1)*9])
            if row_digits != set(range(1, 10)):
                return False

            col_digits = set(self.grid[i::9])
            if col_digits != set(range(1, 10)):
                return False

        for box_row in range(3):
            for box_col in range(3):
                box_digits = set()
                for r in range(box_row * 3, box_row * 3 + 3):
                    for c in range(box_col * 3, box_col * 3 + 3):
                        box_digits.add(self.grid[r * 9 + c])
                if box_digits != set(range(1, 10)):
                    return False

        return True
