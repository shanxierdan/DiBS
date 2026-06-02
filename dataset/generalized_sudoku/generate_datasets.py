#!/usr/bin/env python3
"""
Generalized Sudoku Dataset Generator
Generates 4x4, 16x16, and 25x25 Sudoku puzzles for Table 4 experiments
"""

import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import copy


class GeneralizedSudokuGenerator:
    def __init__(self, size: int):
        self.size = size
        if size == 4:
            self.box_rows, self.box_cols = 2, 2
            self.symbols = '1234'
        elif size == 16:
            self.box_rows, self.box_cols = 4, 4
            self.symbols = '123456789ABCDEFG'
        elif size == 25:
            self.box_rows, self.box_cols = 5, 5
            self.symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXY'
        else:
            raise ValueError(f"Unsupported size: {size}")

    def is_valid(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        for i in range(self.size):
            if grid[row][i] == num or grid[i][col] == num:
                return False

        box_row_start = (row // self.box_rows) * self.box_rows
        box_col_start = (col // self.box_cols) * self.box_cols

        for i in range(box_row_start, box_row_start + self.box_rows):
            for j in range(box_col_start, box_col_start + self.box_cols):
                if grid[i][j] == num:
                    return False
        return True

    def solve(self, grid: List[List[int]]) -> bool:
        for row in range(self.size):
            for col in range(self.size):
                if grid[row][col] == 0:
                    for num in range(1, self.size + 1):
                        if self.is_valid(grid, row, col, num):
                            grid[row][col] = num
                            if self.solve(grid):
                                return True
                            grid[row][col] = 0
                    return False
        return True

    def count_solutions(self, grid: List[List[int]], limit: int = 2) -> int:
        count = [0]

        def backtrack():
            if count[0] >= limit:
                return
            for row in range(self.size):
                for col in range(self.size):
                    if grid[row][col] == 0:
                        for num in range(1, self.size + 1):
                            if self.is_valid(grid, row, col, num):
                                grid[row][col] = num
                                backtrack()
                                grid[row][col] = 0
                                if count[0] >= limit:
                                    return
                        return
            count[0] += 1

        backtrack()
        return count[0]

    def generate_full_grid(self) -> List[List[int]]:
        grid = [[0] * self.size for _ in range(self.size)]

        def fill(pos: int = 0) -> bool:
            if pos == self.size * self.size:
                return True
            row, col = pos // self.size, pos % self.size
            nums = list(range(1, self.size + 1))
            random.shuffle(nums)
            for num in nums:
                if self.is_valid(grid, row, col, num):
                    grid[row][col] = num
                    if fill(pos + 1):
                        return True
                    grid[row][col] = 0
            return False

        fill()
        return grid

    def create_puzzle(self, grid: List[List[int]], difficulty: str = "medium") -> Tuple[List[List[int]], int]:
        puzzle = copy.deepcopy(grid)

        if self.size == 4:
            givens_range = {"easy": (10, 12), "medium": (8, 10), "hard": (6, 8)}
        elif self.size == 16:
            givens_range = {"easy": (140, 160), "medium": (100, 120), "hard": (80, 100)}
        else:
            givens_range = {"easy": (350, 400), "medium": (280, 320), "hard": (220, 260)}

        min_givens, max_givens = givens_range.get(difficulty, givens_range["medium"])
        target_givens = random.randint(min_givens, max_givens)

        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(positions)

        current_givens = self.size * self.size

        for row, col in positions:
            if current_givens <= target_givens:
                break

            backup = puzzle[row][col]
            puzzle[row][col] = 0

            test_grid = copy.deepcopy(puzzle)
            if self.count_solutions(test_grid, 2) != 1:
                puzzle[row][col] = backup
            else:
                current_givens -= 1

        return puzzle, current_givens

    def grid_to_string(self, grid: List[List[int]]) -> str:
        result = []
        for row in grid:
            for val in row:
                if val == 0:
                    result.append('0')
                elif self.size <= 9:
                    result.append(str(val))
                else:
                    result.append(self.symbols[val - 1])
        return ''.join(result)

    def generate_puzzle(self, difficulty: str = "medium") -> Tuple[str, str, int]:
        full_grid = self.generate_full_grid()
        puzzle_grid, givens = self.create_puzzle(full_grid, difficulty)

        puzzle_str = self.grid_to_string(puzzle_grid)
        solution_str = self.grid_to_string(full_grid)

        return puzzle_str, solution_str, givens


def generate_dataset(size: int, num_puzzles: int, difficulty: str, output_dir: Path):
    generator = GeneralizedSudokuGenerator(size)

    puzzles = []
    solutions = []
    givens_list = []

    print(f"Generating {num_puzzles} {size}x{size} puzzles ({difficulty})...")

    for i in range(num_puzzles):
        puzzle, solution, givens = generator.generate_puzzle(difficulty)
        puzzles.append(puzzle)
        solutions.append(solution)
        givens_list.append(givens)

        if (i + 1) % 10 == 0 or (i + 1) == num_puzzles:
            avg_givens = sum(givens_list) / len(givens_list)
            print(f"  Progress: {i+1}/{num_puzzles} | Avg givens: {avg_givens:.1f}")

    size_name = f"{size}x{size}"
    puzzle_file = output_dir / f"sudoku_{size_name}_puzzles.txt"
    solution_file = output_dir / f"sudoku_{size_name}_solutions.txt"

    with open(puzzle_file, 'w') as f:
        for p in puzzles:
            f.write(p + '\n')

    with open(solution_file, 'w') as f:
        for s in solutions:
            f.write(s + '\n')

    print(f"Saved {num_puzzles} puzzles to {puzzle_file}")
    print(f"Saved {num_puzzles} solutions to {solution_file}")

    return puzzles, solutions, givens_list


def main():
    parser = argparse.ArgumentParser(description="Generate generalized Sudoku datasets")
    parser.add_argument("--output", type=str, default="dataset/generalized_sudoku")
    parser.add_argument("--num-4x4", type=int, default=1000, help="Number of 4x4 puzzles")
    parser.add_argument("--num-16x16", type=int, default=1000, help="Number of 16x16 puzzles")
    parser.add_argument("--num-25x25", type=int, default=500, help="Number of 25x25 puzzles")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Generalized Sudoku Dataset Generator")
    print("="*60)

    if args.num_4x4 > 0:
        print(f"\n{'='*60}")
        print("4x4 Sudoku (Mini)")
        print("="*60)
        generate_dataset(4, args.num_4x4, args.difficulty, output_dir)

    if args.num_16x16 > 0:
        print(f"\n{'='*60}")
        print("16x16 Sudoku (Hexadoku)")
        print("="*60)
        generate_dataset(16, args.num_16x16, args.difficulty, output_dir)

    if args.num_25x25 > 0:
        print(f"\n{'='*60}")
        print("25x25 Sudoku (Sudoku Giant)")
        print("="*60)
        generate_dataset(25, args.num_25x25, args.difficulty, output_dir)

    print(f"\n{'='*60}")
    print("Dataset generation complete!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
