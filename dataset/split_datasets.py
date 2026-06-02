#!/usr/bin/env python3
"""
Dataset Split Script for Table 4 and Table 5

Generates train/test splits for:
- Table 4: Generalized Sudoku (4x4, 16x16, 25x25)
- Table 5: N-Queens (8x8, 9x9, 10x10)

Usage:
    python3 dataset/split_datasets.py --table 4 --train 5000 --test 500
    python3 dataset/split_datasets.py --table 5 --train 5000 --test 500
"""

import os
import random
import argparse
from typing import Tuple, List
import multiprocessing as mp


# ============================================================================
# Generalized Sudoku Generator (Table 4)
# ============================================================================

def generate_sudoku_full(grid_size: int, box_size: int) -> List[List[int]]:
    """Generate a complete valid Sudoku grid."""
    grid = [[0] * grid_size for _ in range(grid_size)]

    def is_valid(grid: List[List[int]], row: int, col: int, num: int) -> bool:
        for i in range(grid_size):
            if grid[row][i] == num or grid[i][col] == num:
                return False

        box_row, box_col = box_size * (row // box_size), box_size * (col // box_size)
        for i in range(box_row, box_row + box_size):
            for j in range(box_col, box_col + box_size):
                if grid[i][j] == num:
                    return False
        return True

    def solve(grid: List[List[int]], pos: int = 0) -> bool:
        if pos == grid_size * grid_size:
            return True

        row, col = pos // grid_size, pos % grid_size
        if grid[row][col] != 0:
            return solve(grid, pos + 1)

        nums = list(range(1, grid_size + 1))
        random.shuffle(nums)

        for num in nums:
            if is_valid(grid, row, col, num):
                grid[row][col] = num
                if solve(grid, pos + 1):
                    return True
                grid[row][col] = 0
        return False

    solve(grid)
    return grid


def create_sudoku_puzzle(full_grid: List[List[int]], grid_size: int, givens: int) -> List[List[int]]:
    """Create puzzle by removing cells from full grid."""
    puzzle = [row[:] for row in full_grid]
    cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(cells)

    to_remove = grid_size * grid_size - givens
    for i in range(min(to_remove, len(cells))):
        r, c = cells[i]
        puzzle[r][c] = 0

    return puzzle


def grid_to_string(grid: List[List[int]], grid_size: int) -> str:
    """Convert grid to string format."""
    if grid_size <= 9:
        return ''.join(str(cell) if cell > 0 else '0' for row in grid for cell in row)
    else:
        symbols = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return ''.join(symbols[cell] if cell > 0 else '0' for row in grid for cell in row)


def generate_sudoku_batch(args: Tuple[int, int, int, int]) -> List[Tuple[str, str]]:
    """Generate a batch of Sudoku puzzles."""
    grid_size, box_size, num_puzzles, givens_range = args

    results = []
    for _ in range(num_puzzles):
        full = generate_sudoku_full(grid_size, box_size)
        givens = random.randint(givens_range[0], givens_range[1])
        puzzle = create_sudoku_puzzle(full, grid_size, givens)

        puzzle_str = grid_to_string(puzzle, grid_size)
        solution_str = grid_to_string(full, grid_size)
        results.append((puzzle_str, solution_str))

    return results


def generate_sudoku_dataset(
    grid_size: int,
    box_size: int,
    num_train: int,
    num_test: int,
    output_dir: str,
    num_workers: int = 4
):
    """Generate train/test split for Sudoku."""

    if grid_size == 4:
        givens_range = (6, 10)
    elif grid_size == 16:
        givens_range = (80, 120)
    elif grid_size == 25:
        givens_range = (200, 280)
    else:
        givens_range = (int(grid_size * grid_size * 0.3), int(grid_size * grid_size * 0.5))

    total = num_train + num_test
    print(f"\nGenerating {total} {grid_size}x{grid_size} Sudoku puzzles...")
    print(f"  Givens range: {givens_range[0]}-{givens_range[1]}")

    batch_size = max(1, total // num_workers)
    batches = [(grid_size, box_size, batch_size, givens_range) for _ in range(num_workers)]
    batches[-1] = (grid_size, box_size, total - batch_size * (num_workers - 1), givens_range)

    all_results = []
    with mp.Pool(num_workers) as pool:
        for i, batch_results in enumerate(pool.imap_unordered(generate_sudoku_batch, batches)):
            all_results.extend(batch_results)
            print(f"  Progress: {len(all_results)}/{total}")

    random.shuffle(all_results)
    train_data = all_results[:num_train]
    test_data = all_results[num_train:num_train + num_test]

    os.makedirs(output_dir, exist_ok=True)

    for split_name, data in [("train", train_data), ("test", test_data)]:
        puzzles_file = os.path.join(output_dir, f"sudoku_{grid_size}x{grid_size}_{split_name}_puzzles.txt")
        solutions_file = os.path.join(output_dir, f"sudoku_{grid_size}x{grid_size}_{split_name}_solutions.txt")

        with open(puzzles_file, 'w') as f:
            f.write('\n'.join(p for p, s in data))
        with open(solutions_file, 'w') as f:
            f.write('\n'.join(s for p, s in data))

        print(f"  Saved {len(data)} {split_name} samples")


# ============================================================================
# N-Queens Generator (Table 5)
# ============================================================================

def generate_nqueens_solution(n: int, max_attempts: int = 100) -> List[int]:
    """Generate N-Queens solution (column positions for each row)."""
    for _ in range(max_attempts):
        positions = [-1] * n
        cols = set()
        diag1 = set()
        diag2 = set()

        success = True
        for row in range(n):
            available = [c for c in range(n)
                        if c not in cols
                        and (row - c) not in diag1
                        and (row + c) not in diag2]

            if not available:
                success = False
                break

            col = random.choice(available)
            positions[row] = col
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

        if success:
            return positions

    return None


def create_nqueens_puzzle(positions: List[int], n: int, num_givens: int) -> str:
    """Create N-Queens puzzle with given number of pre-placed queens."""
    board = [['.'] * n for _ in range(n)]

    all_rows = list(range(n))
    random.shuffle(all_rows)

    for i in range(min(num_givens, n)):
        row = all_rows[i]
        col = positions[row]
        board[row][col] = 'Q'

    return ''.join(''.join(row) for row in board)


def solution_to_string(positions: List[int], n: int) -> str:
    """Convert solution to string format."""
    board = [['.'] * n for _ in range(n)]
    for row, col in enumerate(positions):
        board[row][col] = 'Q'
    return ''.join(''.join(row) for row in board)


def generate_nqueens_dataset(
    n: int,
    num_train: int,
    num_test: int,
    output_dir: str,
    givens_range: Tuple[int, int] = (1, 3)
):
    """Generate train/test split for N-Queens."""

    total = num_train + num_test
    print(f"\nGenerating {total} {n}-Queens puzzles...")
    print(f"  Pre-placed queens: {givens_range[0]}-{givens_range[1]}")

    puzzles = []
    solutions = []

    for i in range(total):
        sol = generate_nqueens_solution(n)
        if sol is None:
            print(f"  Warning: Failed to generate solution {i+1}")
            continue

        num_givens = random.randint(givens_range[0], givens_range[1])
        puzzle = create_nqueens_puzzle(sol, n, num_givens)
        solution = solution_to_string(sol, n)

        puzzles.append(puzzle)
        solutions.append(solution)

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{total}")

    train_puzzles = puzzles[:num_train]
    train_solutions = solutions[:num_train]
    test_puzzles = puzzles[num_train:num_train + num_test]
    test_solutions = solutions[num_train:num_train + num_test]

    os.makedirs(output_dir, exist_ok=True)

    for split_name, puz, sol in [("train", train_puzzles, train_solutions), ("test", test_puzzles, test_solutions)]:
        puzzles_file = os.path.join(output_dir, f"nqueens_{n}x{n}_{split_name}_puzzles.txt")
        solutions_file = os.path.join(output_dir, f"nqueens_{n}x{n}_{split_name}_solutions.txt")

        with open(puzzles_file, 'w') as f:
            f.write('\n'.join(puz))
        with open(solutions_file, 'w') as f:
            f.write('\n'.join(sol))

        print(f"  Saved {len(puz)} {split_name} samples")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate train/test splits for Table 4 and Table 5')
    parser.add_argument('--table', type=int, required=True, choices=[4, 5],
                        help='Table number (4 for Sudoku, 5 for N-Queens)')
    parser.add_argument('--train', type=int, default=5000,
                        help='Number of training samples per size')
    parser.add_argument('--test', type=int, default=500,
                        help='Number of test samples per size')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated sizes (default: all sizes for the table)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (Sudoku only)')

    args = parser.parse_args()

    if args.table == 4:
        output_dir = args.output_dir or "dataset/generalized_sudoku"
        sizes = args.sizes.split(',') if args.sizes else ['4', '16', '25']

        size_configs = {
            '4': (4, 2),
            '16': (16, 4),
            '25': (25, 5),
        }

        print("="*60)
        print("TABLE 4: Generalized Sudoku Dataset Generation")
        print("="*60)

        for size in sizes:
            if size not in size_configs:
                print(f"Warning: Unknown size {size}, skipping")
                continue

            grid_size, box_size = size_configs[size]
            generate_sudoku_dataset(
                grid_size, box_size,
                args.train, args.test,
                output_dir, args.workers
            )

    elif args.table == 5:
        output_dir = args.output_dir or "dataset/n_queens"
        sizes = args.sizes.split(',') if args.sizes else ['8', '9', '10']

        print("="*60)
        print("TABLE 5: N-Queens Dataset Generation")
        print("="*60)

        for size in sizes:
            try:
                n = int(size)
                generate_nqueens_dataset(n, args.train, args.test, output_dir)
            except ValueError:
                print(f"Warning: Invalid size {size}, skipping")

    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
