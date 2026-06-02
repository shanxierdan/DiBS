#!/usr/bin/env python3
"""
N-Queens Dataset Generator

Generates N-Queens puzzles with varying numbers of pre-placed queens.
Format: Each puzzle is represented as a string of N*N characters.
- '0' = empty cell
- '1'-'9' = queen (for N <= 9, use '1' for queen)
- For N > 9, use 'Q' for queen, '.' for empty

The goal is to place N queens on an NxN board such that no two queens
attack each other (same row, column, or diagonal).
"""

import random
import argparse
from typing import List, Tuple, Optional
import os


def solve_n_queens(n: int, board: List[List[int]], row: int = 0) -> Optional[List[List[int]]]:
    """Backtracking solver for N-Queens with pre-placed queens."""
    if row == n:
        return [row[:] for row in board]

    if 1 in board[row]:
        return solve_n_queens(n, board, row + 1)

    for col in range(n):
        if is_safe(board, n, row, col):
            board[row][col] = 1
            result = solve_n_queens(n, board, row + 1)
            if result:
                return result
            board[row][col] = 0

    return None


def is_safe(board: List[List[int]], n: int, row: int, col: int) -> bool:
    """Check if placing a queen at (row, col) is safe."""
    for i in range(n):
        if board[row][i] == 1:
            return False
        if board[i][col] == 1:
            return False

    for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row - 1, -1, -1), range(col + 1, n)):
        if board[i][j] == 1:
            return False

    return True


def generate_full_solution(n: int) -> Optional[List[List[int]]]:
    """Generate a complete N-Queens solution."""
    board = [[0] * n for _ in range(n)]
    return solve_n_queens(n, board)


def generate_solution_fast(n: int) -> Optional[List[int]]:
    """
    Fast N-Queens solution generator using backtracking with randomization.
    Returns column positions for each row (0-indexed).
    """
    def backtrack(row: int, cols: set, diag1: set, diag2: set, positions: List[int]) -> bool:
        if row == n:
            return True

        available = [c for c in range(n) if c not in cols and (row - c) not in diag1 and (row + c) not in diag2]
        random.shuffle(available)

        for col in available:
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + c)
            positions.append(col)

            if backtrack(row + 1, cols, diag1, diag2, positions):
                return True

            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + c)
            positions.pop()

        return False

    positions = []
    if backtrack(0, set(), set(), set(), positions):
        return positions
    return None


def generate_solution_iterative(n: int, max_attempts: int = 100) -> Optional[List[int]]:
    """
    Iterative N-Queens solution using forward checking.
    More efficient for larger N.
    """
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


def create_puzzle_from_solution(positions: List[int], n: int, num_givens: int) -> List[List[int]]:
    """
    Create a puzzle by removing queens from a complete solution.
    positions: column index for each row (0-indexed)
    """
    board = [[0] * n for _ in range(n)]
    for row, col in enumerate(positions):
        board[row][col] = 1

    all_positions = [(r, c) for r in range(n) for c in range(n) if board[r][c] == 1]
    random.shuffle(all_positions)

    to_remove = n - num_givens
    for i in range(min(to_remove, len(all_positions))):
        r, c = all_positions[i]
        board[r][c] = 0

    return board


def board_to_string(board: List[List[int]], n: int) -> str:
    """Convert board to string format."""
    chars = []
    for row in board:
        for cell in row:
            if cell == 1:
                chars.append('Q')
            else:
                chars.append('.')
    return ''.join(chars)


def solution_to_string(positions: List[int], n: int) -> str:
    """Convert solution (column positions) to string format."""
    board = [['.'] * n for _ in range(n)]
    for row, col in enumerate(positions):
        board[row][col] = 'Q'
    return ''.join(''.join(row) for row in board)


def generate_puzzles(n: int, num_puzzles: int, givens_range: Tuple[int, int]) -> Tuple[List[str], List[str]]:
    """
    Generate N-Queens puzzles with solutions.

    Args:
        n: Board size (NxN)
        num_puzzles: Number of puzzles to generate
        givens_range: (min, max) number of pre-placed queens

    Returns:
        (puzzles, solutions) as lists of strings
    """
    puzzles = []
    solutions = []

    print(f"Generating {num_puzzles} {n}-Queens puzzles...")

    for i in range(num_puzzles):
        sol_positions = generate_solution_iterative(n)
        if sol_positions is None:
            print(f"  Warning: Failed to generate solution for puzzle {i+1}")
            continue

        num_givens = random.randint(givens_range[0], givens_range[1])
        puzzle_board = create_puzzle_from_solution(sol_positions, n, num_givens)

        puzzle_str = board_to_string(puzzle_board, n)
        solution_str = solution_to_string(sol_positions, n)

        puzzles.append(puzzle_str)
        solutions.append(solution_str)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_puzzles}")

    print(f"  Generated {len(puzzles)} puzzles successfully")
    return puzzles, solutions


def save_puzzles(puzzles: List[str], solutions: List[str], output_dir: str, n: int):
    """Save puzzles and solutions to files."""
    os.makedirs(output_dir, exist_ok=True)

    puzzle_file = os.path.join(output_dir, f"nqueens_{n}x{n}_puzzles.txt")
    solution_file = os.path.join(output_dir, f"nqueens_{n}x{n}_solutions.txt")

    with open(puzzle_file, 'w') as f:
        f.write('\n'.join(puzzles))

    with open(solution_file, 'w') as f:
        f.write('\n'.join(solutions))

    print(f"Saved {len(puzzles)} puzzles to {puzzle_file}")
    print(f"Saved {len(solutions)} solutions to {solution_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate N-Queens puzzles')
    parser.add_argument('--n', type=int, nargs='+', default=[8, 9, 10],
                        help='Board sizes to generate (default: 8 9 10)')
    parser.add_argument('--num-per-size', type=int, default=1000,
                        help='Number of puzzles per size (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='dataset/n_queens',
                        help='Output directory')
    parser.add_argument('--givens-min', type=int, default=1,
                        help='Minimum pre-placed queens')
    parser.add_argument('--givens-max', type=int, default=3,
                        help='Maximum pre-placed queens')

    args = parser.parse_args()

    for n in args.n:
        print(f"\n{'='*60}")
        print(f"Generating {n}-Queens puzzles")
        print(f"{'='*60}")

        givens_range = (min(args.givens_min, n), min(args.givens_max, n))
        puzzles, solutions = generate_puzzles(n, args.num_per_size, givens_range)
        save_puzzles(puzzles, solutions, args.output_dir, n)


if __name__ == '__main__':
    main()
