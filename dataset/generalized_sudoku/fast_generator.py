#!/usr/bin/env python3
"""
Fast Sudoku Generator for Large Grids
Uses optimized backtracking
"""

import random
import argparse
from pathlib import Path
from typing import List, Tuple


def is_valid(grid: List[int], row: int, col: int, num: int, size: int, box_size: int) -> bool:
    for i in range(size):
        if grid[row * size + i] == num or grid[i * size + col] == num:
            return False

    br, bc = (row // box_size) * box_size, (col // box_size) * box_size
    for i in range(br, br + box_size):
        for j in range(bc, bc + box_size):
            if grid[i * size + j] == num:
                return False
    return True


def solve(grid: List[int], size: int, box_size: int) -> bool:
    empty = -1
    for i in range(size * size):
        if grid[i] == 0:
            empty = i
            break

    if empty == -1:
        return True

    row, col = empty // size, empty % size
    nums = list(range(1, size + 1))
    random.shuffle(nums)

    for num in nums:
        if is_valid(grid, row, col, num, size, box_size):
            grid[empty] = num
            if solve(grid, size, box_size):
                return True
            grid[empty] = 0
    return False


def generate_full(size: int, box_size: int) -> List[int]:
    grid = [0] * (size * size)
    solve(grid, size, box_size)
    return grid


def create_puzzle(grid: List[int], size: int, target_givens: int) -> List[int]:
    puzzle = grid[:]
    positions = list(range(size * size))
    random.shuffle(positions)

    removed = 0
    target = size * size - target_givens

    for pos in positions:
        if removed >= target:
            break
        puzzle[pos] = 0
        removed += 1

    return puzzle


def grid_to_string(grid: List[int], size: int) -> str:
    result = []
    for v in grid:
        if v == 0:
            result.append('0')
        elif v <= 9:
            result.append(str(v))
        else:
            result.append(chr(ord('A') + v - 10))
    return ''.join(result)


def generate_puzzles(size: int, num_puzzles: int) -> List[str]:
    box_size = int(size ** 0.5)

    if size == 4:
        givens_range = (6, 10)
    elif size == 16:
        givens_range = (80, 120)
    else:
        givens_range = (200, 280)

    print(f"Generating {num_puzzles} {size}x{size} puzzles...")

    puzzles = []
    for i in range(num_puzzles):
        full = generate_full(size, box_size)
        givens = random.randint(givens_range[0], givens_range[1])
        puzzle = create_puzzle(full, size, givens)
        puzzles.append(grid_to_string(puzzle, size))

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{num_puzzles}")

    return puzzles


def main():
    parser = argparse.ArgumentParser(description="Generate generalized sudoku puzzles")
    parser.add_argument("--output", type=str, default="dataset/generalized_sudoku")
    parser.add_argument("--num-4x4", type=int, default=1000)
    parser.add_argument("--num-16x16", type=int, default=500)
    parser.add_argument("--num-25x25", type=int, default=200)

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_4x4 > 0:
        puzzles = generate_puzzles(4, args.num_4x4)
        with open(output_dir / "sudoku_4x4_puzzles.txt", 'w') as f:
            f.write('\n'.join(puzzles))
        print(f"Saved {len(puzzles)} 4x4 puzzles")

    if args.num_16x16 > 0:
        puzzles = generate_puzzles(16, args.num_16x16)
        with open(output_dir / "sudoku_16x16_puzzles.txt", 'w') as f:
            f.write('\n'.join(puzzles))
        print(f"Saved {len(puzzles)} 16x16 puzzles")

    if args.num_25x25 > 0:
        puzzles = generate_puzzles(25, args.num_25x25)
        with open(output_dir / "sudoku_25x25_puzzles.txt", 'w') as f:
            f.write('\n'.join(puzzles))
        print(f"Saved {len(puzzles)} 25x25 puzzles")

    print("\nDone!")


if __name__ == "__main__":
    main()
