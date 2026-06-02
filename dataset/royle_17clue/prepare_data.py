#!/usr/bin/env python3
"""
prepare_data.py for royle_17clue dataset

This script processes the royle_17clue dataset which contains:
- puzzles0_kaggle: From Kaggle
- puzzles1_unbiased: Unbiased puzzles
- puzzles2_17_clue: 17-clue puzzles (theoretically hardest)
- puzzles3_magictour_top1465: Top 1465 hard puzzles
- puzzles4_forum_hardest_1905: Hardest from forum
- puzzles5_forum_hardest_1905_11+: Even harder subset
- puzzles6_forum_hardest_1106: Another hard set
- puzzles7_serg_benchmark: Benchmark puzzles

Output format: 81-character string with '0' for empty cells
Output directory: dataset/prepared_data/
"""

import os
import sys
from pathlib import Path


def parse_puzzle_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    puzzle = line.replace('.', '0')
    if len(puzzle) >= 81:
        return puzzle[:81]
    return None


def process_file(input_path: str, output_path: str, max_puzzles: int = None) -> int:
    count = 0
    with open(input_path, 'r') as f_in:
        with open(output_path, 'w') as f_out:
            for line in f_in:
                puzzle = parse_puzzle_line(line)
                if puzzle:
                    f_out.write(puzzle + '\n')
                    count += 1
                    if max_puzzles and count >= max_puzzles:
                        break
    return count


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'tdoku' / 'data'
    output_dir = script_dir.parent / 'prepared_data'
    output_dir.mkdir(exist_ok=True)

    datasets = [
        ('puzzles0_kaggle', 'royle_kaggle.txt'),
        ('puzzles1_unbiased', 'royle_unbiased.txt'),
        ('puzzles2_17_clue', 'royle_17clue.txt'),
        ('puzzles3_magictour_top1465', 'royle_magictour_top1465.txt'),
        ('puzzles4_forum_hardest_1905', 'royle_forum_hardest_1905.txt'),
        ('puzzles5_forum_hardest_1905_11+', 'royle_forum_hardest_11plus.txt'),
        ('puzzles6_forum_hardest_1106', 'royle_forum_hardest_1106.txt'),
        ('puzzles7_serg_benchmark', 'royle_serg_benchmark.txt'),
    ]

    print("=" * 60)
    print("Processing royle_17clue dataset")
    print("=" * 60)

    total = 0
    for input_name, output_name in datasets:
        input_path = data_dir / input_name
        output_path = output_dir / output_name

        if not input_path.exists():
            print(f"  [SKIP] {input_name} not found")
            continue

        count = process_file(str(input_path), str(output_path))
        total += count
        print(f"  {input_name} -> {output_name}: {count} puzzles")

    print("-" * 60)
    print(f"Total: {total} puzzles processed")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
