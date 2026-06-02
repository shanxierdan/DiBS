#!/usr/bin/env python3
"""
prepare_data.py for sudoku_kaggle dataset

This script processes the Kaggle Sudoku dataset which contains:
- sudoku.csv: Large dataset with quizzes and solutions

CSV format:
- quizzes: 81-character puzzle string
- solutions: 81-character solution string

Output format: 81-character string with '0' for empty cells
Output directory: dataset/prepared_data/
"""

import os
import csv
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    input_path = script_dir / 'sudoku.csv'
    output_dir = script_dir.parent / 'prepared_data'
    output_dir.mkdir(exist_ok=True)

    puzzles_output = output_dir / 'kaggle_puzzles.txt'
    solutions_output = output_dir / 'kaggle_solutions.txt'

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    print("=" * 60)
    print("Processing sudoku_kaggle dataset")
    print("=" * 60)

    puzzle_count = 0
    with open(input_path, 'r') as f_in:
        reader = csv.DictReader(f_in)

        with open(puzzles_output, 'w') as f_puzzles, \
             open(solutions_output, 'w') as f_solutions:

            for row in reader:
                quiz = row.get('quizzes', '').strip()
                solution = row.get('solutions', '').strip()

                if len(quiz) == 81 and len(solution) == 81:
                    f_puzzles.write(quiz + '\n')
                    f_solutions.write(solution + '\n')
                    puzzle_count += 1

                    if puzzle_count % 100000 == 0:
                        print(f"  Processed {puzzle_count} puzzles...")

    print(f"  Total puzzles: {puzzle_count}")
    print(f"  Puzzles output: {puzzles_output}")
    print(f"  Solutions output: {solutions_output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
