#!/usr/bin/env python3
"""
prepare_data.py for satnet dataset

This script processes the SatNet Sudoku dataset which contains:
- features.pt: PyTorch tensor of puzzles (N x 9 x 9 x 9) - one-hot encoded
- labels.pt: PyTorch tensor of solutions (N x 9 x 9 x 9) - one-hot encoded

The tensors are one-hot encoded: (N, 9, 9, 9) where:
- First 9: channel (digit 1-9)
- Second 9: row
- Third 9: column

Output format: 81-character string with '0' for empty cells
Output directory: dataset/prepared_data/
"""

import os
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: PyTorch is required to process this dataset")
    print("Install with: pip install torch")
    sys.exit(1)


def onehot_to_string(tensor):
    result = []
    for row in range(9):
        for col in range(9):
            cell = tensor[:, row, col]
            if cell.sum() == 0:
                result.append('0')
            else:
                digit = cell.argmax().item() + 1
                result.append(str(digit))
    return ''.join(result)


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'sudoku'
    output_dir = script_dir.parent / 'prepared_data'
    output_dir.mkdir(exist_ok=True)

    features_path = data_dir / 'features.pt'
    labels_path = data_dir / 'labels.pt'

    puzzles_output = output_dir / 'satnet_puzzles.txt'
    solutions_output = output_dir / 'satnet_solutions.txt'

    if not features_path.exists() or not labels_path.exists():
        print(f"Error: features.pt or labels.pt not found in {data_dir}")
        return

    print("=" * 60)
    print("Processing satnet dataset")
    print("=" * 60)

    print("  Loading features.pt...")
    features = torch.load(features_path, map_location='cpu')
    print(f"  Features shape: {features.shape}")

    print("  Loading labels.pt...")
    labels = torch.load(labels_path, map_location='cpu')
    print(f"  Labels shape: {labels.shape}")

    n_samples = features.shape[0]
    print(f"  Total samples: {n_samples}")

    print("  Converting to text format...")
    with open(puzzles_output, 'w') as f_puzzles, \
         open(solutions_output, 'w') as f_solutions:

        for i in range(n_samples):
            puzzle_str = onehot_to_string(features[i])
            solution_str = onehot_to_string(labels[i])

            f_puzzles.write(puzzle_str + '\n')
            f_solutions.write(solution_str + '\n')

            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1}/{n_samples}...")

    print(f"  Puzzles output: {puzzles_output}")
    print(f"  Solutions output: {solutions_output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
