# -*- coding: utf-8 -*-
"""
Test script for Sudoku extension modules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np

print("Testing Sudoku extension modules...")

# 1. Test constraints module
print("\n1. Testing SudokuConstraints...")
try:
    from sudoku_ext.constraints import get_constraints, SudokuConstraints

    constraints = SudokuConstraints()
    print("  [OK] SudokuConstraints initialized")

    groups = constraints.get_groups()
    print("  [OK] Number of groups: {} (expected: 27)".format(len(groups)))

    cell_to_groups = constraints.get_cell_to_groups()
    print("  [OK] Number of cells: {} (expected: 81)".format(len(cell_to_groups)))

    # Test a specific cell
    test_cell = 40  # row 4, col 4 (0-indexed)
    cell_groups = cell_to_groups[test_cell]
    print("  [OK] Cell {} belongs to groups: {}".format(test_cell, cell_groups))

    # Verify groups for cell 40
    row = test_cell // 9
    col = test_cell % 9
    box_row = row // 3
    box_col = col // 3
    expected_row_group = row
    expected_col_group = 9 + col
    expected_box_group = 18 + box_row * 3 + box_col
    expected_groups = [expected_row_group, expected_col_group, expected_box_group]

    assert set(cell_groups) == set(expected_groups), \
        "Cell {}: expected {}, got {}".format(test_cell, expected_groups, cell_groups)
    print("  [OK] Cell {} groups are correct".format(test_cell))

    # Test conflict computation
    print("  Testing conflict computation...")
    # Create a simple board with a conflict
    board = torch.ones((1, 81), dtype=torch.long) * 5  # All 5s (many conflicts)
    conflicts = constraints.compute_cell_conflicts(board)
    print("  [OK] Conflict computation shape: {}".format(conflicts.shape))

    # Cell 0 should have conflicts in its row, column, and box
    conflict_count = conflicts[0, 0].item()
    print("  [OK] Cell 0 conflict count: {} (should be > 0)".format(conflict_count))

    print("  [OK] All constraints tests passed!")

except Exception as e:
    print("  [FAIL] Constraints test failed: {}".format(e))
    import traceback
    traceback.print_exc()

# 2. Test energy module
print("\n2. Testing ConstraintEnergy...")
try:
    from sudoku_ext.energy import ConstraintEnergy
    from sudoku_ext.config import SudokuConfig

    config = SudokuConfig(enable_constraint_energy=True, lambda_e=0.5, energy_type="over")

    energy = ConstraintEnergy(config)
    print("  [OK] ConstraintEnergy initialized")

    # Create test logits for a valid Sudoku solution
    batch_size = 2
    seq_len = 81
    vocab_size = 31  # Typical vocab size

    # Create random logits
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Test energy computation
    energy_val, metrics = energy.compute_energy(logits, t=torch.tensor([0, 1]), T=20)
    print("  [OK] Energy computation successful")
    print("    Energy value: {:.4f}".format(energy_val.item()))
    print("    Metrics: {}".format(list(metrics.keys())))

    # Test with a perfect solution (should have low energy)
    # Vocabulary: token id 1-9 correspond to digits 1-9
    # Create logits that predict a valid Sudoku solution (each digit appears once per group)
    perfect_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1

    # Create a valid Sudoku solution pattern
    # For simplicity, use a known valid solution
    valid_solution = [
        [5,3,4,6,7,8,9,1,2],
        [6,7,2,1,9,5,3,4,8],
        [1,9,8,3,4,2,5,6,7],
        [8,5,9,7,6,1,4,2,3],
        [4,2,6,8,5,3,7,9,1],
        [7,1,3,9,2,4,8,5,6],
        [9,6,1,5,3,7,2,8,4],
        [2,8,7,4,1,9,6,3,5],
        [3,4,5,2,8,6,1,7,9]
    ]
    valid_flat = [valid_solution[i//9][i%9] for i in range(81)]  # digits 1-9

    # Set logits for correct digits to high values (token id = digit value)
    for i, digit in enumerate(valid_flat):
        perfect_logits[:, i, digit] = 10.0  # High logit for correct digit

    energy_perfect, perfect_metrics = energy.compute_energy(perfect_logits)
    print("  [OK] Valid solution energy: {:.4f} (should be close to 0)".format(energy_perfect.item()))

    # Verify that valid solution has very low energy
    assert energy_perfect.item() < 1.0, "Valid solution should have low energy, got {:.4f}".format(energy_perfect.item())
    print("  [OK] Valid solution has low energy as expected")

    # Test with a conflicting solution (should have high energy)
    conflict_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
    # All cells predict digit 5 (maximum conflict)
    conflict_logits[:, :, 5] = 10.0  # All cells predict digit 5

    energy_conflict, _ = energy.compute_energy(conflict_logits)
    print("  [OK] Conflicting solution energy: {:.4f} (should be high)".format(energy_conflict.item()))

    # Verify that conflicting solution has high energy
    assert energy_conflict.item() > energy_perfect.item(), \
        "Conflicting solution should have higher energy than valid solution"
    print("  [OK] Conflicting solution has higher energy as expected")

    # Test add_to_loss
    base_loss = torch.tensor(1.0, requires_grad=True)
    total_loss, loss_metrics = energy.add_to_loss(logits, base_loss)
    print("  [OK] add_to_loss successful")
    print("    Base loss: {:.4f}".format(base_loss.item()))
    print("    Total loss: {:.4f}".format(total_loss.item()))
    print("    Energy contribution: {:.4f}".format(loss_metrics.get('energy_weighted', 0)))

    # Test gradient flow
    total_loss.backward()
    print("  [OK] Gradient computation successful")

    print("  [OK] All energy tests passed!")

except Exception as e:
    print("  [FAIL] Energy test failed: {}".format(e))
    import traceback
    traceback.print_exc()

# 3. Test guided topk module
print("\n3. Testing GuidedTopK...")
try:
    from sudoku_ext.guided_topk import GuidedTopK

    config = SudokuConfig(enable_guided_topk=True, alpha_u=1.0, beta_c=1.0)

    guided_topk = GuidedTopK(config)
    print("  [OK] GuidedTopK initialized")

    # Create test data
    batch_size = 2
    seq_len = 81
    vocab_size = 31

    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    current_tokens = torch.randint(vocab_size, (batch_size, seq_len))
    maskable_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Make some positions non-maskable
    maskable_mask[:, 0:10] = False

    # Test compute_scores
    t = 10  # Mid-point
    T = 20
    scores, metrics = guided_topk.compute_scores(
        logits, current_tokens, maskable_mask, t, T
    )
    print("  [OK] Score computation successful")
    print("    Scores shape: {}".format(scores.shape))
    print("    Metrics: {}".format(list(metrics.keys())))

    # Test select_indices
    indices_mask, select_metrics = guided_topk.select_indices(
        scores, maskable_mask, t, T
    )
    print("  [OK] Index selection successful")
    print("    Selected {} positions".format(indices_mask.sum().item()))
    print("    Maskable positions: {}".format(maskable_mask.sum().item()))

    # Verify selected indices are within maskable positions
    selected_in_masked = indices_mask & maskable_mask
    assert torch.all(indices_mask == selected_in_masked), \
        "Selected indices should be within maskable positions"
    print("  [OK] All selected indices are within maskable positions")

    # Test full guided_topk
    indices_mask_full, full_metrics = guided_topk.guided_topk(
        logits, current_tokens, maskable_mask, t, T
    )
    print("  [OK] Full guided_topk successful")

    print("  [OK] All guided_topk tests passed!")

except Exception as e:
    print("  [FAIL] GuidedTopK test failed: {}".format(e))
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All Sudoku extension modules tested!")
print("="*50)
