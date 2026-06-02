"""
Sudoku constraints generation.
Generates groups (27 constraint groups: rows, columns, boxes) and
cell-to-groups mapping for 81 sudoku cells.
"""

from typing import List, Tuple
import torch

class SudokuConstraints:
    """
    Generates and stores Sudoku constraint groups.

    Attributes:
        groups: List[List[int]], 27 groups each containing 9 cell indices (0..80)
        cell_to_groups: List[List[int]], 81 entries each containing 3 group indices
    """

    def __init__(self):
        """Initialize Sudoku constraints with 9x9 grid."""
        self.n_cells = 81
        self.grid_size = 9
        self.box_size = 3

        self.groups = []
        self.cell_to_groups = [[] for _ in range(self.n_cells)]

        self._generate_groups()
        self._validate_groups()

    def _generate_groups(self):
        """Generate row, column, and box constraint groups."""
        # Generate rows (groups 0-8)
        for row in range(self.grid_size):
            row_group = []
            for col in range(self.grid_size):
                idx = row * self.grid_size + col
                row_group.append(idx)
            self.groups.append(row_group)

        # Generate columns (groups 9-17)
        for col in range(self.grid_size):
            col_group = []
            for row in range(self.grid_size):
                idx = row * self.grid_size + col
                col_group.append(idx)
            self.groups.append(col_group)

        # Generate boxes (3x3 subgrids, groups 18-26)
        for box_row in range(self.box_size):
            for box_col in range(self.box_size):
                box_group = []
                for i in range(self.box_size):
                    for j in range(self.box_size):
                        row = box_row * self.box_size + i
                        col = box_col * self.box_size + j
                        idx = row * self.grid_size + col
                        box_group.append(idx)
                self.groups.append(box_group)

        # Build cell_to_groups mapping
        for group_id, group in enumerate(self.groups):
            for cell_idx in group:
                self.cell_to_groups[cell_idx].append(group_id)

    def _validate_groups(self) -> None:
        """Validate generated groups for correctness."""
        # Check number of groups
        assert len(self.groups) == 27, f"Expected 27 groups, got {len(self.groups)}"

        # Check each group has 9 cells
        for i, group in enumerate(self.groups):
            assert len(group) == 9, f"Group {i} has {len(group)} cells, expected 9"
            # Check all indices are valid
            for idx in group:
                assert 0 <= idx < 81, f"Invalid cell index {idx} in group {i}"

        # Check each cell belongs to exactly 3 groups
        for cell_idx, group_ids in enumerate(self.cell_to_groups):
            assert len(group_ids) == 3, f"Cell {cell_idx} belongs to {len(group_ids)} groups, expected 3"
            # Check the groups are distinct
            assert len(set(group_ids)) == 3, f"Cell {cell_idx} has duplicate groups: {group_ids}"

            # Verify the groups are correct: one row, one column, one box
            row_group = cell_idx // 9
            col_group = 9 + (cell_idx % 9)

            # Calculate box group
            row = cell_idx // 9
            col = cell_idx % 9
            box_row = row // 3
            box_col = col // 3
            box_group = 18 + box_row * 3 + box_col

            expected_groups = [row_group, col_group, box_group]
            assert set(group_ids) == set(expected_groups), \
                f"Cell {cell_idx}: expected groups {expected_groups}, got {group_ids}"

    def get_groups(self) -> List[List[int]]:
        """Return the list of constraint groups."""
        return self.groups

    def get_cell_to_groups(self) -> List[List[int]]:
        """Return cell-to-groups mapping."""
        return self.cell_to_groups

    def get_groups_tensor(self, device: torch.device = None) -> torch.Tensor:
        """
        Return groups as a tensor of shape (27, 9).

        Args:
            device: PyTorch device to place tensor on.

        Returns:
            Tensor of group indices.
        """
        tensor = torch.tensor(self.groups, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def get_cell_to_groups_tensor(self, device: torch.device = None) -> torch.Tensor:
        """
        Return cell_to_groups as a tensor of shape (81, 3).

        Args:
            device: PyTorch device to place tensor on.

        Returns:
            Tensor of group indices per cell.
        """
        tensor = torch.tensor(self.cell_to_groups, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def compute_cell_conflicts(self,
                               predictions: torch.Tensor,
                               cell_to_groups_tensor: torch.Tensor = None,
                               groups_tensor: torch.Tensor = None) -> torch.Tensor:
        """
        Compute conflict count per cell using hard predictions.

        Args:
            predictions: Tensor of shape (batch_size, 81) with predicted digits (1-9)
            cell_to_groups_tensor: Optional precomputed cell_to_groups tensor
            groups_tensor: Optional precomputed groups tensor

        Returns:
            Tensor of shape (batch_size, 81) with conflict counts per cell.
        """
        if cell_to_groups_tensor is None:
            cell_to_groups_tensor = self.get_cell_to_groups_tensor(predictions.device)
        if groups_tensor is None:
            groups_tensor = self.get_groups_tensor(predictions.device)

        batch_size = predictions.shape[0]
        device = predictions.device

        # Initialize conflict counts
        conflicts = torch.zeros_like(predictions, dtype=torch.float)

        # For each group
        for group_id in range(27):
            group_cells = groups_tensor[group_id]  # shape (9,)

            # Get predictions for cells in this group
            group_preds = predictions[:, group_cells]  # shape (batch_size, 9)

            # For each cell in the group
            for pos_in_group, cell_idx in enumerate(group_cells):
                # Count how many other cells in group have same prediction
                same_pred_mask = (group_preds == group_preds[:, pos_in_group:pos_in_group+1])
                # Subtract 1 to exclude self
                same_count = same_pred_mask.sum(dim=1) - 1
                # Add to conflict count for this cell
                conflicts[:, cell_idx] += same_count.float()

        return conflicts

# Singleton instance for easy access
_constraints_instance = SudokuConstraints()

def get_constraints() -> SudokuConstraints:
    """Get singleton SudokuConstraints instance."""
    return _constraints_instance

def get_groups() -> List[List[int]]:
    """Get constraint groups from singleton."""
    return _constraints_instance.get_groups()

def get_cell_to_groups() -> List[List[int]]:
    """Get cell-to-groups mapping from singleton."""
    return _constraints_instance.get_cell_to_groups()