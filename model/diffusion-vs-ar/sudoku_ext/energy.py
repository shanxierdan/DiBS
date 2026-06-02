"""
Constraint Energy module for Sudoku training.
Computes energy penalties based on Sudoku constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .constraints import get_constraints
from .config import get_config


class ConstraintEnergy:
    """
    Computes constraint energy for Sudoku training.

    The energy penalizes violations of Sudoku constraints:
    - Each digit (1-9) should appear exactly once in each row, column, and box.
    """

    def __init__(self, config=None):
        """
        Initialize ConstraintEnergy.

        Args:
            config: SudokuConfig instance (uses global config if None).
        """
        self.config = config or get_config()
        self.constraints = get_constraints()

        # Precompute group and cell_to_groups tensors (will be moved to device when needed)
        self.groups_tensor = self.constraints.get_groups_tensor()
        self.cell_to_groups_tensor = self.constraints.get_cell_to_groups_tensor()

        # Constants
        self.n_digits = 9  # Digits 1-9
        self.n_groups = 27  # 9 rows + 9 columns + 9 boxes

    def compute_energy(self,
                       logits: torch.Tensor,
                       t: Optional[torch.Tensor] = None,
                       T: Optional[int] = None,
                       mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Compute constraint energy for given logits.

        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size) or (batch_size, seq_len).
                   If seq_len is 81, assumes full Sudoku grid.
            t: Optional timestep tensor of shape (batch_size,) for time gating.
            T: Total timesteps (for time gating calculation).
            mask: Optional mask tensor of shape (batch_size, seq_len) indicating valid positions.

        Returns:
            Tuple of (energy, metrics_dict)
            - energy: Scalar energy value
            - metrics: Dictionary with auxiliary metrics
        """
        batch_size = logits.shape[0]

        # Convert logits to probabilities for digits 1-9
        if logits.dim() == 2:
            # (batch_size, seq_len) - assume these are already probabilities or need reshaping
            raise ValueError("logits should have shape (batch_size, seq_len, vocab_size)")
        elif logits.dim() == 3:
            # (batch_size, seq_len, vocab_size)
            # Extract only digit probabilities (indices 1-9 for digits 1-9)
            # Vocabulary: ["0", "1", "2", ..., "9", ...] -> token id 0-9 for digits 0-9
            # Digits 1-9 correspond to token id 1-9 (not the last 9 dimensions!)
            digit_logits = logits[..., 1:10]  # token id 1-9 for digits 1-9
            # Apply softmax over digit dimension with numerical stability
            digit_logits = torch.clamp(digit_logits, min=-10, max=10)
            q_digits = F.softmax(digit_logits, dim=-1)
        else:
            raise ValueError(f"logits must be 2D or 3D, got shape {logits.shape}")

        # Ensure we have 81 cells for Sudoku
        if q_digits.shape[1] != 81:
            raise ValueError(f"Expected sequence length 81 for Sudoku, got {q_digits.shape[1]}")

        # Apply mask if provided (zero out masked positions)
        if mask is not None:
            if mask.shape != (batch_size, 81):
                raise ValueError(f"mask must have shape (batch_size, 81), got {mask.shape}")
            q_digits = q_digits * mask.unsqueeze(-1)

        # Move groups tensor to same device as q_digits
        device = q_digits.device
        if self.groups_tensor.device != device:
            self.groups_tensor = self.groups_tensor.to(device)
            self.cell_to_groups_tensor = self.cell_to_groups_tensor.to(device)

        # Compute soft counts per group per digit
        # q_digits shape: (batch_size, 81, 9)
        soft_counts = torch.zeros(batch_size, self.n_groups, self.n_digits,
                                  device=device, dtype=q_digits.dtype)

        # For each group, sum probabilities of member cells
        for group_idx in range(self.n_groups):
            cells_in_group = self.groups_tensor[group_idx]  # shape (9,)
            # Gather q values for cells in this group
            group_q = q_digits[:, cells_in_group, :]  # shape (batch_size, 9, 9)
            # Sum over cells in group
            soft_counts[:, group_idx, :] = group_q.sum(dim=1)

        # Compute energy based on energy_type
        if self.config.energy_type == "over":
            # Only penalize overcounting (soft_counts > 1)
            over_counts = torch.relu(soft_counts - 1.0)
            energy_per_group_digit = over_counts ** 2
        elif self.config.energy_type == "dup":
            # Penalize both overcounting and undercounting
            diff = soft_counts - 1.0
            energy_per_group_digit = diff ** 2
        else:
            raise ValueError(f"Unknown energy_type: {self.config.energy_type}")

        # Sum over groups and digits, then average over batch
        # This ensures energy scale is independent of batch_size
        energy = energy_per_group_digit.sum() / batch_size

        # Compute time gating factor
        omega_t = 1.0
        if t is not None and T is not None:
            # t is shape (batch_size,), compute omega for each sample
            t_e = int(self.config.t_e_ratio * T)
            # omega_t = 1 if t <= t_e, else 0 (per development doc)
            omega_t = float((t <= t_e).float().mean().item())

        # Apply time gating
        energy = energy * omega_t

        # Compute metrics
        metrics = {
            "energy_raw": energy.item() if torch.is_tensor(energy) else energy,
            "omega_t": omega_t,
            "soft_counts_mean": soft_counts.mean().item(),
            "soft_counts_std": soft_counts.std().item(),
            "max_violation": (soft_counts - 1.0).abs().max().item(),
        }

        return energy, metrics

    def add_to_loss(self,
                    logits: torch.Tensor,
                    base_loss: torch.Tensor,
                    t: Optional[torch.Tensor] = None,
                    T: Optional[int] = None,
                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Add constraint energy to base loss.

        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            base_loss: Base diffusion loss scalar
            t: Timestep tensor (batch_size,) if available
            T: Total diffusion steps if available
            mask: Position mask (batch_size, seq_len) if available

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        if not self.config.enable_constraint_energy:
            return base_loss, {"energy": 0.0, "energy_weighted": 0.0, "total_loss": base_loss.item()}

        # Compute constraint energy
        energy, energy_metrics = self.compute_energy(logits, t, T, mask)

        # Weight energy by lambda_e
        weighted_energy = self.config.lambda_e * energy

        # Add to base loss
        total_loss = base_loss + weighted_energy

        # Update metrics
        metrics = {
            **energy_metrics,
            "energy": energy.item() if torch.is_tensor(energy) else energy,
            "energy_weighted": weighted_energy.item(),
            "total_loss": total_loss.item(),
            "base_loss": base_loss.item(),
            "energy_ratio": weighted_energy.item() / total_loss.item() if total_loss.item() != 0 else 0.0,
        }

        return total_loss, metrics


# Convenience function
def compute_energy(logits: torch.Tensor,
                   t: Optional[torch.Tensor] = None,
                   T: Optional[int] = None,
                   mask: Optional[torch.Tensor] = None,
                   config=None) -> Tuple[torch.Tensor, dict]:
    """
    Convenience function to compute constraint energy.

    Args:
        logits: Model logits
        t: Timestep tensor
        T: Total timesteps
        mask: Position mask
        config: SudokuConfig instance (uses global config if None)

    Returns:
        Tuple of (energy, metrics)
    """
    energy_module = ConstraintEnergy(config)
    return energy_module.compute_energy(logits, t, T, mask)


def add_energy_to_loss(logits: torch.Tensor,
                       base_loss: torch.Tensor,
                       t: Optional[torch.Tensor] = None,
                       T: Optional[int] = None,
                       mask: Optional[torch.Tensor] = None,
                       config=None) -> Tuple[torch.Tensor, dict]:
    """
    Convenience function to add constraint energy to loss.

    Args:
        logits: Model logits
        base_loss: Base loss
        t: Timestep tensor
        T: Total timesteps
        mask: Position mask
        config: SudokuConfig instance (uses global config if None)

    Returns:
        Tuple of (total_loss, metrics)
    """
    energy_module = ConstraintEnergy(config)
    return energy_module.add_to_loss(logits, base_loss, t, T, mask)