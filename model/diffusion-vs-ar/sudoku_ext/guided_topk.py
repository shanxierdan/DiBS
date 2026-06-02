"""
Guided TopK Decoding module for Sudoku inference.
Selects positions to update based on uncertainty and conflict scores.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .constraints import get_constraints
from .config import get_config


class GuidedTopK:
    """
    Guided TopK selection for Sudoku inference.

    Selects positions to update based on:
    - Uncertainty: 1 - max probability
    - Conflict: Number of duplicates in constraint groups
    - Mask priority: Whether position is still masked
    """

    def __init__(self, config=None):
        """
        Initialize GuidedTopK.

        Args:
            config: SudokuConfig instance (uses global config if None).
        """
        self.config = config or get_config()
        self.constraints = get_constraints()

        # Precompute tensors
        self.groups_tensor = self.constraints.get_groups_tensor()
        self.cell_to_groups_tensor = self.constraints.get_cell_to_groups_tensor()

        # Constants
        self.n_cells = 81

    def compute_scores(self,
                       logits: torch.Tensor,
                       current_tokens: torch.Tensor,
                       maskable_mask: torch.Tensor,
                       t: int,
                       T: int) -> Tuple[torch.Tensor, dict]:
        """
        Compute guided scores for each position.

        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            current_tokens: Tensor of shape (batch_size, seq_len) with current token ids
            maskable_mask: Boolean tensor of shape (batch_size, seq_len) indicating maskable positions
            t: Current timestep (0 <= t < T)
            T: Total timesteps

        Returns:
            Tuple of (scores, metrics_dict)
            - scores: Tensor of shape (batch_size, seq_len) with selection scores
            - metrics: Dictionary with auxiliary metrics
        """
        batch_size = logits.shape[0]
        device = logits.device

        # Ensure tensors are on correct device
        if self.groups_tensor.device != device:
            self.groups_tensor = self.groups_tensor.to(device)
            self.cell_to_groups_tensor = self.cell_to_groups_tensor.to(device)

        # 1. Compute uncertainty scores
        # Extract digit logits (token id 1-9 for digits 1-9)
        # Vocabulary: ["0", "1", "2", ..., "9", ...] -> token id 0-9 for digits 0-9
        # Digits 1-9 correspond to token id 1-9 (not the last 9 dimensions!)
        digit_logits = logits[..., 1:10]  # (batch_size, seq_len, 9) for digits 1-9
        # Apply numerical stability
        digit_logits = torch.clamp(digit_logits, min=-10, max=10)
        digit_probs = F.softmax(digit_logits, dim=-1)
        max_probs, _ = digit_probs.max(dim=-1)  # (batch_size, seq_len)
        uncertainty = 1.0 - max_probs  # (batch_size, seq_len)

        # 2. Compute conflict scores
        # Get hard predictions for digits
        # digit_probs.argmax(dim=-1) returns 0-8, +1 gives 1-9
        predicted_digits = digit_probs.argmax(dim=-1) + 1  # 1-9  # (batch_size, seq_len)
        # For positions that are not digits yet, we should use current tokens
        # But for conflict calculation, we only care about positions that have digit predictions
        conflict_scores = self._compute_conflict_scores(predicted_digits, maskable_mask, t, T)

        # 3. Compute mask priority (if enabled)
        mask_priority = torch.zeros_like(uncertainty)
        if self.config.gamma_m > 0:
            # Priority for still-masked positions
            # current_tokens contains token ids - need to check which are mask tokens
            # We'll assume mask token id is known or can be inferred
            # For now, use maskable_mask directly (positions that were originally maskable)
            mask_priority = maskable_mask.float()

        # 4. Combine scores
        alpha = self.config.alpha_u
        beta = self.config.get_conflict_weight(t, T)
        gamma = self.config.gamma_m

        scores = (
            alpha * uncertainty +
            beta * conflict_scores +
            gamma * mask_priority
        )

        # 5. Normalize scores if needed (optional)
        # For now, just return raw scores

        # Compute metrics
        metrics = {
            "uncertainty_mean": uncertainty.mean().item(),
            "uncertainty_max": uncertainty.max().item(),
            "conflict_mean": conflict_scores.mean().item(),
            "conflict_max": conflict_scores.max().item(),
            "mask_priority_mean": mask_priority.mean().item(),
            "score_mean": scores.mean().item(),
            "score_std": scores.std().item(),
            "t": t,
            "beta_effective": beta,
        }

        return scores, metrics

    def _compute_conflict_scores(self,
                                 predicted_digits: torch.Tensor,
                                 maskable_mask: torch.Tensor,
                                 t: int,
                                 T: int) -> torch.Tensor:
        """
        Compute conflict scores for each cell.

        Args:
            predicted_digits: Tensor of shape (batch_size, seq_len) with predicted digits (1-9)
            maskable_mask: Boolean tensor indicating maskable positions
            t: Current timestep
            T: Total timesteps

        Returns:
            Conflict scores tensor of shape (batch_size, seq_len)
        """
        batch_size = predicted_digits.shape[0]
        device = predicted_digits.device

        # Initialize conflict scores
        conflict_scores = torch.zeros_like(predicted_digits, dtype=torch.float)

        # For each group
        for group_idx in range(27):
            cells_in_group = self.groups_tensor[group_idx]  # shape (9,)

            # Get predictions for cells in this group
            group_preds = predicted_digits[:, cells_in_group]  # shape (batch_size, 9)

            # For each cell in the group
            for pos_in_group, cell_idx in enumerate(cells_in_group):
                # Create mask for other cells with same prediction
                same_pred_mask = (group_preds == group_preds[:, pos_in_group:pos_in_group+1])
                # Count duplicates (excluding self)
                duplicate_count = same_pred_mask.sum(dim=1) - 1
                # Add to conflict score for this cell
                conflict_scores[:, cell_idx] += duplicate_count.float()

        # Normalize conflict scores (each cell can have max 3 groups * 8 duplicates = 24)
        # Clamp to [0, 3] as suggested in the doc
        conflict_scores = torch.clamp(conflict_scores, 0, 3)

        return conflict_scores

    def select_indices(self,
                       scores: torch.Tensor,
                       maskable_mask: torch.Tensor,
                       t: int,
                       T: int) -> Tuple[torch.Tensor, dict]:
        """
        Select TopK indices based on scores.

        Args:
            scores: Tensor of shape (batch_size, seq_len) with selection scores
            maskable_mask: Boolean tensor of shape (batch_size, seq_len) indicating maskable positions
            t: Current timestep
            T: Total timesteps

        Returns:
            Tuple of (indices_mask, metrics_dict)
            - indices_mask: Boolean tensor of shape (batch_size, seq_len) with True for selected positions
            - metrics: Dictionary with auxiliary metrics
        """
        batch_size = scores.shape[0]

        # Get dynamic K value
        k = self.config.get_dynamic_k(t, T)

        # Set large negative scores for non-maskable positions
        scores_masked = scores.clone()
        scores_masked[~maskable_mask] = -1e10

        # Select TopK positions
        _, topk_indices = torch.topk(scores_masked, k=k, dim=1, sorted=False)

        # Create boolean mask
        batch_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1).expand(-1, k)
        indices_mask = torch.zeros_like(scores, dtype=torch.bool)
        indices_mask[batch_indices.reshape(-1), topk_indices.reshape(-1)] = True

        # Compute metrics
        selected_scores = scores_masked[indices_mask]
        metrics = {
            "k": k,
            "selected_mean_score": selected_scores.mean().item() if len(selected_scores) > 0 else 0.0,
            "selected_max_score": selected_scores.max().item() if len(selected_scores) > 0 else 0.0,
            "selected_min_score": selected_scores.min().item() if len(selected_scores) > 0 else 0.0,
            "n_selected": indices_mask.sum().item(),
            "maskable_count": maskable_mask.sum().item(),
        }

        return indices_mask, metrics

    def guided_topk(self,
                    logits: torch.Tensor,
                    current_tokens: torch.Tensor,
                    maskable_mask: torch.Tensor,
                    t: int,
                    T: int) -> Tuple[torch.Tensor, dict]:
        """
        Complete guided TopK selection process.

        Args:
            logits: Model logits
            current_tokens: Current token ids
            maskable_mask: Maskable positions
            t: Current timestep
            T: Total timesteps

        Returns:
            Tuple of (indices_mask, metrics_dict)
        """
        # Compute guided scores
        scores, score_metrics = self.compute_scores(logits, current_tokens, maskable_mask, t, T)

        # Select indices
        indices_mask, select_metrics = self.select_indices(scores, maskable_mask, t, T)

        # Combine metrics
        metrics = {**score_metrics, **select_metrics}

        return indices_mask, metrics


# Convenience functions
def compute_guided_scores(logits: torch.Tensor,
                          current_tokens: torch.Tensor,
                          maskable_mask: torch.Tensor,
                          t: int,
                          T: int,
                          config=None) -> Tuple[torch.Tensor, dict]:
    """
    Convenience function to compute guided scores.

    Args:
        logits: Model logits
        current_tokens: Current token ids
        maskable_mask: Maskable positions
        t: Current timestep
        T: Total timesteps
        config: SudokuConfig instance (uses global config if None)

    Returns:
        Tuple of (scores, metrics)
    """
    guided_topk = GuidedTopK(config)
    return guided_topk.compute_scores(logits, current_tokens, maskable_mask, t, T)


def guided_topk_selection(logits: torch.Tensor,
                          current_tokens: torch.Tensor,
                          maskable_mask: torch.Tensor,
                          t: int,
                          T: int,
                          config=None) -> Tuple[torch.Tensor, dict]:
    """
    Convenience function for guided TopK selection.

    Args:
        logits: Model logits
        current_tokens: Current token ids
        maskable_mask: Maskable positions
        t: Current timestep
        T: Total timesteps
        config: SudokuConfig instance (uses global config if None)

    Returns:
        Tuple of (indices_mask, metrics)
    """
    guided_topk = GuidedTopK(config)
    return guided_topk.guided_topk(logits, current_tokens, maskable_mask, t, T)