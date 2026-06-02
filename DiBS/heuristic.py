import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import hashlib
import math

from .model_wrapper import DiffusionModelWrapper
from .constraints import SudokuConstraints
from .config import DiBSConfig


@dataclass
class HeuristicConfig:
    alpha: float = 0.8
    beta: float = 0.5
    use_lcv: bool = False
    verbose: bool = False
    cache_logits: bool = True
    update_frequency: int = 1
    use_global_consistency: bool = True
    use_key_cell_detection: bool = True


class DiffusionHeuristic:
    def __init__(self, model_wrapper: DiffusionModelWrapper, config: HeuristicConfig):
        self.model = model_wrapper
        self.config = config
        self._logits_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_logits = None
        self._last_grid_hash = None
        self._consistency_cache = {}

    def _grid_hash(self, grid: np.ndarray) -> str:
        return hashlib.md5(grid.tobytes()).hexdigest()

    def get_logits(self, grid: np.ndarray, force_update: bool = False) -> np.ndarray:
        if self.config.cache_logits and not force_update:
            cache_key = self._grid_hash(grid)

            if cache_key == self._last_grid_hash and self._last_logits is not None:
                self._cache_hits += 1
                return self._last_logits

            if cache_key in self._logits_cache:
                self._cache_hits += 1
                return self._logits_cache[cache_key]

        self._cache_misses += 1
        logits = self.model.get_logits(grid)

        if self.config.cache_logits:
            cache_key = self._grid_hash(grid)
            self._logits_cache[cache_key] = logits
            self._last_logits = logits
            self._last_grid_hash = cache_key
            self._consistency_cache.clear()

            if len(self._logits_cache) > 1000:
                keys = list(self._logits_cache.keys())[:500]
                for k in keys:
                    del self._logits_cache[k]

        return logits

    def select_cell(self,
                    grid: np.ndarray,
                    constraints: SudokuConstraints,
                    mrv_cells: List[int]) -> int:
        if len(mrv_cells) == 1:
            return mrv_cells[0]

        logits = self.get_logits(grid)

        if self.config.use_key_cell_detection:
            return self._select_key_cell(logits, constraints, mrv_cells)
        else:
            return self._select_lowest_entropy_cell(logits, constraints, mrv_cells)

    def _select_key_cell(self, logits: np.ndarray, constraints: SudokuConstraints,
                         mrv_cells: List[int]) -> int:
        max_importance = -1
        selected_cell = mrv_cells[0]

        for cell in mrv_cells:
            candidates = constraints.get_candidates(cell)
            if not candidates:
                continue

            if len(candidates) == 2:
                probs = self._compute_projected_probs(logits[cell, :], candidates)
                confidence = max(probs.values())
                importance = 2.0 + confidence
            else:
                entropy = self._compute_entropy_for_candidates(logits[cell, :], candidates)
                importance = -entropy

            if importance > max_importance:
                max_importance = importance
                selected_cell = cell

        return selected_cell

    def _select_lowest_entropy_cell(self, logits: np.ndarray, constraints: SudokuConstraints,
                                    mrv_cells: List[int]) -> int:
        min_entropy = float('inf')
        selected_cell = mrv_cells[0]

        for cell in mrv_cells:
            candidates = constraints.get_candidates(cell)
            if not candidates:
                continue

            entropy = self._compute_entropy_for_candidates(logits[cell, :], candidates)

            if entropy < min_entropy:
                min_entropy = entropy
                selected_cell = cell

        return selected_cell

    def order_values(self,
                     cell: int,
                     constraints: SudokuConstraints,
                     logits: np.ndarray,
                     use_lcv: Optional[bool] = None) -> List[int]:
        candidates = constraints.get_candidates(cell)

        if not candidates:
            return []

        cell_logits = logits[cell, :]
        probs = self._compute_projected_probs(cell_logits, candidates)

        use_lcv = use_lcv if use_lcv is not None else self.config.use_lcv

        if use_lcv:
            return self._order_values_lcv(cell, candidates, probs, constraints)
        else:
            if self.config.use_global_consistency:
                return self._order_values_global(cell, candidates, probs, logits, constraints)
            else:
                return sorted(candidates, key=lambda d: -probs.get(d, 0))

    def _order_values_lcv(self, cell: int, candidates: List[int],
                          probs: Dict[int, float], constraints: SudokuConstraints) -> List[int]:
        scored_values = []
        for digit in candidates:
            model_score = probs.get(digit, 0)
            lcv_score = constraints.compute_lcv(cell, digit)

            combined_score = (self.config.beta * model_score -
                              (1 - self.config.beta) * lcv_score / 20)
            scored_values.append((digit, combined_score))

        scored_values.sort(key=lambda x: -x[1])
        return [d for d, _ in scored_values]

    def _order_values_global(self, cell: int, candidates: List[int],
                             probs: Dict[int, float], logits: np.ndarray,
                             constraints: SudokuConstraints) -> List[int]:
        scored_values = []

        for digit in candidates:
            model_score = probs.get(digit, 0)

            consistency_score = self._compute_consistency_score(cell, digit, logits, constraints)

            combined_score = 0.7 * model_score + 0.3 * consistency_score
            scored_values.append((digit, combined_score))

        scored_values.sort(key=lambda x: -x[1])
        return [d for d, _ in scored_values]

    def _compute_consistency_score(self, cell: int, digit: int,
                                   logits: np.ndarray, constraints: SudokuConstraints) -> float:
        cache_key = (cell, digit)
        if cache_key in self._consistency_cache:
            return self._consistency_cache[cache_key]

        row, col = cell // 9, cell % 9
        box_row, box_col = (row // 3) * 3, (col // 3) * 3

        consistency_scores = []

        for other_cell in range(81):
            if other_cell == cell:
                continue
            if constraints.grid[other_cell] != 0:
                continue

            other_row, other_col = other_cell // 9, other_cell % 9
            same_row = other_row == row
            same_col = other_col == col
            same_box = (other_row // 3 == row // 3 and other_col // 3 == col // 3)

            if not (same_row or same_col or same_box):
                continue

            other_candidates = constraints.get_candidates(other_cell)
            if digit in other_candidates:
                other_logits = logits[other_cell, :]
                other_probs = self._compute_projected_probs(other_logits, other_candidates)
                digit_prob = other_probs.get(digit, 0)
                consistency_scores.append(1.0 - digit_prob)

        if consistency_scores:
            score = sum(consistency_scores) / len(consistency_scores)
        else:
            score = 1.0

        self._consistency_cache[cache_key] = score
        return score

    def _compute_projected_probs(self, logits: np.ndarray, candidates: List[int]) -> dict:
        candidate_logits = np.array([logits[d - 1] for d in candidates])

        max_logit = np.max(candidate_logits)
        exp_logits = np.exp(candidate_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        return {d: p for d, p in zip(candidates, probs)}

    def _compute_entropy_for_candidates(self, logits: np.ndarray, candidates: List[int]) -> float:
        probs = self._compute_projected_probs(logits, candidates)
        return self._compute_entropy(probs)

    def _compute_entropy(self, probs: dict) -> float:
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def clear_cache(self):
        self._logits_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_logits = None
        self._last_grid_hash = None
        self._consistency_cache.clear()

    def get_cache_stats(self) -> Tuple[int, int]:
        return self._cache_hits, self._cache_misses

    def get_top_k_predictions(self, grid: np.ndarray, k: int = 5) -> List[Tuple[int, int, float]]:
        logits = self.get_logits(grid)

        predictions = []
        for cell in range(81):
            if grid[cell] != 0:
                continue

            cell_logits = logits[cell, :]
            exp_logits = np.exp(cell_logits - np.max(cell_logits))
            probs = exp_logits / np.sum(exp_logits)

            top_indices = np.argsort(probs)[-k:][::-1]
            for idx in top_indices:
                digit = idx + 1
                prob = probs[idx]
                predictions.append((cell, digit, prob))

        predictions.sort(key=lambda x: -x[2])
        return predictions[:k * 10]
