import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import json
import time
import math

from .config import DiBSConfig
from .model_wrapper import DiffusionModelWrapper
from .constraints import SudokuConstraints
from .metrics import SolveMetrics, MetricsCollector


@dataclass
class SmartCallConfig:
    call_on_mrv_2_only: bool = True
    min_interval: int = 5
    min_mrv_threshold: int = 2


class DiBSSolver:
    def __init__(self,
                 model_path: Optional[str] = None,
                 config: Optional[DiBSConfig] = None,
                 use_heuristic: bool = True,
                 use_lcv: bool = False,
                 use_fc: bool = True,
                 alpha: float = 0.8,
                 beta: float = 0.5,
                 smart_call: bool = True,
                 timeout_ms: float = 30000,
                 denoise_steps: Optional[int] = None,
                 denoise_fill_ratio: Optional[float] = None):

        self.config = config or DiBSConfig()

        if model_path:
            self.config.checkpoint_path = model_path
        self.config.use_heuristic = use_heuristic
        self.config.use_lcv = use_lcv
        self.config.alpha = alpha
        self.config.beta = beta
        if denoise_steps is not None:
            self.config.denoise_steps = int(denoise_steps)
        if denoise_fill_ratio is not None:
            self.config.denoise_fill_ratio = float(denoise_fill_ratio)

        self.use_fc = use_fc
        self._timeout_ms = timeout_ms
        self._start_time = None

        self.model: Optional[DiffusionModelWrapper] = None
        self.constraints = SudokuConstraints()
        self.metrics_collector = MetricsCollector()

        self.smart_call = smart_call
        self.smart_config = SmartCallConfig()

        self._current_logits = None
        self._nodes_since_call = 0
        self._model_calls = 0
        self._total_nodes = 0
        self._start_time = None

        if self.config.use_heuristic:
            self._init_model()

    def _init_model(self):
        if self.model is None:
            self.model = DiffusionModelWrapper(
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device
            )

    def _should_call_model(self) -> bool:
        if not self.smart_call:
            return True

        if self._nodes_since_call < self.smart_config.min_interval:
            return False

        mrv_cells = self.constraints.get_mrv_cells()
        if not mrv_cells:
            return False

        min_candidates = min(
            len(self.constraints.get_candidates(cell))
            for cell in mrv_cells
        )

        if min_candidates >= self.smart_config.min_mrv_threshold:
            return True

        return False

    def _call_model(self):
        if self.model is None:
            return

        t_model_start = time.perf_counter()
        denoise_steps = max(1, int(getattr(self.config, "denoise_steps", 1)))
        fill_ratio = float(getattr(self.config, "denoise_fill_ratio", 0.15))
        fill_ratio = max(0.0, min(1.0, fill_ratio))
        denoise_strategy = str(getattr(self.config, "denoise_strategy", "legacy_repeat"))
        mdm_decoding = str(getattr(self.config, "mdm_decoding_strategy", "deterministic-cosine"))

        if denoise_strategy == "mdm_iterative":
            self._current_logits = self.model.get_logits_mdm_iterative(
                self.constraints.grid,
                diffusion_steps=denoise_steps,
                decoding_strategy=mdm_decoding,
            )
            self._model_calls += denoise_steps
            self._nodes_since_call = 0
            self._cell_probs = {}
            for cell in range(81):
                if self.constraints.grid[cell] == 0:
                    candidates = self.constraints.get_candidates(cell)
                    if candidates:
                        cell_logits = self._current_logits[cell, :]
                        candidate_logits = np.array([cell_logits[d - 1] for d in candidates])
                        max_logit = np.max(candidate_logits)
                        exp_logits = np.exp(candidate_logits - max_logit)
                        probs = exp_logits / np.sum(exp_logits)
                        self._cell_probs[cell] = {d: probs[i] for i, d in enumerate(candidates)}

            inference_time = (time.perf_counter() - t_model_start) * 1000.0
            self.metrics_collector.add_model_time(inference_time)
            for _ in range(max(0, denoise_steps - 1)):
                self.metrics_collector.increment_model_calls()
            return

        working_grid = np.array(self.constraints.grid, copy=True)
        final_logits = None

        for step in range(denoise_steps):
            # Keep legacy behavior when denoise_steps == 1.
            timestep = max(0, int(getattr(self.config, "diffusion_timestep", 10)) - step)
            logits = self.model.get_logits(working_grid, timestep=timestep)
            final_logits = logits

            # Iterative pseudo-denoising: fill a small set of high-confidence unknowns,
            # then query the model again on the refined board.
            if step >= denoise_steps - 1 or fill_ratio <= 0.0:
                continue

            empty_cells = np.where(working_grid == 0)[0].tolist()
            if not empty_cells:
                break

            candidates_to_fill = []
            for cell in empty_cells:
                cand = self._get_candidates_from_grid(working_grid, int(cell))
                if not cand:
                    continue
                cell_logits = logits[cell, :]
                cand_logits = np.array([cell_logits[d - 1] for d in cand], dtype=np.float64)
                max_logit = float(np.max(cand_logits))
                exp_logits = np.exp(cand_logits - max_logit)
                den = float(np.sum(exp_logits))
                if den <= 0:
                    continue
                probs = exp_logits / den
                best_idx = int(np.argmax(probs))
                best_digit = int(cand[best_idx])
                best_prob = float(probs[best_idx])
                candidates_to_fill.append((best_prob, int(cell), best_digit))

            if not candidates_to_fill:
                continue

            candidates_to_fill.sort(key=lambda x: x[0], reverse=True)
            k = max(1, int(np.ceil(len(candidates_to_fill) * fill_ratio)))
            for _, cell, digit in candidates_to_fill[:k]:
                if working_grid[cell] == 0:
                    working_grid[cell] = digit

        self._current_logits = final_logits if final_logits is not None else self.model.get_logits(self.constraints.grid)
        self._model_calls += denoise_steps
        self._nodes_since_call = 0

        self._cell_probs = {}
        for cell in range(81):
            if self.constraints.grid[cell] == 0:
                candidates = self.constraints.get_candidates(cell)
                if candidates:
                    cell_logits = self._current_logits[cell, :]
                    candidate_logits = np.array([cell_logits[d - 1] for d in candidates])
                    max_logit = np.max(candidate_logits)
                    exp_logits = np.exp(candidate_logits - max_logit)
                    probs = exp_logits / np.sum(exp_logits)
                    self._cell_probs[cell] = {d: probs[i] for i, d in enumerate(candidates)}

        inference_time = (time.perf_counter() - t_model_start) * 1000.0
        self.metrics_collector.add_model_time(inference_time)
        for _ in range(max(0, denoise_steps - 1)):
            self.metrics_collector.increment_model_calls()

    def _get_candidates_from_grid(self, flat_grid: np.ndarray, cell: int) -> List[int]:
        if flat_grid[cell] != 0:
            return []
        row, col = divmod(cell, 9)
        used = set()
        for i in range(9):
            used.add(int(flat_grid[row * 9 + i]))
            used.add(int(flat_grid[i * 9 + col]))
        box_r = (row // 3) * 3
        box_c = (col // 3) * 3
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                used.add(int(flat_grid[r * 9 + c]))
        return [d for d in range(1, 10) if d not in used]

    def _select_cell_with_logits(self, mrv_cells: List[int]) -> int:
        if self._current_logits is None:
            return mrv_cells[0]

        min_entropy = float('inf')
        selected_cell = mrv_cells[0]

        for cell in mrv_cells:
            probs_dict = self._cell_probs.get(cell, {})
            if not probs_dict:
                continue

            entropy = 0.0
            for p in probs_dict.values():
                if p > 0:
                    entropy -= p * math.log2(p)

            if entropy < min_entropy:
                min_entropy = entropy
                selected_cell = cell

        return selected_cell

    def _order_values_with_logits(self, cell: int, candidates: List[int]) -> List[int]:
        if self._current_logits is None or not candidates:
            return candidates

        probs_dict = self._cell_probs.get(cell, {})
        if not probs_dict:
            return candidates

        probs = np.array([probs_dict.get(d, 0.0) for d in candidates])

        consistency_scores = []
        for digit in candidates:
            score = self._compute_consistency_score(cell, digit)
            consistency_scores.append(score)

        alpha = self.config.alpha
        combined_scores = alpha * probs + (1 - alpha) * np.array(consistency_scores)

        sorted_indices = np.argsort(combined_scores)[::-1]
        return [candidates[i] for i in sorted_indices]

    def _compute_consistency_score(self, cell: int, digit: int) -> float:
        if not hasattr(self, '_cell_probs') or not self._cell_probs:
            return 0.0

        row, col = cell // 9, cell % 9

        consistency_values = []

        for other_cell in range(81):
            if other_cell == cell:
                continue
            if self.constraints.grid[other_cell] != 0:
                continue

            other_row, other_col = other_cell // 9, other_cell % 9
            same_row = other_row == row
            same_col = other_col == col
            same_box = (other_row // 3 == row // 3 and other_col // 3 == col // 3)

            if not (same_row or same_col or same_box):
                continue

            probs_dict = self._cell_probs.get(other_cell, {})
            if not probs_dict:
                continue

            digit_prob = probs_dict.get(digit, 0.0)
            consistency_values.append(1.0 - digit_prob)

        if not consistency_values:
            return 0.0

        return np.mean(consistency_values)

    def solve(self, puzzle: str) -> tuple:
        self.metrics_collector.start_solve()

        self._current_logits = None
        self._nodes_since_call = 0
        self._model_calls = 0
        self._total_nodes = 0
        self._start_time = time.time()

        grid = self._parse_puzzle(puzzle)

        if not self.constraints.initialize(grid):
            self.metrics_collector.end_solve(False, False)
            return None, self.metrics_collector.metrics_list[-1]

        if self.use_fc:
            changed, valid = self.constraints.propagate()
            self.metrics_collector.increment_propagation()

        if self.use_fc and not valid:
            self.metrics_collector.end_solve(False, False)
            return None, self.metrics_collector.metrics_list[-1]

        if self.constraints.is_complete():
            solution = self.constraints.get_solution_string()
            is_valid = self.constraints.validate_solution()
            self.metrics_collector.end_solve(True, is_valid)
            return solution, self.metrics_collector.metrics_list[-1]

        if self.config.use_heuristic and self._should_call_model():
            self._call_model()

        success = self._solve_recursive()

        if success:
            solution = self.constraints.get_solution_string()
            is_valid = self.constraints.validate_solution()
            self.metrics_collector.end_solve(True, is_valid)
            return solution, self.metrics_collector.metrics_list[-1]
        else:
            self.metrics_collector.end_solve(False, False)
            return None, self.metrics_collector.metrics_list[-1]

    def _solve_recursive(self) -> bool:
        if self.constraints.is_complete():
            return True

        if self.metrics_collector._current_metrics.expanded_nodes >= self.config.max_nodes:
            return False

        self.metrics_collector.increment_nodes()
        self._total_nodes += 1
        self._nodes_since_call += 1

        mrv_cells = self.constraints.get_mrv_cells()

        if not mrv_cells:
            return False

        candidates = self.constraints.get_candidates(mrv_cells[0])
        min_candidates = len(candidates)

        if self.config.use_heuristic:
            if self._should_call_model():
                self._call_model()

        if self.config.use_heuristic and self._current_logits is not None:
            cell = self._select_cell_with_logits(mrv_cells)
            cell_candidates = self.constraints.get_candidates(cell)
            ordered_values = self._order_values_with_logits(cell, cell_candidates)
        else:
            cell = mrv_cells[0]
            ordered_values = self.constraints.get_candidates(cell)

        if not ordered_values:
            return False

        for digit in ordered_values:
            if self._start_time is not None and (time.time() - self._start_time) * 1000 > self._timeout_ms:
                return False

            state = self.constraints.save_state()
            valid = False

            if self.constraints.assign(cell, digit):
                if self.use_fc:
                    result = self.constraints.propagate()
                    changed, valid = result[0], result[1]
                    # Count actual propagation iterations
                    iteration_count = result[2] if len(result) > 2 else 1
                    for _ in range(iteration_count):
                        self.metrics_collector.increment_propagation()
                else:
                    valid = True

            if valid:
                if self._solve_recursive():
                    return True

            self.constraints.restore_state(state)
            self.metrics_collector.increment_backtracks()

        return False

    def _parse_puzzle(self, puzzle: str) -> np.ndarray:
        puzzle = puzzle.strip().replace('.', '0')

        if len(puzzle) == 81:
            return np.array([int(c) if c.isdigit() else 0 for c in puzzle], dtype=np.int64)
        elif len(puzzle) == 81 * 2 + 8:
            lines = puzzle.split('\n')
            digits = []
            for line in lines:
                for c in line:
                    if c.isdigit() or c == '.':
                        digits.append(int(c) if c.isdigit() else 0)
            return np.array(digits[:81], dtype=np.int64)
        else:
            raise ValueError(f"Invalid puzzle format: length={len(puzzle)}")

    def solve_batch(self, puzzles: List[str]) -> List[tuple]:
        results = []
        for puzzle in puzzles:
            solution, metrics = self.solve(puzzle)
            results.append((solution, metrics))
        return results

    def get_benchmark_results(self):
        return self.metrics_collector.compute_benchmark_results()

    def clear_metrics(self):
        self.metrics_collector.clear()
        self._current_logits = None
        self._cell_probs = {}
        self._nodes_since_call = 0
        self._model_calls = 0
        self._total_nodes = 0


class BaselineSolver:
    def __init__(self,
                 use_lcv: bool = False,
                 use_fc: bool = True,
                 use_degree_tiebreak: bool = False,
                 max_nodes: int = 100000,
                 timeout_ms: float = 30000):
        self.use_lcv = use_lcv
        self.use_fc = use_fc
        self.use_degree_tiebreak = use_degree_tiebreak
        self.max_nodes = max_nodes
        self._timeout_ms = timeout_ms
        self._start_time = None
        self.constraints = SudokuConstraints()
        self.metrics_collector = MetricsCollector()

    def solve(self, puzzle: str) -> tuple:
        self.metrics_collector.start_solve()
        self._start_time = time.time()

        grid = self._parse_puzzle(puzzle)

        if not self.constraints.initialize(grid):
            self.metrics_collector.end_solve(False, False)
            return None, self.metrics_collector.metrics_list[-1]

        if self.use_fc:
            result = self.constraints.propagate()
            changed, valid = result[0], result[1]
            # Count actual propagation iterations, not just 1
            iteration_count = result[2] if len(result) > 2 else 1
            for _ in range(iteration_count):
                self.metrics_collector.increment_propagation()
        else:
            valid = True

        if not valid:
            self.metrics_collector.end_solve(False, False)
            return None, self.metrics_collector.metrics_list[-1]

        if self.constraints.is_complete():
            solution = self.constraints.get_solution_string()
            is_valid = self.constraints.validate_solution()
            self.metrics_collector.end_solve(True, is_valid)
            return solution, self.metrics_collector.metrics_list[-1]

        success = self._solve_recursive()

        if success:
            solution = self.constraints.get_solution_string()
            is_valid = self.constraints.validate_solution()
            self.metrics_collector.end_solve(True, is_valid)
            return solution, self.metrics_collector.metrics_list[-1]
        else:
            self.metrics_collector.end_solve(False, False)
            return None, self.metrics_collector.metrics_list[-1]

    def _solve_recursive(self) -> bool:
        if self.constraints.is_complete():
            return True

        if self.metrics_collector._current_metrics.expanded_nodes >= self.max_nodes:
            return False

        self.metrics_collector.increment_nodes()

        mrv_cells = self.constraints.get_mrv_cells()

        if not mrv_cells:
            return False

        cell = mrv_cells[0]
        if self.use_degree_tiebreak and len(mrv_cells) > 1:
            cell = self._select_degree_tiebreak_cell(mrv_cells)
        candidates = self.constraints.get_candidates(cell)

        if self.use_lcv:
            # Sort by LCV score in descending order (least constraining value first)
            # Higher LCV score = appears in more neighbors' candidate sets = less constraining
            candidates = sorted(candidates,
                                key=lambda d: self.constraints.compute_lcv(cell, d),
                                reverse=True)

        if not candidates:
            return False

        for digit in candidates:
            if self._start_time is not None and (time.time() - self._start_time) * 1000 > self._timeout_ms:
                return False

            state = self.constraints.save_state()
            valid = False

            if self.constraints.assign(cell, digit):
                if self.use_fc:
                    result = self.constraints.propagate()
                    changed, valid = result[0], result[1]
                    # Count actual propagation iterations
                    iteration_count = result[2] if len(result) > 2 else 1
                    for _ in range(iteration_count):
                        self.metrics_collector.increment_propagation()
                else:
                    valid = True

            if valid:
                    if self._solve_recursive():
                        return True

            self.constraints.restore_state(state)
            self.metrics_collector.increment_backtracks()

        return False

    def _parse_puzzle(self, puzzle: str) -> np.ndarray:
        puzzle = puzzle.strip().replace('.', '0')
        return np.array([int(c) if c.isdigit() else 0 for c in puzzle[:81]], dtype=np.int64)

    def _select_degree_tiebreak_cell(self, mrv_cells: List[int]) -> int:
        # Degree tie-break on MRV ties: prefer the variable with more unassigned neighbors.
        best_cell = mrv_cells[0]
        best_degree = -1
        for cell in mrv_cells:
            degree = len(self.constraints.get_neighbors(cell))
            if degree > best_degree:
                best_degree = degree
                best_cell = cell
        return best_cell

    def solve_batch(self, puzzles: List[str]) -> List[tuple]:
        results = []
        for puzzle in puzzles:
            solution, metrics = self.solve(puzzle)
            results.append((solution, metrics))
        return results

    def get_benchmark_results(self):
        return self.metrics_collector.compute_benchmark_results()

    def clear_metrics(self):
        self.metrics_collector.clear()
