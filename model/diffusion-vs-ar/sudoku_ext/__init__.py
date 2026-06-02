"""
Sudoku extension modules for diffusion-vs-ar project.
Implements constraint energy and guided topk decoding.
"""

from .constraints import SudokuConstraints, get_constraints, get_groups, get_cell_to_groups
from .energy import ConstraintEnergy, compute_energy, add_energy_to_loss
from .guided_topk import GuidedTopK, compute_guided_scores, guided_topk_selection
from .config import SudokuConfig, get_config, update_config, reset_config

__all__ = ['SudokuConstraints', 'get_constraints', 'get_groups', 'get_cell_to_groups',
           'ConstraintEnergy', 'compute_energy', 'add_energy_to_loss',
           'GuidedTopK', 'compute_guided_scores', 'guided_topk_selection',
           'SudokuConfig', 'get_config', 'update_config', 'reset_config']