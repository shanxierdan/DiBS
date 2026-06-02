"""
Configuration for Sudoku extension modules.
All features are disabled by default to ensure baseline compatibility.
Supports environment variable configuration for multi-process training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Literal


def _get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    val = os.environ.get(name, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def _get_env_float(name: str, default: float = 0.0) -> float:
    """Get float value from environment variable."""
    val = os.environ.get(name)
    return float(val) if val is not None else default


def _get_env_int(name: str, default: int = 0) -> int:
    """Get int value from environment variable."""
    val = os.environ.get(name)
    return int(val) if val is not None else default


def _get_env_str(name: str, default: str = "") -> str:
    """Get string value from environment variable."""
    return os.environ.get(name, default)


@dataclass
class SudokuConfig:
    """
    Configuration for Sudoku enhancement modules.

    All features are disabled by default to maintain baseline compatibility.
    Configuration can be set via environment variables (SUDOKU_*) for multi-process training.
    """

    # Module A: Constraint Energy
    enable_constraint_energy: bool = field(default_factory=lambda: _get_env_bool("SUDOKU_ENABLE_ENERGY", False))
    """Whether to enable constraint energy during training."""

    lambda_e: float = field(default_factory=lambda: _get_env_float("SUDOKU_LAMBDA_E", 0.5))
    """Weight for constraint energy term in loss."""

    energy_type: str = field(default_factory=lambda: _get_env_str("SUDOKU_ENERGY_TYPE", "over"))
    """Type of energy function: 'over' for overcounting only, 'dup' for both over and undercounting."""

    t_e_ratio: float = field(default_factory=lambda: _get_env_float("SUDOKU_T_E_RATIO", 0.3))
    """Ratio of timesteps to apply energy (only apply in last t_e_ratio * T steps)."""

    # Module B: Guided TopK Decoding
    enable_guided_topk: bool = field(default_factory=lambda: _get_env_bool("SUDOKU_ENABLE_GUIDED", False))
    """Whether to enable guided topk decoding during inference."""

    alpha_u: float = field(default_factory=lambda: _get_env_float("SUDOKU_ALPHA_U", 1.0))
    """Weight for uncertainty term in guided topk scoring."""

    beta_c: float = field(default_factory=lambda: _get_env_float("SUDOKU_BETA_C", 1.0))
    """Weight for conflict term in guided topk scoring."""

    gamma_m: float = field(default_factory=lambda: _get_env_float("SUDOKU_GAMMA_M", 0.0))
    """Weight for mask priority term in guided topk scoring (if used)."""

    k_max: int = field(default_factory=lambda: _get_env_int("SUDOKU_K_MAX", 81))
    """Maximum number of positions to update per step (at start of diffusion)."""

    k_min: int = field(default_factory=lambda: _get_env_int("SUDOKU_K_MIN", 12))
    """Minimum number of positions to update per step (at end of diffusion)."""

    use_dynamic_k: bool = field(default_factory=lambda: _get_env_bool("SUDOKU_USE_DYNAMIC_K", True))
    """Whether to use dynamic K scheduling (linearly decreasing)."""

    t_conf_ratio: Optional[float] = field(default_factory=lambda: _get_env_float("SUDOKU_T_CONF_RATIO", -1) if _get_env_float("SUDOKU_T_CONF_RATIO", -1) >= 0 else None)
    """Optional ratio of timesteps to enable conflict term (if None, always enabled)."""

    # General settings
    verbose: bool = field(default_factory=lambda: _get_env_bool("SUDOKU_VERBOSE", False))
    """Whether to print debug/logging information."""

    # Internal state (not meant to be set by user)
    _constraints_initialized: bool = field(default=False, init=False, repr=False)
    """Whether constraints have been initialized."""

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.enable_constraint_energy:
            assert self.lambda_e >= 0, f"lambda_e must be >= 0, got {self.lambda_e}"
            assert self.energy_type in ["over", "dup"], \
                f"energy_type must be 'over' or 'dup', got {self.energy_type}"
            assert 0 <= self.t_e_ratio <= 1, \
                f"t_e_ratio must be between 0 and 1, got {self.t_e_ratio}"

        if self.enable_guided_topk:
            assert self.alpha_u >= 0, f"alpha_u must be >= 0, got {self.alpha_u}"
            assert self.beta_c >= 0, f"beta_c must be >= 0, got {self.beta_c}"
            assert self.gamma_m >= 0, f"gamma_m must be >= 0, got {self.gamma_m}"
            assert 0 < self.k_min <= self.k_max <= 81, \
                f"k_min ({self.k_min}) <= k_max ({self.k_max}) <= 81 required"
            if self.t_conf_ratio is not None:
                assert 0 <= self.t_conf_ratio <= 1, \
                    f"t_conf_ratio must be between 0 and 1, got {self.t_conf_ratio}"

    def get_dynamic_k(self, t: int, T: int) -> int:
        """
        Compute dynamic K value based on timestep.

        Args:
            t: Current timestep (0 <= t < T)
            T: Total timesteps

        Returns:
            Number of positions to update at timestep t.
        """
        if not self.use_dynamic_k or T <= 1:
            return self.k_max

        # Linear schedule from k_max to k_min
        ratio = t / (T - 1)
        k = self.k_max - (self.k_max - self.k_min) * ratio
        return int(k + 0.5)  # Round to nearest integer

    def get_omega_t(self, t: int, T: int) -> float:
        """
        Compute time gating factor for constraint energy.

        Args:
            t: Current timestep (small t means close to final step)
            T: Total timesteps

        Returns:
            Weight factor for constraint energy (0 or 1).
        """
        t_e = int(self.t_e_ratio * T)
        return 1.0 if t <= t_e else 0.0

    def get_conflict_weight(self, t: int, T: int) -> float:
        """
        Compute conflict term weight with optional timestep gating.

        Args:
            t: Current timestep
            T: Total timesteps

        Returns:
            Effective beta_c weight considering t_conf_ratio.
        """
        if self.t_conf_ratio is None:
            return self.beta_c

        t_conf = int(self.t_conf_ratio * T)
        return self.beta_c if t <= t_conf else 0.0


# Global default configuration - reads from environment variables at import time
_default_config = SudokuConfig()

def get_config() -> SudokuConfig:
    """Get the global default configuration."""
    return _default_config

def update_config(**kwargs) -> None:
    """
    Update global configuration with new values.

    Args:
        **kwargs: Configuration parameters to update.
    """
    for key, value in kwargs.items():
        if hasattr(_default_config, key):
            setattr(_default_config, key, value)
        else:
            raise AttributeError(f"SudokuConfig has no attribute '{key}'")
    _default_config.validate()

def reset_config() -> None:
    """Reset configuration to default values (all features disabled)."""
    global _default_config
    _default_config = SudokuConfig()
