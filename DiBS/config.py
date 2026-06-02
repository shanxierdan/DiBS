from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "model" / "diffusion-vs-ar" / "output" / "sudoku" / "checkpoint"
)


@dataclass
class DiBSConfig:
    checkpoint_path: str = field(
        default_factory=lambda: os.environ.get(
            "DIBS_CHECKPOINT_PATH",
            str(DEFAULT_CHECKPOINT_PATH),
        )
    )

    device: str = field(default_factory=lambda: os.environ.get("DIBS_DEVICE", "cuda"))

    use_heuristic: bool = field(
        default_factory=lambda: os.environ.get("DIBS_USE_HEURISTIC", "true").lower() == "true"
    )

    use_lcv: bool = field(
        default_factory=lambda: os.environ.get("DIBS_USE_LCV", "false").lower() == "true"
    )

    alpha: float = field(
        default_factory=lambda: float(os.environ.get("DIBS_ALPHA", "0.8"))
    )

    beta: float = field(
        default_factory=lambda: float(os.environ.get("DIBS_BETA", "0.5"))
    )

    diffusion_timestep: int = field(
        default_factory=lambda: int(os.environ.get("DIBS_TIMESTEP", "10"))
    )

    denoise_steps: int = field(
        default_factory=lambda: int(os.environ.get("DIBS_DENOISE_STEPS", "1"))
    )

    denoise_fill_ratio: float = field(
        default_factory=lambda: float(os.environ.get("DIBS_DENOISE_FILL_RATIO", "0.15"))
    )

    denoise_strategy: str = field(
        default_factory=lambda: os.environ.get("DIBS_DENOISE_STRATEGY", "legacy_repeat")
    )

    mdm_decoding_strategy: str = field(
        default_factory=lambda: os.environ.get("DIBS_MDM_DECODING_STRATEGY", "deterministic-cosine")
    )

    max_nodes: int = field(
        default_factory=lambda: int(os.environ.get("DIBS_MAX_NODES", "100000"))
    )

    verbose: bool = field(
        default_factory=lambda: os.environ.get("DIBS_VERBOSE", "false").lower() == "true"
    )

    def __post_init__(self):
        self.alpha = max(0.0, min(1.0, self.alpha))
        self.beta = max(0.0, min(1.0, self.beta))
        self.diffusion_timestep = max(0, min(19, self.diffusion_timestep))
        self.denoise_steps = max(1, self.denoise_steps)
        self.denoise_fill_ratio = max(0.0, min(1.0, self.denoise_fill_ratio))
        if self.denoise_strategy not in {"legacy_repeat", "mdm_iterative"}:
            self.denoise_strategy = "legacy_repeat"
