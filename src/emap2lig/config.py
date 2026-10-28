"""Runtime configuration for the Emap2lig pipeline.

Architecture hyperparameters are fixed by the released checkpoints and live in
:mod:`emap2lig.model.pretrained`.  The options below are what users may change;
the Typer CLI and the web GUI expose them with these defaults.
"""

from dataclasses import dataclass

# Fixed by the checkpoints: segmentation sliding-window size, its output
# channels, and the crop size of density blobs fed to the structure model.
DETECTION_SPATIAL_SIZE = 64
DETECTION_LABELS = ("backbone", "sidechain", "sugar", "phosphate", "base", "ligand")
STRUCTURE_SPATIAL_SIZE = 48


@dataclass
class InferenceConfig:
    """User-adjustable runtime options."""

    gpu: int = 0
    # Find stage
    detection_batch_size: int = 16
    contour_level: float | None = None
    extended_val: int = 16
    # Build stage
    multiplicity: int = 1
    max_parallel_multiplicity: int = 8
    num_sampling_steps: int = 100
    num_workers: int = 0

    def __post_init__(self) -> None:
        for name in (
            "detection_batch_size",
            "multiplicity",
            "max_parallel_multiplicity",
            "num_sampling_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
