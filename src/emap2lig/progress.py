"""Structured pipeline progress events.

Library stages call :func:`emit` with a :class:`ProgressEvent`. Callers attach
an :data:`ProgressCallback` (CLI tqdm bar, web job store, tests). A callback
may raise :class:`~emap2lig.errors.JobCancelledError` to stop the stage at the
next event.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Stage = Literal[
    "preprocess",
    "segment",
    "split_blobs",
    "prepare_ligands",
    "load_model",
    "predict",
    "rank",
]

DETECT_STAGES: tuple[Stage, ...] = ("preprocess", "segment", "split_blobs")
BUILD_STAGES: tuple[Stage, ...] = (
    "prepare_ligands",
    "load_model",
    "predict",
    "rank",
)

STAGE_LABELS: dict[Stage, str] = {
    "preprocess": "Preprocess",
    "segment": "Segment",
    "split_blobs": "Split blobs",
    "prepare_ligands": "Prepare ligands",
    "load_model": "Load model",
    "predict": "Predict",
    "rank": "Rank",
}


@dataclass(frozen=True)
class ProgressEvent:
    """One progress update from a pipeline stage.

    Args:
        stage: Named pipeline stage.
        current: Completed items in this stage (0 at start).
        total: Expected items, or ``None`` when unknown.
        message: Human-readable status line.
        failed: Running count of failed items in this stage.
    """

    stage: Stage
    current: int
    total: int | None
    message: str
    failed: int = 0


ProgressCallback = Callable[[ProgressEvent], None]


def emit(
    callback: ProgressCallback | None,
    stage: Stage,
    current: int,
    total: int | None,
    message: str,
    *,
    failed: int = 0,
) -> None:
    """Send a :class:`ProgressEvent` when a callback is attached.

    Args:
        callback: Receiver, or ``None`` to no-op.
        stage: Named pipeline stage.
        current: Completed items in this stage.
        total: Expected items, or ``None`` when unknown.
        message: Human-readable status line.
        failed: Running count of failed items in this stage.

    Raises:
        emap2lig.errors.JobCancelledError: Propagated from *callback*.
    """
    if callback is None:
        return
    callback(
        ProgressEvent(
            stage=stage,
            current=current,
            total=total,
            message=message,
            failed=failed,
        )
    )
