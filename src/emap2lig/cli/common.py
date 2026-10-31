"""Helpers shared by the Typer commands."""

import logging
from collections.abc import Callable

import typer
from tqdm import tqdm

from ..config import InferenceConfig
from ..errors import Emap2ligError
from ..progress import ProgressCallback, ProgressEvent, STAGE_LABELS
from ..utils.device import resolve_device

logger = logging.getLogger("emap2lig.cli")

DEFAULTS = InferenceConfig()


def make_config(**options) -> InferenceConfig:
    """Validated :class:`InferenceConfig`; bad values become usage errors."""
    try:
        cfg = InferenceConfig(**options)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        resolve_device(cfg.gpu)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--gpu") from exc
    return cfg


def run_stage[T](stage: Callable[..., T], *args, **kwargs) -> T:
    """Call a pipeline stage, turning :class:`Emap2ligError` into exit code 1."""
    try:
        return stage(*args, **kwargs)
    except Emap2ligError as exc:
        logger.error(str(exc))
        raise typer.Exit(1) from exc


def make_tqdm_callback() -> ProgressCallback:
    """Render pipeline :class:`~emap2lig.progress.ProgressEvent`s as tqdm bars.

    One bar is kept per stage. Completing a stage closes its bar.
    """
    bars: dict[str, tqdm] = {}

    def callback(event: ProgressEvent) -> None:
        bar = bars.get(event.stage)
        if bar is None:
            bar = tqdm(
                total=event.total,
                desc=STAGE_LABELS[event.stage],
                unit="step",
                leave=True,
            )
            bars[event.stage] = bar
        if event.total is not None and bar.total != event.total:
            bar.total = event.total
            bar.refresh()
        delta = event.current - bar.n
        if delta:
            bar.update(delta)
        postfix = event.message
        if event.failed:
            postfix = f"{postfix} ({event.failed} failed)"
        bar.set_postfix_str(postfix, refresh=True)
        if event.total is not None and event.total > 0 and event.current >= event.total:
            bar.close()

    return callback
