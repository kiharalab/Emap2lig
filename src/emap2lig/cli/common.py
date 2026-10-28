"""Helpers shared by the Typer commands."""

import logging
from collections.abc import Callable

import typer

from ..config import InferenceConfig
from ..errors import Emap2ligError
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
