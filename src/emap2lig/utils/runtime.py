"""Process-wide numeric and library settings for inference."""

import random

import numpy as np
import torch

_configured = False


def configure_runtime() -> None:
    """Enable TF32 matmuls and raise torch.compile cache limits (idempotent).

    Called by the pipeline stages before building models, so CLI, web and
    programmatic callers all run with the same numerics.
    """
    global _configured
    if _configured:
        return
    _configured = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch._dynamo.config.cache_size_limit = 2048
        torch._dynamo.config.accumulated_cache_size_limit = 2048


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
