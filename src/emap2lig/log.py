"""Root-logger configuration for Emap2lig entry points.

Library modules only create ``logging.getLogger(__name__)``; call
:func:`setup_logging` once from a CLI entry point, never from library code.
"""

import logging

from rich.logging import RichHandler

# Third-party loggers that are too chatty at INFO (e.g. one line per HTTP request).
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Replace root-logger handlers with a single rich console handler.

    Args:
        level: Minimum level shown on the console.

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)

    handler = RichHandler(show_time=False)
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return root
