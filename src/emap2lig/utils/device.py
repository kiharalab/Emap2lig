"""Accelerator selection: CUDA on Linux, MPS on macOS, nothing else."""

import platform
import sys

import torch

UNSUPPORTED_MESSAGE = (
    "No supported accelerator is available. Emap2lig inference supports "
    "Linux/CUDA and macOS/MPS only."
)


def _mps_available() -> bool:
    return bool(
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _macos_version_at_least(major: int, minor: int) -> bool:
    version = platform.mac_ver()[0]
    if not version:
        return False
    parts = version.split(".")
    try:
        current = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return False
    return current >= (major, minor)


def resolve_device(gpu: int = 0) -> torch.device:
    """Return the inference device for *gpu*, or raise if it is unusable.

    Raises:
        ValueError: The device ID is invalid for this platform.
        RuntimeError: No supported accelerator is available.
    """
    if sys.platform.startswith("linux"):
        if gpu < 0:
            raise ValueError(
                "CPU inference is not supported. Please provide --gpu with a CUDA "
                "device ID (0..N-1)."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available on this Linux environment. Emap2lig "
                "requires an NVIDIA GPU (CUDA 12/13) on Linux."
            )
        device_count = torch.cuda.device_count()
        if gpu >= device_count:
            raise ValueError(
                f"Invalid GPU device ID {gpu}. Available CUDA devices: 0..{device_count - 1}."
            )
        return torch.device(f"cuda:{gpu}")

    if sys.platform == "darwin":
        if gpu not in (0, -1):
            raise ValueError("MPS exposes a single device. Use --gpu 0 for MPS.")
        if not _mps_available():
            raise RuntimeError(
                "Apple MPS is not available on this macOS environment. Emap2lig "
                "requires MPS on macOS."
            )
        if not _macos_version_at_least(13, 2):
            raise RuntimeError(
                "MPS Conv3d inference requires macOS 13.2 or newer. Please "
                "upgrade macOS to run Emap2lig on MPS."
            )
        return torch.device("mps")

    raise RuntimeError(
        f"Unsupported platform {sys.platform!r}. Emap2lig inference supports "
        "Linux/CUDA and macOS/MPS only."
    )


def trainer_device_kwargs(device: torch.device) -> dict[str, object]:
    """Lightning ``Trainer`` accelerator/devices arguments for *device*."""
    if device.type == "cuda":
        return {"accelerator": "gpu", "devices": [device.index or 0]}
    if device.type == "mps":
        return {"accelerator": "mps", "devices": 1}
    raise RuntimeError(f"Unsupported inference device: {device}")


def list_accelerators() -> dict:
    """Describe the usable accelerators on this machine (for the web GUI)."""
    info: dict = {
        "cuda_available": False,
        "mps_available": False,
        "accelerator": None,
        "gpus": [],
    }
    if sys.platform.startswith("linux") and torch.cuda.is_available():
        info.update(cuda_available=True, accelerator="cuda")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {
                    "id": i,
                    "name": props.name,
                    "memory_mb": props.total_memory // (1024 * 1024),
                }
            )
    elif _mps_available() and _macos_version_at_least(13, 2):
        info.update(mps_available=True, accelerator="mps")
        info["gpus"].append({"id": 0, "name": "Apple MPS", "memory_mb": 0})
    return info


def validate_device_ids(gpu_ids: list[int]) -> None:
    """Raise ``ValueError`` unless *gpu_ids* is a valid selection on this machine."""
    info = list_accelerators()
    if not info["gpus"]:
        raise ValueError(UNSUPPORTED_MESSAGE)
    if not gpu_ids:
        raise ValueError("No accelerator selected. Please select a device in Setup.")

    available_ids = {gpu["id"] for gpu in info["gpus"]}
    invalid_ids = sorted({gpu_id for gpu_id in gpu_ids if gpu_id not in available_ids})
    if invalid_ids:
        available_str = ", ".join(str(i) for i in sorted(available_ids))
        invalid_str = ", ".join(str(i) for i in invalid_ids)
        raise ValueError(
            f"Invalid accelerator id(s): {invalid_str}. Available id(s): {available_str}."
        )
    if info["accelerator"] == "mps" and len(gpu_ids) > 1:
        raise ValueError("MPS exposes a single device. Select only device 0.")
