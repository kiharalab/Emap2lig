"""On-disk layout of a pipeline output directory.

::

    <root>/
    ├── preprocess/unified.mrc, ligands/<NAME>.npz
    ├── find_maps/<label>.mrc, <label>_mask.mrc
    ├── find_blobs/blob_<i>.npz, mask_<i>.mrc
    └── build_struct/blob_<i>/..., best/
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputLayout:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def preprocess(self) -> Path:
        return self.root / "preprocess"

    @property
    def unified_map(self) -> Path:
        return self.preprocess / "unified.mrc"

    @property
    def ligands(self) -> Path:
        return self.preprocess / "ligands"

    @property
    def find_maps(self) -> Path:
        return self.root / "find_maps"

    def label_map(self, label: str) -> Path:
        return self.find_maps / f"{label}.mrc"

    def label_mask(self, label: str) -> Path:
        return self.find_maps / f"{label}_mask.mrc"

    @property
    def find_blobs(self) -> Path:
        return self.root / "find_blobs"

    def blob(self, blob_id: int) -> Path:
        return self.find_blobs / f"blob_{blob_id}.npz"

    def blob_mask(self, blob_id: int) -> Path:
        return self.find_blobs / f"mask_{blob_id}.mrc"

    def blob_paths(self, blob_ids: list[int] | None = None) -> list[Path]:
        """Detected blob files sorted by name, optionally restricted to *blob_ids*."""
        paths = sorted(self.find_blobs.glob("blob_*.npz"))
        if blob_ids is not None:
            allowed = set(blob_ids)
            paths = [p for p in paths if blob_id_of(p) in allowed]
        return paths

    @property
    def build_struct(self) -> Path:
        return self.root / "build_struct"

    @property
    def best(self) -> Path:
        return self.build_struct / "best"


def blob_id_of(path: Path) -> int:
    """``.../blob_7.npz`` or ``.../blob_7`` → ``7``."""
    return int(Path(path).stem.split("_")[1])
