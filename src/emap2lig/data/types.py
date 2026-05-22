from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import gemmi
import numpy as np
from jaxtyping import Float, Num

from loguru import logger


####################################################################################################
# SERIALIZABLE
####################################################################################################


@dataclass(frozen=True)
class NumpySerializable:
    """Serializable datatype."""

    @classmethod
    def load(cls: type["NumpySerializable"], path: Path) -> "NumpySerializable":
        """Load the object from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        return cls(**np.load(path, allow_pickle=True))

    def dump(self, path: Path) -> None:
        """Dump the object to an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        np.savez_compressed(str(path), **asdict(self))


####################################################################################################
# Cryo-EM MAP
####################################################################################################


@dataclass(frozen=True)
class MapObject(NumpySerializable):
    """Cryo-EM map object."""

    grid_data: Num[np.ndarray, "d h w"]
    voxel_size: Float[np.ndarray, "3"]
    global_origin: Float[np.ndarray, "3"]
    emdb_id: str | None = None

    def __repr__(self) -> str:
        """Get string representation of the MRCObject.

        Returns:
            String representation of the object.
        """
        return f"Map(grid_data of shape {self.grid_data.shape}: voxel_size={self.voxel_size}, global_origin={self.global_origin})"

    @property
    def grid_size(self) -> tuple[int, int, int]:
        """Get the shape of the grid data.

        Returns:
            Tuple of (height, width, depth) dimensions.
        """
        shape = self.grid_data.shape
        if len(shape) != 3:
            raise ValueError("Grid data must be 3-dimensional")
        return (int(shape[0]), int(shape[1]), int(shape[2]))

    @property
    def is_empty(self) -> bool:
        """Check if the voxel data is effectively empty (all values close to zero)."""
        return bool(np.allclose(self.grid_data, 0, atol=1e-12))

    def __mul__(self, other: "MapObject") -> "MapObject":
        """Multiply two maps element-wise."""
        # assert voxel size and global origin are the same
        assert np.allclose(self.voxel_size, other.voxel_size)
        assert np.allclose(self.global_origin, other.global_origin)
        return MapObject(
            grid_data=self.grid_data * other.grid_data,
            voxel_size=self.voxel_size,
            global_origin=self.global_origin,
        )


####################################################################################################
# STRUCTURE
####################################################################################################


class CIFObject:
    """Class for handling CIF (Crystallographic Information File) objects.

    This class provides functionality for working with structural data from CIF files,
    including handling atomic coordinates, residues, and various structural analyses.
    """

    def __init__(self, structure: gemmi.Structure, pdb_id: str | None = None):
        """Initialize CIFObject.

        Args:
            structure: GEMMI Structure object.
            pdb_id: Optional PDB ID for the structure.
        """
        self.structure = structure
        self.pdb_id = pdb_id
        self._safe_scan()

    def __repr__(self) -> str:
        """Return string representation of CIFObject.

        Returns:
            String representation including PDB ID if available.
        """
        return f"CIFObject(pdb_id={self.pdb_id}, structure={self.structure})"

    def _safe_scan(self):
        # Check number of models
        if len(self.structure) > 1:
            logger.warning(f"Multiple models found in {self.pdb_id}.")

        # Check entity type
        for entity in self.structure.entities:
            # for polymer, only accept DNA, RNA, Protein
            if entity.entity_type == gemmi.EntityType.Polymer:
                if entity.polymer_type not in [
                    gemmi.PolymerType.Dna,
                    gemmi.PolymerType.Rna,
                    gemmi.PolymerType.PeptideD,
                    gemmi.PolymerType.PeptideL,
                ]:
                    logger.warning(
                        f"Unsupported polymer type {entity.polymer_type} found in {self.pdb_id}.",
                    )
            elif entity.entity_type in [gemmi.EntityType.Unknown]:
                logger.warning(
                    f"Unsupported entity type {entity.entity_type} found in {self.pdb_id}.",
                )

    @property
    def spatial_size(self) -> tuple[float, ...]:
        """Get spatial size of the structure.

        Returns:
            Tuple of floats representing the spatial size.
        """
        position_box = self.structure.calculate_box()
        return tuple(
            float(x) for x in (position_box.maximum - position_box.minimum).tolist()
        )

    @property
    def num_UNK(self) -> int:
        """Get number of unknown residues in the structure.

        Returns:
            Number of unknown residues.
        """
        return sum(
            residue.name == "UNK"
            for model in self.structure
            for chain in model
            for residue in chain
        )

    @property
    def num_residues(self) -> int:
        """Get total number of residues in the structure.

        Returns:
            Total number of residues.
        """
        return sum(len(chain) for model in self.structure for chain in model)

    @property
    def num_chains(self) -> int:
        """Get number of chains in the structure.

        Returns:
            Number of chains.
        """
        return len(self.structure[0])  # only consider the first model


####################################################################################################
# FEATURES
####################################################################################################

# Numpy structured array definitions
Atom = [
    ("name", np.dtype("4i1")),  # 4-char atom name
    ("element", np.dtype("i1")),  # Atomic number
    ("charge", np.dtype("i1")),  # Formal charge
    ("coords", np.dtype("3f4")),  # 3D coordinates
    ("ref_pos", np.dtype("3f4")),  # Reference position
    ("is_present", np.dtype("?")),  # Whether atom exists in structure
    ("chirality", np.dtype("7?")),  # One-hot chirality type
    ("in_ring", np.dtype("4?")),  # One-hot ring membership (3,4,5,6)
    ("residue_id", np.dtype("i4")),  # Residue ID for multi-residue entities
]

Bond = [
    ("atom_1", np.dtype("i4")),  # First atom index
    ("atom_2", np.dtype("i4")),  # Second atom index
    (
        "type",
        np.dtype("5?"),
    ),  # One-hot bond type (single,double,triple,dative,aromatic)
    ("in_ring", np.dtype("4?")),  # One-hot ring membership (3,4,5,6)
]


@dataclass(frozen=True)
class DensityObject(NumpySerializable):
    """Ligand object datatype containing density information."""

    object_id: int  # Object ID
    # Density info
    density_grid: np.ndarray  # Density map (d,h,w)
    instance_grid: np.ndarray  # Instance map (d,h,w)
    voxel_size: np.ndarray  # Voxel size (3,)
    global_origin: np.ndarray  # Global origin (3,)


@dataclass(frozen=True)
class LigandObject(NumpySerializable):
    """Reference molecular object datatype containing density information."""

    smiles: str  # SMILES string
    atom_names: list[str]  # Atom names
    atoms: np.ndarray  # Atom array with dtype=Atom
    bonds: np.ndarray  # Bond array with dtype=Bond
    name: str  # Ligand name, LIGx, or CCD code
    residue_names: list[
        str
    ]  # Residue names (e.g., ["NAG", "NAG"] for branched entities)
    symmetries: list  # Symmetry operations
    blobs: list[int] | None = None  # Blob indices for the ligand record


@dataclass(frozen=True)
class LigandRecord:
    """Ligand record datatype."""

    type: Literal["SMILES", "BRANCHED", "CCD"]  # Type of ligand (SMILES, BRANCHED, CCD)
    name: str  # Ligand name, LIGx, or CCD code
    blobs: list[int] | None = None  # Blob indices for the ligand record
    smiles: str | None = None  # SMILES string for the ligand record
    residues: dict[int, str] | None = None  # Residue names for the ligand record
    bonds: list[tuple[int, str, int, str]] | None = None  # Bonds for the ligand record
