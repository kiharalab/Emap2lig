"""Ligand records → :class:`LigandObject` (atom/bond feature arrays) on disk."""

import logging
from pathlib import Path

import numpy as np
from rdkit import Chem

from ..errors import LigandPreparationError
from .ccd import get_ccd_mol, get_conformer_from_smiles
from .const import bond_type_ids, bond_types, chirality_type_ids, chirality_types
from .types import Atom, Bond, LigandObject, LigandRecord

logger = logging.getLogger(__name__)

_RING_SIZES = (3, 4, 5, 6)
# Inter-residue links of branched ligands (glycosidic bonds) are single bonds.
_SINGLE_BOND = tuple(i == bond_type_ids["SINGLE"] for i in range(len(bond_types)))
_NO_RING = (False,) * len(_RING_SIZES)


def prepare_ligands(records: list[LigandRecord], ligands_dir: Path) -> list[Path]:
    """Featurize *records* into ``<ligands_dir>/<name>.npz``; failures are skipped."""
    ligands_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for record in records:
        try:
            ligand = build_ligand_object(record)
        except LigandPreparationError as exc:
            logger.warning(f"Skipping ligand {record.name}: {exc}")
            continue
        path = ligands_dir / f"{ligand.name}.npz"
        ligand.dump(path)
        logger.info(f"Saved ligand object for {ligand.name} to {path}")
        written.append(path)
    logger.info(f"Successfully processed {len(written)} ligands")
    return written


def build_ligand_object(record: LigandRecord) -> LigandObject:
    """Build the featurized ligand for a CCD, SMILES or BRANCHED record."""
    builders = {"CCD": _from_ccd, "SMILES": _from_smiles, "BRANCHED": _from_branched}
    builder = builders.get(record.type)
    if builder is None:
        raise LigandPreparationError(f"unrecognized ligand type {record.type!r}")
    logger.info(f"Processing {record.type} ligand {record.name}")
    try:
        return builder(record)
    except LigandPreparationError:
        raise
    except Exception as exc:  # RDKit/CCD failures must not abort the whole run
        raise LigandPreparationError(str(exc)) from exc


def _from_ccd(record: LigandRecord) -> LigandObject:
    mol = get_ccd_mol(record.name)
    smiles = Chem.MolToSmiles(mol)
    return _single_residue_object(Chem.RemoveHs(mol, sanitize=False), record, smiles)


def _from_smiles(record: LigandRecord) -> LigandObject:
    if not record.smiles:
        raise LigandPreparationError("no SMILES string provided")
    result, mol = get_conformer_from_smiles(record.smiles)
    if result == "failed":
        raise LigandPreparationError(f"conformer generation failed for {record.smiles}")
    return _single_residue_object(mol, record, record.smiles)


def _single_residue_object(
    mol: Chem.Mol, record: LigandRecord, smiles: str
) -> LigandObject:
    atoms, atom_names = atom_features(mol, residue_id=1)
    return LigandObject(
        smiles=smiles,
        atom_names=atom_names,
        atoms=np.array(atoms, dtype=Atom),
        bonds=np.array(bond_features(mol), dtype=Bond),
        name=record.name,
        residue_names=[record.name],
        symmetries=[],
        blobs=record.blobs,
    )


def _from_branched(record: LigandRecord) -> LigandObject:
    residue_names = list((record.residues or {}).values())
    atoms: list[tuple] = []
    bonds: list[tuple] = []
    atom_names: list[str] = []
    # Global atom index of each residue's atoms, keyed by atom name.
    residue_atom_index: list[dict[str, int]] = []

    for residue_id, res_name in enumerate(residue_names, start=1):
        mol = Chem.RemoveHs(get_ccd_mol(res_name), sanitize=False)
        offset = len(atoms)
        res_atoms, res_names = atom_features(mol, residue_id=residue_id, auto_name=True)
        atoms.extend(res_atoms)
        atom_names.extend(res_names)
        bonds.extend(bond_features(mol, atom_offset=offset))
        index: dict[str, int] = {}
        for i, name in enumerate(res_names):
            index.setdefault(name, offset + i)
        residue_atom_index.append(index)

    for link in record.bonds or []:
        if len(link) != 4:
            logger.warning(f"Invalid bond format: {link}")
            continue
        res1, name1, res2, name2 = link
        if not (1 <= res1 <= len(residue_names) and 1 <= res2 <= len(residue_names)):
            logger.warning(f"Invalid residue indices in bond: {link}")
            continue
        atom1 = residue_atom_index[res1 - 1].get(name1)
        atom2 = residue_atom_index[res2 - 1].get(name2)
        if atom1 is None or atom2 is None:
            logger.warning(f"Could not find atoms for bond: {link}")
            continue
        bonds.append((atom1, atom2, _SINGLE_BOND, _NO_RING))

    return LigandObject(
        smiles="",
        atom_names=atom_names,
        atoms=np.array(atoms, dtype=Atom),
        bonds=np.array(bonds, dtype=Bond),
        name=record.name,
        residue_names=residue_names,
        symmetries=[],
        blobs=record.blobs,
    )


def encode_atom_name(name: str) -> tuple[int, int, int, int]:
    """Encode an atom name as 4 ints (``ord(c) - 32``, zero-padded; PDB style)."""
    codes = [ord(c) - 32 for c in name.strip()]
    if len(codes) > 4:
        raise ValueError(f"Atom name {name!r} exceeds 4 characters (got {len(codes)})")
    return tuple(codes + [0] * (4 - len(codes)))  # type: ignore[return-value]


def atom_features(
    mol: Chem.Mol, residue_id: int = 1, auto_name: bool = False
) -> tuple[list[tuple], list[str]]:
    """Per-atom records matching the :data:`Atom` dtype, plus the atom names.

    Atoms without a ``name`` property get ``<SYMBOL><INDEX>`` when *auto_name*
    is set, otherwise an empty name.
    """
    try:
        conformer = mol.GetConformer()
    except ValueError:
        conformer = None

    records: list[tuple] = []
    names: list[str] = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.HasProp("name"):
            name = atom.GetProp("name")
        else:
            name = f"{atom.GetSymbol()}{idx + 1}" if auto_name else ""
        names.append(name)

        chirality = [False] * len(chirality_types)
        chirality[chirality_type_ids.get(atom.GetChiralTag().name, 0)] = True
        if conformer is None:
            ref_pos = (0.0, 0.0, 0.0)
        else:
            pos = conformer.GetAtomPosition(atom.GetIdx())
            ref_pos = (pos.x, pos.y, pos.z)

        records.append(
            (
                encode_atom_name(name),
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                (0.0, 0.0, 0.0),
                ref_pos,
                False,
                tuple(chirality),
                tuple(atom.IsInRingSize(n) for n in _RING_SIZES),
                residue_id,
            )
        )
    return records, names


def bond_features(mol: Chem.Mol, atom_offset: int = 0) -> list[tuple]:
    """Per-bond records matching the :data:`Bond` dtype (indices shifted by offset)."""
    records: list[tuple] = []
    for bond in mol.GetBonds():
        bond_type = [False] * len(bond_types)
        type_id = bond_type_ids.get(bond.GetBondType().name, 0)
        if type_id < len(bond_types):
            bond_type[type_id] = True
        records.append(
            (
                bond.GetBeginAtomIdx() + atom_offset,
                bond.GetEndAtomIdx() + atom_offset,
                tuple(bond_type),
                tuple(bond.IsInRingSize(n) for n in _RING_SIZES),
            )
        )
    return records
