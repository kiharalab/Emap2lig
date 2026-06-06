"""CCD conformer utilities and on-demand local caching."""

import logging
import pickle
import shutil
import tempfile
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from pdbeccdutils.core import ccd_reader
from pdbeccdutils.core.component import ConformerType
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Chem.rdchem import Conformer, Mol

logger = logging.getLogger(__name__)

_HF_REPO_ID = "KiharaLab/Emap2lig"
_DEFAULT_CCD_DATE = "250523"
_CCD_DIR = Path.home() / ".emap2lig" / "ccd"
_LEGACY_CCD_DIR = Path.home() / ".emap2lig" / "models" / "ccd"
_RCSB_CIF_URL = "https://files.rcsb.org/ligands/download/{code}.cif"

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


class CCDFetchError(RuntimeError):
    """Raised when a CCD molecule cannot be fetched or parsed from RCSB."""


def _ensure_ccd_dir() -> None:
    """Create the CCD cache directory if it does not exist."""
    _CCD_DIR.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_bulk_dict(date: str) -> None:
    """Move legacy CCD bulk dictionary file into the new CCD root."""
    legacy_path = _LEGACY_CCD_DIR / f"ccd_dict_{date}.pkl"
    new_path = _CCD_DIR / f"ccd_dict_{date}.pkl"
    if legacy_path.exists() and not new_path.exists():
        _ensure_ccd_dir()
        shutil.move(str(legacy_path), str(new_path))
        logger.info("Migrated CCD dictionary from %s to %s", legacy_path, new_path)


def _download_bulk_dict(date: str) -> Path:
    """Download the bulk CCD dictionary and place it in the CCD cache directory.

    Args:
        date: CCD release date string used in the HuggingFace filename.

    Returns:
        Local path to the downloaded dictionary file.
    """
    target_path = _CCD_DIR / f"ccd_dict_{date}.pkl"
    if target_path.exists():
        return target_path

    downloaded_path = Path(
        hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=f"ccd/ccd_dict_{date}.pkl",
        )
    )
    shutil.copy2(downloaded_path, target_path)
    return target_path


@cache
def _load_bulk_dict(date: str = _DEFAULT_CCD_DATE) -> Mapping[str, Mol]:
    """Load the bulk CCD dictionary from local cache or HuggingFace.

    Checks for a locally cached dictionary, migrates from the legacy path
    if needed, downloads from HuggingFace as a last resort, then loads
    the pickle.

    Args:
        date: CCD release date string (default ``"250523"``).

    Returns:
        Mapping from CCD three-letter code to RDKit ``Mol``.
    """
    _ensure_ccd_dir()
    _migrate_legacy_bulk_dict(date)
    local_path = _CCD_DIR / f"ccd_dict_{date}.pkl"
    if not local_path.exists():
        local_path = _download_bulk_dict(date)
    with local_path.open("rb") as handle:
        ccd_dict = pickle.load(handle)
    return ccd_dict


def _fetch_from_rcsb(code: str) -> Mol:
    """Fetch a CCD component from the RCSB CIF endpoint.

    Downloads the per-ligand CCD CIF file and parses it with
    ``pdbeccdutils``, which sets proper atom names, leaving-atom
    flags, and Ideal/Model conformers from the CCD definition.

    Args:
        code: Normalized CCD three-letter code.

    Returns:
        Molecule with CCD atom names, leaving-atom flags, and
        Ideal/Model conformers.

    Raises:
        CCDFetchError: If the CIF download or parsing fails.
    """
    url = _RCSB_CIF_URL.format(code=code)
    try:
        response = requests.get(url, timeout=20)
    except requests.RequestException as exc:
        raise CCDFetchError(f"Network error fetching CCD {code}: {exc}") from exc
    if response.status_code != 200:
        raise CCDFetchError(
            f"RCSB returned status {response.status_code} for CCD {code}"
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
            tmp.write(response.text)
            tmp_path = tmp.name

        result = ccd_reader.read_pdb_cif_file(tmp_path, sanitize=False)
        mol = result.component.mol
    except Exception as exc:
        raise CCDFetchError(f"Failed to parse RCSB CIF for CCD {code}: {exc}") from exc
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

    if mol.GetNumAtoms() == 0:
        raise CCDFetchError(f"RCSB CIF for CCD {code} contains no atoms")

    mol.SetProp("PDB_NAME", code)
    return mol


def get_ccd_mol(code: str, date: str = _DEFAULT_CCD_DATE) -> Mol:
    """Resolve a CCD molecule from per-CCD cache, bulk dict, then RCSB.

    Lookup order:
        1. Bulk CCD dictionary (downloaded from HuggingFace).
        2. Per-CCD pickle in ``~/.emap2lig/ccd/<CODE>.pkl`` for fallback entries.
        3. RCSB CIF endpoint (parsed by ``pdbeccdutils``).

    RCSB fallback hits are persisted to the per-CCD pickle so subsequent
    lookups for CCD entries missing from the bulk dictionary are instant.

    Args:
        code: CCD three-letter code (case-insensitive, whitespace trimmed).
        date: CCD release date string used to locate the bulk dictionary.

    Returns:
        RDKit ``Mol`` with 3D coordinates and atom names.

    Raises:
        CCDFetchError: If the code cannot be resolved from any source.
    """
    normalized_code = code.strip().upper()
    if not normalized_code:
        raise CCDFetchError(f"Empty CCD code: {code!r}")

    _ensure_ccd_dir()
    bulk_dict = _load_bulk_dict(date)
    if normalized_code in bulk_dict:
        return bulk_dict[normalized_code]

    ccd_pickle = _CCD_DIR / f"{normalized_code}.pkl"
    if ccd_pickle.exists():
        with ccd_pickle.open("rb") as handle:
            return pickle.load(handle)

    mol = _fetch_from_rcsb(normalized_code)
    with ccd_pickle.open("wb") as handle:
        pickle.dump(mol, handle)
    return mol


def _etkdg_embed(mol: Mol, version: str, *, use_random_coords: bool) -> int:
    """Run ETKDG embedding followed by UFF relaxation.

    Args:
        mol: RDKit molecule to process (modified in place).
        version: ETKDG version — ``"v3"`` or ``"v2"``.
        use_random_coords: When ``True``, seed the embedder with random
            coordinates.  This helps large or charged molecules that fail
            distance-geometry initialization.

    Returns:
        Conformer id on success, or ``-1`` when embedding fails.
    """
    if version == "v3":
        options = rdDistGeom.ETKDGv3()
    elif version == "v2":
        options = rdDistGeom.ETKDGv2()
    else:
        raise ValueError(f"Unsupported ETKDG version: {version}")

    options.clearConfs = False
    options.useRandomCoords = use_random_coords

    try:
        conf_id = rdDistGeom.EmbedMolecule(mol, options)
        if conf_id == -1:
            return -1
        rdForceFieldHelpers.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)
    except (RuntimeError, ValueError):
        logger.debug(
            "ETKDG embedding failed: version=%s random_coords=%s",
            version,
            use_random_coords,
        )
        return -1

    return conf_id


def compute_3d(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using the ETKDG method.

    Adapted from ``pdbeccdutils.core.component.Component``.

    Tries the requested ETKDG version first, then retries with random
    starting coordinates, and finally falls back to ETKDGv2.

    Args:
        mol: RDKit molecule to process (modified in place).
        version: ETKDG version — ``"v3"`` or ``"v2"`` (defaults to ``"v3"``).

    Returns:
        ``True`` if a 3D conformer was successfully embedded.
    """
    versions = [version]
    if version == "v3":
        versions.append("v2")

    for etkdg_version in versions:
        for use_random_coords in (False, True):
            conf_id = _etkdg_embed(
                mol,
                etkdg_version,
                use_random_coords=use_random_coords,
            )
            if conf_id == -1:
                continue

            conformer = mol.GetConformer(conf_id)
            conformer.SetProp("name", ConformerType.Computed.name)
            conformer.SetProp("coord_generation", f"ETKDG{etkdg_version}")
            return True

    return False


def get_conformer(mol: Mol, c_type: ConformerType) -> Conformer:
    """Retrieve a conformer of the requested type.

    Adapted from ``pdbeccdutils.core.component.Component``.

    Args:
        mol: Molecule to search.
        c_type: Desired conformer type.

    Returns:
        The first conformer whose ``name`` property matches *c_type*.

    Raises:
        ValueError: If no conformer of the requested type exists.
    """
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == c_type.name:
                return c
        except KeyError:
            pass

    raise ValueError(f"Conformer {c_type.name} does not exist.")


def compute_symmetries(mol: Mol) -> list[list[int]]:
    """Compute the automorphism permutations of a molecule.

    Each permutation maps non-leaving atom indices to their symmetric
    counterparts.  The result is also serialized into a hex-encoded
    pickle stored as the ``symmetries`` property on *mol*.

    Args:
        mol: Molecule to process (modified in place).

    Returns:
        List of index permutations (one per automorphism).
    """
    mol = Chem.RemoveHs(mol)
    idx_map: dict[int, int] = {}
    atom_idx = 0
    for i, atom in enumerate(mol.GetAtoms()):
        if int(atom.GetProp("leaving_atom")):
            continue
        idx_map[i] = atom_idx
        atom_idx += 1

    permutations: list[list[int]] = []
    raw_permutations = mol.GetSubstructMatches(mol, uniquify=False)
    for raw_permutation in raw_permutations:
        try:
            if {raw_permutation[idx] for idx in idx_map} == set(idx_map.keys()):
                permutation = [
                    idx_map[idx] for idx in raw_permutation if idx in idx_map
                ]
                permutations.append(permutation)
        except IndexError:
            logger.debug("Skipping malformed symmetry permutation")
    serialized_permutations = pickle.dumps(permutations)
    mol.SetProp("symmetries", serialized_permutations.hex())
    return permutations


def add_conformer(mol: Mol) -> tuple[str, Mol]:
    """Attempt to add a 3D conformer to a molecule.

    For single-atom molecules the result is ``"single"``.  Otherwise
    an ETKDGv3 conformer is computed; if that fails the existing ideal
    coordinates are used.  If neither is available the result is
    ``"failed"``.

    Args:
        mol: Molecule to process (modified in place).

    Returns:
        Tuple of (result_tag, molecule).  *result_tag* is one of
        ``"single"``, ``"computed"``, ``"ideal"``, or ``"failed"``.
    """
    # Check if single atom
    if mol.GetNumAtoms() == 1:
        result = "single"
    else:
        # Get the 3D conformer
        try:
            # Try to generate a 3D conformer with RDKit
            success = compute_3d(mol, version="v3")
            if success:
                _ = get_conformer(mol, ConformerType.Computed)
                result = "computed"

            # Otherwise, default to the ideal coordinates
            else:
                _ = get_conformer(mol, ConformerType.Ideal)
                result = "ideal"
        except ValueError:
            result = "failed"

    # Output the results
    return result, mol


def _assign_canonical_atom_names(mol: Mol, smiles: str) -> None:
    """Assign canonical ``<SYMBOL><RANK>`` atom names to a heavy-atom molecule.

    Args:
        mol: Heavy-atom RDKit molecule (modified in place).
        smiles: Source SMILES string, included in error messages.

    Raises:
        ValueError: If an atom name exceeds 4 characters.
    """
    canonical_order = Chem.CanonicalRankAtoms(mol)
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            raise ValueError(
                f"{smiles} has an atom with a name longer than 4 characters: {atom_name}"
            )
        atom.SetProp("name", atom_name)


def get_conformer_from_smiles(smiles: str) -> tuple[str, Mol]:
    """Build a molecule from a SMILES string and generate a 3D conformer.

    Hydrogens are added for ETKDG embedding, then removed before
    canonical atom names are assigned on the heavy-atom molecule.
    Atom names longer than 4 characters raise ``ValueError``.

    Args:
        smiles: SMILES string to parse.

    Returns:
        Tuple of (result_tag, molecule) from :func:`add_conformer`.

    Raises:
        ValueError: If an atom name exceeds 4 characters.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    result, mol = add_conformer(mol)

    if result == "failed":
        return result, mol

    mol = Chem.RemoveHs(mol)
    _assign_canonical_atom_names(mol, smiles)
    return result, mol
