"""Compute conformers and symmetries for all the CCD molecules."""

import pickle
from pathlib import Path

import rdkit
from rdkit import Chem
from pdbeccdutils.core.component import ConformerType
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Conformer, Mol
from huggingface_hub import hf_hub_download

_MODEL_DIR = str(Path.home() / ".emap2lig" / "models")


def get_ccd_dict(date: str = "250523"):
    local_path = Path(_MODEL_DIR) / f"ccd/ccd_dict_{date}.pkl"
    if local_path.exists():
        ccd_dict_path = str(local_path)
    else:
        ccd_dict_path = hf_hub_download(
            repo_id="KiharaLab/Emap2lig",
            filename=f"ccd/ccd_dict_{date}.pkl",
            local_dir=_MODEL_DIR,
        )
    ccd_dict = pickle.load(open(ccd_dict_path, "rb"))
    return ccd_dict


def compute_3d(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults ot v3

    Returns
    -------
    bool
        Whether computation was successful.

    """
    if version == "v3":
        options = rdkit.Chem.AllChem.ETKDGv3()
    elif version == "v2":
        options = rdkit.Chem.AllChem.ETKDGv2()
    else:
        options = rdkit.Chem.AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = rdkit.Chem.AllChem.EmbedMolecule(mol, options)
        rdkit.Chem.AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", ConformerType.Computed.name)
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False


def get_conformer(mol: Mol, c_type: ConformerType) -> Conformer:
    """Retrieve an rdkit object for a deemed conformer.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The molecule to process.
    c_type: ConformerType
        The conformer type to extract.

    Returns
    -------
    Conformer
        The desired conformer, if any.

    Raises
    ------
    ValueError
        If there are no conformers of the given tyoe.

    """
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == c_type.name:
                return c
        except KeyError:
            pass

    msg = f"Conformer {c_type.name} does not exist."
    raise ValueError(msg)


def compute_symmetries(mol: Mol) -> list[list[int]]:
    """Compute the symmetries of a molecule.

    Parameters
    ----------
    mol : Mol
        The molecule to process

    Returns
    -------
    list[list[int]]
        The symmetries as a list of index permutations

    """
    mol = AllChem.RemoveHs(mol)
    idx_map = {}
    atom_idx = 0
    for i, atom in enumerate(mol.GetAtoms()):
        # Skip if leaving atoms
        if int(atom.GetProp("leaving_atom")):
            continue
        idx_map[i] = atom_idx
        atom_idx += 1

    # Calculate self permutations
    permutations = []
    raw_permutations = mol.GetSubstructMatches(mol, uniquify=False)
    for raw_permutation in raw_permutations:
        # Filter out permutations with leaving atoms
        try:
            if {raw_permutation[idx] for idx in idx_map} == set(idx_map.keys()):
                permutation = [
                    idx_map[idx] for idx in raw_permutation if idx in idx_map
                ]
                permutations.append(permutation)
        except Exception:
            pass
    serialized_permutations = pickle.dumps(permutations)
    mol.SetProp("symmetries", serialized_permutations.hex())
    return permutations


def add_conformer(mol: Mol) -> tuple[str, str]:
    """Process a CCD component.

    Parameters
    ----------
    mol : Mol
        The molecule to process
    output : str
        The directory to save the molecules

    Returns
    -------
    str
        The name of the component
    str
        The result of the conformer generation

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


def get_conformer_from_smiles(smiles: str) -> Mol:
    """Get a conformer from a smiles string.

    Parameters
    ----------
    smiles : str
        The smiles string to process

    Returns
    -------
    Mol
        The molecule with a conformer

    """
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)

    # Set atom names
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            raise ValueError(
                f"{smiles} has an atom with a name longer than 4 characters: {atom_name}"
            )
        atom.SetProp("name", atom_name)

    mol = AllChem.RemoveHs(mol)
    mol = add_conformer(mol)
    return mol
