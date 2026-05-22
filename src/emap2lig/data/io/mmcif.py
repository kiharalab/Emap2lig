from pathlib import Path
from collections.abc import Iterator

import gemmi  # type: ignore
import ihm
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup  # type: ignore
from rdkit import Chem

from emap2lig.data.types import CIFObject, LigandObject

from loguru import logger


def _parse_mmcif(
    path: Path,
    remove_alt_conf: bool = False,
    remove_water: bool = True,
    pdb_id: str | None = None,
) -> CIFObject:
    """Parse MMCIF file and return CIFObject.

    Args:
        cif_path: Path to MMCIF file.
        remove_alt_conf: Remove alternative conformations.
        remove_water: Remove water molecules.
        pdb_id: Optional PDB ID.

    Returns:
        CIFObject instance.
    """
    cif_path = path.as_posix()
    structure = gemmi.read_structure(cif_path)
    if remove_alt_conf:
        structure.remove_alternative_conformations()
    if remove_water:
        structure.remove_waters()
    return CIFObject(structure=structure, pdb_id=pdb_id)


def parse_mmcif(
    path: Path,
    remove_alt_conf: bool = False,
    remove_water: bool = True,
    pdb_id: str | None = None,
) -> CIFObject:
    return _parse_mmcif(
        path,
        remove_alt_conf=remove_alt_conf,
        remove_water=remove_water,
        pdb_id=pdb_id,
    )


def _create_branch_links_from_bonds(
    ligand: LigandObject, sorted_residue_ids: list[int]
) -> list:
    """Create IHM branch links from ligand bond information.

    This function analyzes inter-residue bonds to create branch links
    for oligosaccharides and other branched entities.

    Args:
        ligand: LigandObject instance containing bond information
        sorted_residue_ids: List of residue IDs in sequence order

    Returns:
        List of ihm.BranchLink objects representing inter-residue connections
    """
    branch_links = []

    # Create mapping from atom index to residue info
    atom_to_residue = {}
    for i, atom in enumerate(ligand.atoms):
        if atom["is_present"]:
            atom_to_residue[i] = {
                "residue_id": atom["residue_id"],
                "atom_name": "".join([chr(c + 32) for c in atom["name"] if c != 0]),
            }

    # Analyze bonds for inter-residue connections
    for bond in ligand.bonds:
        atom1_idx = bond["atom_1"]
        atom2_idx = bond["atom_2"]

        if atom1_idx in atom_to_residue and atom2_idx in atom_to_residue:
            res1_info = atom_to_residue[atom1_idx]
            res2_info = atom_to_residue[atom2_idx]

            # Check if this is an inter-residue bond
            if res1_info["residue_id"] != res2_info["residue_id"]:
                # Map residue IDs to sequence positions (1-based)
                try:
                    num1 = sorted_residue_ids.index(res1_info["residue_id"]) + 1
                    num2 = sorted_residue_ids.index(res2_info["residue_id"]) + 1

                    # Create branch link
                    # Note: We don't have leaving atom information, so we use the same atoms
                    branch_link = ihm.BranchLink(
                        num1=num1,
                        atom_id1=res1_info["atom_name"],
                        leaving_atom_id1=res1_info[
                            "atom_name"
                        ],  # Would need proper leaving atom
                        num2=num2,
                        atom_id2=res2_info["atom_name"],
                        leaving_atom_id2=res2_info[
                            "atom_name"
                        ],  # Would need proper leaving atom
                        order="sing",  # Assume single bond
                        details=f"Bond between residue {res1_info['residue_id']} and {res2_info['residue_id']}",
                    )
                    branch_links.append(branch_link)

                except ValueError:
                    # Residue ID not found in sorted list
                    logger.warning("Residue ID not found when creating branch link")
                    continue

    return branch_links


def to_mmcif(
    ligand: LigandObject,
    path: Path,
) -> None:
    """Write ligand structure to MMCIF file.

    Args:
        path: Path to output MMCIF file.
        ligand: LigandObject instance containing structure information.
    """
    # Create system
    system = System()

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Get residue names from residue_names field
    residue_names = ligand.residue_names
    class_name_str = (
        "-".join(residue_names) if len(residue_names) > 1 else residue_names[0]
    )

    # Analyze ligand to find unique residues and their sequence
    unique_residue_ids = set()
    for atom in ligand.atoms:
        if atom["is_present"]:
            unique_residue_ids.add(atom["residue_id"])

    # Handle edge case of no present atoms
    if not unique_residue_ids:
        logger.warning(f"No present atoms found in ligand {class_name_str}")
        unique_residue_ids = {1}  # Default to residue ID 1

    # Sort residue IDs to create consistent sequence
    sorted_residue_ids = sorted(unique_residue_ids)
    num_residues = len(sorted_residue_ids)

    logger.debug(f"Ligand {class_name_str}: Found residue IDs {sorted_residue_ids}")

    # Determine if this is a branched entity
    is_branched = num_residues > 1

    # Create entity for the ligand
    if num_residues == 1:
        residue_name = residue_names[0] if residue_names else class_name_str
        chem_comp = ihm.NonPolymerChemComp(id=residue_name)
        entity = Entity([chem_comp])
    else:
        # Multiple residues
        sequence = []
        for i, res_id in enumerate(sorted_residue_ids):
            # Use the actual residue name if available, otherwise use class_name
            residue_name = (
                residue_names[i]
                if i < len(residue_names)
                else f"{class_name_str}_{res_id}"
            )
            # Saccharide component
            chem_comp = ihm.SaccharideChemComp(id=residue_name)
            sequence.append(chem_comp)

        entity = Entity(sequence)

        # For branched entities, add branch links if available
        if is_branched:
            logger.debug(f"Ligand {class_name_str}: Detected as branched entity")
            # Try to create branch links from bond information
            try:
                branch_links = _create_branch_links_from_bonds(
                    ligand, sorted_residue_ids
                )
                if branch_links:
                    entity.branch_links = branch_links
                    logger.debug(f"Added {len(branch_links)} branch links to entity")
            except Exception as e:
                logger.warning(f"Failed to create branch links: {e}")

    # Create asym unit
    asym = AsymUnit(
        entity,
        details=f"Ligand {class_name_str}",
        id="A",  # Single chain A
    )

    # Note: sequence_range_for_model is not available in current modelcif version
    # The sequence range is automatically handled by the AsymUnit creation

    # Create assembly with single asym unit
    modeled_assembly = Assembly([asym], name="Modeled assembly")

    class _MyModel(AbInitioModel):
        def get_atoms(self) -> Iterator[Atom]:
            # Add all atom sites
            for atom in ligand.atoms:
                # Skip atoms that are not present
                if not atom["is_present"]:
                    continue

                # Get atom name
                name = atom["name"]
                name = "".join([chr(c + 32) for c in name if c != 0])

                # Use the original residue ID from the atom (matches dataset structure)
                seq_id = atom["residue_id"]

                # Get element symbol from atomic number
                element = periodic_table.GetElementSymbol(atom["element"].item())
                element = element.upper()

                # Get coordinates
                pos = atom["coords"]

                yield Atom(
                    asym_unit=asym,
                    type_symbol=element,
                    seq_id=seq_id,
                    atom_id=name,
                    x=f"{pos[0]:.5f}",
                    y=f"{pos[1]:.5f}",
                    z=f"{pos[2]:.5f}",
                    het=True,  # Ligand atoms are hetatoms
                    biso=100.0,  # Default B-factor
                    occupancy=1.0,
                )

    # Create model and add to system
    model = _MyModel(assembly=modeled_assembly, name="Model")
    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)

    # Write to file
    with open(path, "w") as f:
        dumper.write(f, [system])
