num_elements = 128

####################################################################################################
# ATOMS
####################################################################################################

chirality_types = [
    "CHI_OTHER",
    "CHI_OCTAHEDRAL",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_SQUAREPLANAR",
]
chirality_type_ids = {chirality: i for i, chirality in enumerate(chirality_types)}

bond_types = [
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "DATIVE",
    "AROMATIC",
]
bond_type_ids = {bond: i for i, bond in enumerate(bond_types)}
