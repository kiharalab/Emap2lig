"""Data types, IO and chemistry for Emap2lig."""

from rdkit import Chem

# Keep atom properties (e.g. atom names) when RDKit molecules are pickled,
# both for the cached CCD dictionary and for DataLoader workers.
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
