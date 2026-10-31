"""Ligand structure modeling from cryo-EM maps.

Typical use::

    from emap2lig import (
        InferenceConfig,
        detect_ligand_objects,
        parse_ligand_list,
        run_structure_modeling,
    )

    cfg = InferenceConfig(gpu=0, multiplicity=4)
    detect_ligand_objects("map.mrc", "out", cfg)
    run_structure_modeling("out", parse_ligand_list("ligands.yaml"), cfg)

Exports are imported on first access so ``import emap2lig`` stays cheap.
"""

from importlib import import_module

_EXPORTS = {
    "InferenceConfig": "emap2lig.config",
    "OutputLayout": "emap2lig.layout",
    "Emap2ligError": "emap2lig.errors",
    "NoBlobsDetectedError": "emap2lig.errors",
    "TooManyBlobsError": "emap2lig.errors",
    "LigandPreparationError": "emap2lig.errors",
    "JobCancelledError": "emap2lig.errors",
    "LigandRecord": "emap2lig.data.types",
    "parse_ligand_list": "emap2lig.data.ligand_list",
    "write_ligand_list": "emap2lig.data.ligand_list",
    "prepare_ligands": "emap2lig.data.featurize",
    "detect_ligand_objects": "emap2lig.pipeline",
    "run_structure_modeling": "emap2lig.pipeline",
    "run_fragment_detection": "emap2lig.pipeline",
    "rank_conformers": "emap2lig.pipeline",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
