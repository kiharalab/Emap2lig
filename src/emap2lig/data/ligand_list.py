"""Ligand list files: YAML (or one-line plain text) ↔ :class:`LigandRecord`.

Supported entries::

    - CCD: HEM                      # or the string form "CCD:HEM"
    - SMILES: CC(=O)O               # or "SMILES:CC(=O)O"
    - BRANCHED:
        residues: ["1. NAG", "2. NAG"]
        bonds: [[2, "C1", 1, "O4"]]
      blob_id: [1, 2]               # optional, on any entry (alias: blobs)

CCD records are named by their code; SMILES and BRANCHED records are named
``LIG1``, ``LIG2``, ... in file order.
"""

import itertools
import logging
from collections.abc import Iterator
from pathlib import Path

import yaml

from .types import LigandRecord

logger = logging.getLogger(__name__)

_STRING_PREFIXES = ("CCD", "SMILES")


def parse_ligand_list(path: Path | str) -> list[LigandRecord]:
    """Parse a ligand list file; unrecognized entries are skipped with a warning."""
    content = Path(path).read_text().strip()
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        logger.info(f"YAML parsing failed, trying plain text format for {path}")
        return _parse_plain_text(content, path)

    items = data if isinstance(data, list) else ([] if data is None else [data])
    names = (f"LIG{i}" for i in itertools.count(1))
    records = []
    for item in items:
        record = _parse_entry(item, names)
        if record is None:
            logger.warning(f"Unrecognized ligand entry: {item}")
        else:
            records.append(record)
    return records


def write_ligand_list(records: list[LigandRecord], path: Path | str) -> Path:
    """Write *records* as a YAML ligand list readable by :func:`parse_ligand_list`."""
    items = [_record_to_entry(record) for record in records]
    path = Path(path)
    with open(path, "w") as f:
        yaml.safe_dump(items, f, sort_keys=False)
    return path


def _parse_plain_text(content: str, path: Path | str) -> list[LigandRecord]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) == 1:
        record = _parse_entry(lines[0], (f"LIG{i}" for i in itertools.count(1)))
        if record is not None:
            return [record]
    logger.error(
        f"Unsupported file format for {path}. Supported formats: YAML files with "
        "lists/single items, or plain text with a single CCD:/SMILES: entry."
    )
    return []


def _parse_entry(item: object, names: Iterator[str]) -> LigandRecord | None:
    if isinstance(item, str):
        key, sep, value = item.partition(":")
        if not sep or key not in _STRING_PREFIXES:
            return None
        item = {key: value.strip()}
    if not isinstance(item, dict):
        return None

    blobs = item.get("blob_id", item.get("blobs"))
    if blobs is not None and not isinstance(blobs, list):
        blobs = [blobs]

    if "BRANCHED" in item:
        branched = item["BRANCHED"]
        residues = dict(enumerate(_residue_names(branched.get("residues", [])), 1))
        return LigandRecord(
            type="BRANCHED",
            name=next(names),
            blobs=blobs,
            residues=residues,
            bonds=branched.get("bonds", []),
        )
    if "CCD" in item:
        return LigandRecord(type="CCD", name=item["CCD"], blobs=blobs)
    if "SMILES" in item:
        return LigandRecord(
            type="SMILES", name=next(names), blobs=blobs, smiles=item["SMILES"]
        )
    return None


def _residue_names(entries: list[str]) -> list[str]:
    """``["1. NAG", "2. MAN"]`` → ``["NAG", "MAN"]`` (malformed entries dropped)."""
    names = []
    for entry in entries:
        parts = entry.strip().split(".")
        if len(parts) >= 2:
            names.append(parts[1].strip())
    return names


def _record_to_entry(record: LigandRecord) -> dict:
    if record.type == "CCD":
        entry: dict = {"CCD": record.name}
    elif record.type == "SMILES":
        entry = {"SMILES": record.smiles}
    else:
        residues = sorted((record.residues or {}).items())
        branched: dict = {"residues": [f"{idx}. {name}" for idx, name in residues]}
        if record.bonds:
            branched["bonds"] = [list(bond) for bond in record.bonds]
        entry = {"BRANCHED": branched}
    if record.blobs:
        entry["blob_id"] = list(record.blobs)
    return entry
