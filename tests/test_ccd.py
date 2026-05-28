"""Tests for CCD lookup and conformer generation."""

import pickle
from collections.abc import Callable
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from emap2lig.data import ccd as ccd_module
from emap2lig.data.ccd import (
    CCDFetchError,
    _migrate_legacy_bulk_dict,
    get_ccd_mol,
    get_conformer_from_smiles,
)


class _MockResponse:
    """Minimal mock for ``requests.Response`` used by RCSB fetch tests."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


@pytest.fixture
def fixture_cif_text() -> str:
    """CIF content of the A1CS4 test fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "A1CS4.cif"
    return fixture_path.read_text(encoding="utf-8")


def _mol_with_pdb_name(pdb_name: str) -> Mol:
    """Create a small RDKit molecule with the requested PDB name."""
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    mol.SetProp("PDB_NAME", pdb_name)
    return mol


def _mock_requests_get(
    cif_text: str, status_code: int = 200
) -> Callable[[str, int], _MockResponse]:
    """Return a ``requests.get`` mock that serves the fixture CIF."""

    def _get(url: str, timeout: int) -> _MockResponse:
        _ = (url, timeout)
        return _MockResponse(status_code, cif_text)

    return _get


def test_migrate_legacy_bulk_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ccd_dir = tmp_path / "ccd"
    legacy_dir = tmp_path / "models" / "ccd"
    legacy_dir.mkdir(parents=True)
    monkeypatch.setattr(ccd_module, "_CCD_DIR", ccd_dir)
    monkeypatch.setattr(ccd_module, "_LEGACY_CCD_DIR", legacy_dir)

    date = "250523"
    legacy_path = legacy_dir / f"ccd_dict_{date}.pkl"
    with legacy_path.open("wb") as f:
        pickle.dump({"FOO": "bar"}, f)

    _migrate_legacy_bulk_dict(date)

    assert not legacy_path.exists()
    assert (ccd_dir / f"ccd_dict_{date}.pkl").exists()


def test_migrate_legacy_no_op_when_target_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ccd_dir = tmp_path / "ccd"
    legacy_dir = tmp_path / "models" / "ccd"
    ccd_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    monkeypatch.setattr(ccd_module, "_CCD_DIR", ccd_dir)
    monkeypatch.setattr(ccd_module, "_LEGACY_CCD_DIR", legacy_dir)

    date = "250523"
    target_path = ccd_dir / f"ccd_dict_{date}.pkl"
    with target_path.open("wb") as f:
        pickle.dump({"FOO": "bar"}, f)
    legacy_path = legacy_dir / f"ccd_dict_{date}.pkl"
    with legacy_path.open("wb") as f:
        pickle.dump({"OLD": "old"}, f)

    _migrate_legacy_bulk_dict(date)

    assert legacy_path.exists()


def test_per_ccd_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_ccd_dir: Path,
) -> None:
    mol = _mol_with_pdb_name("FOO")
    with (tmp_ccd_dir / "FOO.pkl").open("wb") as handle:
        pickle.dump(mol, handle)
    monkeypatch.setattr(ccd_module, "_load_bulk_dict", lambda date="250523": {})

    resolved = get_ccd_mol("FOO")

    assert resolved is not None
    assert resolved.GetProp("PDB_NAME") == "FOO"


def test_bulk_dict_hit_does_not_write_per_ccd_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_ccd_dir: Path,
) -> None:
    mol = _mol_with_pdb_name("FOO")
    monkeypatch.setattr(
        ccd_module,
        "_load_bulk_dict",
        lambda date="250523": {"FOO": mol},
    )

    resolved = get_ccd_mol("FOO")

    assert resolved is mol
    assert not (tmp_ccd_dir / "FOO.pkl").exists()


def test_rcsb_cif_parse_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_ccd_dir: Path,
    fixture_cif_text: str,
) -> None:
    """pdbeccdutils parses CCD CIF with proper atom names and conformers."""
    monkeypatch.setattr(ccd_module, "_load_bulk_dict", lambda date="250523": {})
    monkeypatch.setattr(
        ccd_module.requests, "get", _mock_requests_get(fixture_cif_text)
    )
    resolved = get_ccd_mol("A1CS4")

    assert resolved is not None
    assert resolved.GetNumHeavyAtoms() == 10
    assert resolved.GetNumConformers() >= 1
    for atom in resolved.GetAtoms():
        assert atom.HasProp("name"), f"Atom {atom.GetIdx()} missing name"
        assert atom.GetProp("name") != ""
    assert (tmp_ccd_dir / "A1CS4.pkl").exists()


def test_rcsb_cif_404_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_ccd_dir: Path,
) -> None:
    """A 404 from RCSB should raise CCDFetchError."""
    monkeypatch.setattr(ccd_module, "_load_bulk_dict", lambda date="250523": {})
    monkeypatch.setattr(
        ccd_module.requests, "get", _mock_requests_get("", status_code=404)
    )
    with pytest.raises(CCDFetchError, match="status 404"):
        get_ccd_mol("ZZZZ")


def test_empty_code_raises() -> None:
    with pytest.raises(CCDFetchError, match="Empty CCD code"):
        get_ccd_mol("")


def test_smiles_conformer() -> None:
    result, mol = get_conformer_from_smiles("CCO")

    assert result in {"computed", "ideal"}
    assert mol.GetNumConformers() >= 1
    atom_names = [
        atom.GetProp("name") for atom in mol.GetAtoms() if atom.HasProp("name")
    ]
    assert len(atom_names) == mol.GetNumAtoms()


@pytest.mark.network
def test_rcsb_live_a1cs4(monkeypatch: pytest.MonkeyPatch, tmp_ccd_dir: Path) -> None:
    monkeypatch.setattr(ccd_module, "_load_bulk_dict", lambda date="250523": {})
    resolved = get_ccd_mol("A1CS4")
    assert resolved is not None
    assert resolved.GetNumAtoms() > 0
    assert (tmp_ccd_dir / "A1CS4.pkl").exists()
